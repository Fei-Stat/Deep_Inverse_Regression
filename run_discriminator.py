"""
Fisher discriminator for FD004 and OSSL, with the paper's per-batch score and
the bias-corrected score side by side.

The calibration (standardization, forward design, Residual Batch PCA) is the
same as in run_fd004.py / run_ossl.py, so the nuisance subspace matches the
one used for the prediction experiments. Only validation batches are scored;
test batches are scored separately as a post-selection audit (--audit-test).

Examples
--------
python run_discriminator.py ossl --all-l1 data/ossl_all_L1_v1.2.csv.gz
python run_discriminator.py fd004 --data-dir data/CMAPSS \
    --valid-engine-ids-file fd004_valid_engines_COUNT_MATCH_ONLY_seed277.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.ResidualBatchPCA import ResidualBatchPCA
from scripts.fisher_discriminator import fisher_discriminator, worst_case


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_ossl(args):
    from datasets.ossl import load_ossl, build_ossl_forward_design, make_equal_source_weights

    data = load_ossl(all_l1_path=args.all_l1, mir_l0_path=args.mir_l0,
                     soillab_l1_path=args.soillab_l1)
    design, Xf_train, _, _ = build_ossl_forward_design(data, n_knots=7, degree=3)
    w = make_equal_source_weights(data.train.source)
    rbpca = ResidualBatchPCA(rank=None, variance_threshold=0.90, ridge_alpha=1.0,
                             crossfit="none", centroid_shrinkage_kappa=0.0)
    rbpca.fit(Y=data.train.Y, X_forward=Xf_train, batch=data.train.source, sample_weight=w)
    print("RB-PCA:", {k: v for k, v in rbpca.diagnostics().items()
                      if k in ("n_batches", "selected_rank", "maximum_rank")},
          "cum. explained:", np.round(rbpca.cumulative_explained_variance_, 4).tolist())
    q_range = (float(data.train.q.min()), float(data.train.q.max()))

    out, valid_res = {}, None
    parts = [("valid", data.valid)] + ([("test_audit", data.test)] if args.audit_test else [])
    for name, part in parts:
        res = fisher_discriminator(
            rbpca, Xf_train, data.train.source, design.transform,
            part.q, part.Y, part.source, q_range,
            sample_weight_train=w, n_boot=args.n_boot, random_state=args.seed)
        print(f"\n=== OSSL {name} ===\n{res.summary()}")
        out[name] = dict(per_batch=res.per_batch, macro=res.macro)
        valid_res = valid_res or res
    return out, valid_res


def run_fd004(args):
    from datasets.fd004 import load_fd004, build_fd004_forward_design, read_engine_ids_file

    ids = read_engine_ids_file(args.valid_engine_ids_file) if args.valid_engine_ids_file else None
    d = Path(args.data_dir)
    data = load_fd004(train_path=d / "train_FD004.txt", test_path=d / "test_FD004.txt",
                      rul_path=d / "RUL_FD004.txt", n_valid_engines=50,
                      split_seed=args.split_seed, valid_engine_ids=ids,
                      rul_cap=125.0, window_length=20, stride=2)
    design, Xf_train, _, _ = build_fd004_forward_design(
        data, n_regimes=6, n_knots=5, spline_degree=3, random_state=args.split_seed)
    rbpca = ResidualBatchPCA(rank=None, variance_threshold=0.90, ridge_alpha=1.0,
                             crossfit="group_kfold", n_splits=5, random_state=args.split_seed,
                             centroid_shrinkage_kappa=args.centroid_shrinkage_kappa)
    rbpca.fit(Y=data.train.Y, X_forward=Xf_train, batch=data.train.engine_id)
    print("RB-PCA:", {k: v for k, v in rbpca.diagnostics().items()
                      if k in ("n_batches", "selected_rank")},
          "cum. explained:", np.round(rbpca.cumulative_explained_variance_[:6], 4).tolist())
    q_range = (float(data.train.q.min()), float(data.train.q.max()))

    out, valid_res = {}, None
    parts = [("valid", data.valid)] + ([("test_audit", data.test)] if args.audit_test else [])
    for name, part in parts:
        res = fisher_discriminator(
            rbpca, Xf_train, data.train.engine_id,
            lambda q, s=part.settings: design.transform(q, s),
            part.q, part.Y, part.engine_id, q_range,
            Y_train=data.train.Y, n_boot=args.n_boot, random_state=args.seed)
        m = res.macro
        print(f"\n=== FD004 {name}: {len(res.per_batch)} engines ===")
        print(f"G={m['G']:.2f}  C={m['C']:.2f}  bias={m['bias']:.2f}")
        print(f"Psi_paper={m['psi_paper']:.2f}  Psi_corr={m['psi_corr']:.2f}")
        if "psi_corr_batch_lo" in m:
            print(f"corrected, engine-bootstrap 95% CI: [{m['psi_corr_batch_lo']:.2f}, {m['psi_corr_batch_hi']:.2f}]")
        n_pos_p = sum(r["psi_paper"] > 0 for r in res.per_batch.values())
        n_pos_c = sum(r["psi_corr"] > 0 for r in res.per_batch.values())
        print(f"engines with positive score: paper {n_pos_p}, corrected {n_pos_c}")
        out[name] = dict(macro=m, n_pos_paper=n_pos_p, n_pos_corr=n_pos_c,
                         per_batch=res.per_batch)
        valid_res = valid_res or res
    return out, valid_res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="dataset", required=True)
    o = sub.add_parser("ossl")
    g = o.add_mutually_exclusive_group(required=True)
    g.add_argument("--all-l1"); g.add_argument("--mir-l0")
    o.add_argument("--soillab-l1")
    f = sub.add_parser("fd004")
    f.add_argument("--data-dir", required=True)
    f.add_argument("--valid-engine-ids-file")
    f.add_argument("--split-seed", type=int, default=42)
    f.add_argument("--centroid-shrinkage-kappa", type=float, default=0.0)
    for p in (o, f):
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--audit-test", action="store_true",
                       help="also score the test batches (post-selection audit only)")
        p.add_argument("--rho", type=float, default=None,
                       help="a-priori bound on the deployment nuisance shift, for Psi_wc")
        p.add_argument("--output", default=None)
    args = ap.parse_args()

    out, valid_res = run_ossl(args) if args.dataset == "ossl" else run_fd004(args)
    if args.rho is not None:
        out["worst_case"] = worst_case(valid_res, args.rho)
        print("\nworst case:", out["worst_case"])
    path = Path(args.output or f"results/discriminator_{args.dataset}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(out), indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
