"""Synthetic check of the discriminator (python -m pytest tests/ or python tests/test_fisher_discriminator.py).

DGP of the paper's Section 5.1: m=20, r=3, sigma^2=0.25, 30 training batches of 80.
The target of the score is the risk gap of the deployed plug-in GLS estimator
(raw risk - quotient risk), measured here on fresh test batches. The corrected score
must track it (up to the anisotropy that per-feature standardization introduces,
which Assumption 4.1 ignores); the paper's per-batch score is shifted up by
K(Sigma_nu + sigma^2/n I)K'.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ResidualBatchPCA import ResidualBatchPCA            # noqa: E402
from scripts.fisher_discriminator import fisher_discriminator, calibration_from_rbpca  # noqa: E402

M, R, S2 = 20, 3, 0.25


def _sample(rng, A, j, n_b, n, shift):
    q = rng.standard_normal((n_b, n))
    nu = rng.standard_normal((n_b, R)) + shift
    Y = q[..., None] * j + (nu @ A.T)[:, None, :] + np.sqrt(S2) * rng.standard_normal((n_b, n, M))
    return q.ravel(), Y.reshape(-1, M), np.repeat(np.arange(n_b), n)


def _one(rep, alpha=0.3, delta=0.0):
    rng = np.random.default_rng(rep)
    Q, _ = np.linalg.qr(rng.standard_normal((M, M)))
    A = Q[:, :R]
    j = np.sqrt(alpha) * A[:, 0] + np.sqrt(1 - alpha) * Q[:, R]
    shift = np.array([delta, 0.0, 0.0])
    qt, Yt, bt = _sample(rng, A, j, 30, 80, 0.0)
    qv, Yv, bv = _sample(rng, A, j, 10, 80, shift)
    qs, Ys, _ = _sample(rng, A, j, 40, 100, shift)
    X = np.column_stack([np.ones_like(qt), qt])
    rb = ResidualBatchPCA(crossfit="none", ridge_alpha=1e-6).fit(Yt, X, bt)
    res = fisher_discriminator(rb, X, bt, lambda q: np.column_stack([np.ones_like(q), q]),
                               qv, Yv, bv, (qt.min(), qt.max()), n_boot=20, random_state=rep)
    # deployed plug-in GLS estimator on the standardized scale
    cal = calibration_from_rbpca(rb, bt)
    m0, jh = rb.forward_model_.coef_[:, 0], rb.forward_model_.coef_[:, 1]
    Ah, P, s2 = cal["A"], rb.P_hat_, cal["sigma2"]
    Sy_inv = P / s2 + Ah @ np.linalg.inv(cal["Sigma_nu"] + s2 * np.eye(cal["rank"])) @ Ah.T
    Z = rb.scaler_.transform(Ys) - m0
    q_raw = Z @ (Sy_inv @ jh) / (jh @ Sy_inv @ jh)
    q_quo = Z @ (P @ jh) / (jh @ P @ jh)
    gap = np.mean((q_raw - qs) ** 2) - np.mean((q_quo - qs) ** 2)
    return dict(res.macro, gap=gap)


def test_tracks_deployed_gap():
    for delta in (0.0, 2.0):
        out = [_one(r, delta=delta) for r in range(80)]
        gap = np.mean([o["gap"] for o in out])
        corr = np.mean([o["psi_corr"] for o in out])
        paper = np.mean([o["psi_paper"] for o in out])
        bias = np.mean([o["bias"] for o in out])
        # the corrected score is much closer to the deployed gap than the paper's score
        assert abs(corr - gap) < 0.5 * abs(paper - gap), (delta, corr, paper, gap)
        assert paper - corr > 0.9 * bias, (delta, paper, corr, bias)
        print(f"delta={delta}: gap={gap:.4f}  corrected={corr:.4f}  paper={paper:.4f}")


if __name__ == "__main__":
    test_tracks_deployed_gap()
    print("ok")
