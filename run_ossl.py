from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
)

from datasets.ossl import (
    load_ossl,
    build_ossl_forward_design,
    make_equal_source_weights,
    mean_pool_mir_426_to_71,
    describe_ossl,
)

from scripts.ResidualBatchPCA import (
    ResidualBatchPCA,
)

from scripts.config import (
    DEVICE,
    SEEDS,
    train_config,
    configs_for_dataset,
)

from scripts.training_utils import (
    set_seed,
    build_torch_model,
    fit_torch_regressor,
    evaluate_torch,
    train_xgboost,
    evaluate_xgboost,
)


BACKBONES = [
    "resnet",
    "ft_transformer",
    "tabm",
    "xgboost",
]


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    X,
    y,
    batch_size,
    shuffle,
    sample_weight=None,
):

    dataset = TensorDataset(
        torch.as_tensor(
            X,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y,
            dtype=torch.float32,
        ),
    )

    if sample_weight is not None:

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                sample_weight,
                dtype=torch.double,
            ),
            num_samples=len(
                sample_weight
            ),
            replacement=True,
        )

        shuffle = False

    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(
            shuffle
            if sampler is None
            else False
        ),
        sampler=sampler,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )


# ============================================================
# Helpers
# ============================================================

def parse_list_or_all(
    value,
    allowed,
    cast=str,
):
    if str(value).lower() == "all":
        return list(allowed)

    return [
        cast(item.strip())
        for item in str(value).split(",")
        if item.strip()
    ]


def save_rows(
    path,
    rows,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    write_header = (
        not path.exists()
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()

        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "OSSL MIR organic-carbon "
            "raw-versus-quotient regression."
        )
    )

    # Either use all-L1
    # or MIR-L0 + soil-lab-L1.

    source_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    source_group.add_argument(
        "--all-l1",
        help=(
            "Path to "
            "ossl_all_L1_v1.2.csv.gz"
        ),
    )

    source_group.add_argument(
        "--mir-l0",
        help=(
            "Path to "
            "ossl_mir_L0_v1.2.csv.gz"
        ),
    )

    parser.add_argument(
        "--soillab-l1",
        help=(
            "Path to "
            "ossl_soillab_L1_v1.2.csv.gz. "
            "Required when --mir-l0 is used."
        ),
    )

    parser.add_argument(
        "--backbone",
        default="all",
    )

    parser.add_argument(
        "--seed",
        default="all",
    )

    parser.add_argument(
        "--representation",
        choices=[
            "raw",
            "quotient",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config.epochs,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=train_config.patience,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=train_config.batch_size,
    )

    parser.add_argument(
        "--output",
        default="results/ossl_results.csv",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.mir_l0 is not None
        and args.soillab_l1 is None
    ):
        parser.error(
            "--soillab-l1 is required "
            "when --mir-l0 is used."
        )

    # ========================================================
    # 1. Load OSSL
    # ========================================================

    data = load_ossl(
        all_l1_path=args.all_l1,
        mir_l0_path=args.mir_l0,
        soillab_l1_path=(
            args.soillab_l1
        ),
    )

    print(
        "OSSL data summary:"
    )

    print(
        describe_ossl(data)
    )

    # ========================================================
    # 2. Organic-carbon forward spline
    # ========================================================

    (
        design,
        X_forward_train,
        X_forward_valid,
        X_forward_test,
    ) = build_ossl_forward_design(
        data,
        n_knots=7,
        degree=3,
    )

    # ========================================================
    # 3. Equal-source weights
    # ========================================================

    source_weights = (
        make_equal_source_weights(
            data.train.source
        )
    )

    # ========================================================
    # 4. Residual Batch PCA
    #
    # batch = source/laboratory
    # ========================================================

    rbpca = ResidualBatchPCA(
        rank=None,
        variance_threshold=0.90,
        ridge_alpha=1.0,
        crossfit="none",
    )

    rbpca.fit(
        Y=data.train.Y,
        X_forward=X_forward_train,
        batch=data.train.source,
        sample_weight=source_weights,
    )

    print(
        "Residual Batch PCA diagnostics:"
    )

    print(
        rbpca.diagnostics()
    )

    # ========================================================
    # 5. Raw / quotient in ORIGINAL 426D MIR space
    # ========================================================

    representations_426 = {

        "raw": (
            rbpca.transform_raw(
                data.train.Y
            ),
            rbpca.transform_raw(
                data.valid.Y
            ),
            rbpca.transform_raw(
                data.test.Y
            ),
        ),

        "quotient": (
            rbpca.transform(
                data.train.Y
            ),
            rbpca.transform(
                data.valid.Y
            ),
            rbpca.transform(
                data.test.Y
            ),
        ),
    }

    # ========================================================
    # 6. Deterministic pooling
    #
    # IMPORTANT:
    # quotient happens BEFORE pooling.
    #
    # 426 -> 71
    # ========================================================

    representations = {}

    for name, values in (
        representations_426.items()
    ):

        representations[name] = tuple(
            mean_pool_mir_426_to_71(
                X
            )
            for X in values
        )

    # ========================================================
    # 7. Representation choices
    # ========================================================

    if args.representation == "both":

        representation_names = [
            "raw",
            "quotient",
        ]

    else:

        representation_names = [
            args.representation
        ]

    # ========================================================
    # 8. Backbones
    # ========================================================

    backbones = parse_list_or_all(
        args.backbone,
        BACKBONES,
    )

    unknown = sorted(
        set(backbones)
        - set(BACKBONES)
    )

    if unknown:
        raise ValueError(
            f"Unknown backbones: {unknown}"
        )

    # ========================================================
    # 9. Seeds
    # ========================================================

    seeds = parse_list_or_all(
        args.seed,
        SEEDS,
        cast=int,
    )

    if args.smoke:

        backbones = backbones[:1]
        seeds = seeds[:1]

        args.epochs = min(
            args.epochs,
            3,
        )

        args.patience = min(
            args.patience,
            2,
        )

    # ========================================================
    # 10. Model configs
    # ========================================================

    configs = configs_for_dataset(
        "ossl"
    )

    # ========================================================
    # 11. Experiments
    # ========================================================

    for backbone in backbones:

        for seed in seeds:

            for representation in (
                representation_names
            ):

                print()

                print(
                    "[OSSL] "
                    f"backbone={backbone} "
                    f"seed={seed} "
                    f"representation={representation}"
                )

                set_seed(seed)

                (
                    X_train,
                    X_valid,
                    X_test,
                ) = representations[
                    representation
                ]

                # ============================================
                # XGBoost
                # ============================================

                if backbone == "xgboost":

                    model = train_xgboost(
                        X_train,
                        data.train.q,
                        X_valid,
                        data.valid.q,
                        config=configs[
                            "xgboost"
                        ],
                        seed=seed,
                        sample_weight=(
                            source_weights
                        ),
                    )

                    test_metrics = (
                        evaluate_xgboost(
                            model,
                            X_test,
                            data.test.q,
                            return_predictions=True,
                        )
                    )

                # ============================================
                # Neural backbones
                # ============================================

                else:

                    # Weighted sampling ensures that large
                    # sources do not dominate training.

                    train_loader = (
                        make_loader(
                            X_train,
                            data.train.q,
                            args.batch_size,
                            shuffle=False,
                            sample_weight=(
                                source_weights
                            ),
                        )
                    )

                    valid_loader = (
                        make_loader(
                            X_valid,
                            data.valid.q,
                            args.batch_size,
                            shuffle=False,
                        )
                    )

                    test_loader = (
                        make_loader(
                            X_test,
                            data.test.q,
                            args.batch_size,
                            shuffle=False,
                        )
                    )

                    model = (
                        build_torch_model(
                            backbone,
                            configs[
                                backbone
                            ],
                        )
                    )

                    fit_torch_regressor(
                        backbone,
                        model,
                        train_loader,
                        valid_loader,
                        epochs=args.epochs,
                        patience=args.patience,
                        device=DEVICE,
                    )

                    test_metrics = (
                        evaluate_torch(
                            backbone,
                            model,
                            test_loader,
                            device=DEVICE,
                            return_predictions=True,
                        )
                    )

                # ============================================
                # Save results
                # ============================================

                row = {

                    "dataset":
                        "OSSL_OC",

                    "backbone":
                        backbone,

                    "seed":
                        seed,

                    "representation":
                        representation,

                    "rbpca_rank":
                        rbpca.selected_rank_,

                    "rbpca_cumvar":
                        float(
                            rbpca
                            .cumulative_explained_variance_[
                                rbpca.selected_rank_
                                - 1
                            ]
                        ),

                    "test_mse":
                        float(
                            test_metrics["mse"]
                        ),

                    "test_rmse":
                        float(
                            test_metrics["rmse"]
                        ),

                    "test_mae":
                        float(
                            test_metrics["mae"]
                        ),
                }

                print(row)

                save_rows(
                    args.output,
                    [row],
                )

    print()

    print(
        f"Saved results to {args.output}"
    )


if __name__ == "__main__":
    main()
