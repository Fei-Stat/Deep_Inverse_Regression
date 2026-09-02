from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets.fd004 import (
    load_fd004,
    build_fd004_forward_design,
    append_standardized_settings,
    describe_fd004,
)

from scripts.ResidualBatchPCA import ResidualBatchPCA

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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Test-engine endpoint mask
# ============================================================

def endpoint_mask(
    engine_id,
    cycle,
):
    """
    Select the final observed window for every test engine.
    """

    engine_id = np.asarray(engine_id)
    cycle = np.asarray(cycle)

    mask = np.zeros(
        len(engine_id),
        dtype=bool,
    )

    for engine in np.unique(engine_id):

        index = np.flatnonzero(
            engine_id == engine
        )

        last_index = index[
            np.argmax(cycle[index])
        ]

        mask[last_index] = True

    return mask


# ============================================================
# CLI helpers
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

    write_header = not path.exists()

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
            "FD004 raw-versus-quotient "
            "RUL regression experiment."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Directory containing "
            "train_FD004.txt, "
            "test_FD004.txt and "
            "RUL_FD004.txt."
        ),
    )

    parser.add_argument(
        "--backbone",
        default="all",
        help=(
            "resnet, ft_transformer, "
            "tabm, xgboost, or all."
        ),
    )

    parser.add_argument(
        "--seed",
        default="all",
        help=(
            "Single seed, comma-separated "
            "seeds, or all."
        ),
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
        "--split-seed",
        type=int,
        default=42,
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
        default="results/fd004_results.csv",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Quick pipeline test: "
            "one backbone, one seed, "
            "at most three epochs."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # 1. Load FD004
    # ========================================================

    data_dir = Path(
        args.data_dir
    )

    data = load_fd004(
        train_path=(
            data_dir
            / "train_FD004.txt"
        ),
        test_path=(
            data_dir
            / "test_FD004.txt"
        ),
        rul_path=(
            data_dir
            / "RUL_FD004.txt"
        ),
        n_valid_engines=50,
        split_seed=args.split_seed,
        rul_cap=125.0,
        window_length=20,
        stride=2,
    )

    print(
        "FD004 data summary:"
    )
    print(
        describe_fd004(data)
    )

    # ========================================================
    # 2. Construct forward regression design
    #
    # spline(RUL)
    # + settings
    # + operating regimes
    # + spline(RUL) x regime
    # ========================================================

    (
        design,
        X_forward_train,
        X_forward_valid,
        X_forward_test,
    ) = build_fd004_forward_design(
        data,
        n_regimes=6,
        n_knots=5,
        spline_degree=3,
        random_state=args.split_seed,
    )

    # ========================================================
    # 3. Residual Batch PCA
    #
    # batch = engine_id
    # ========================================================

    rbpca = ResidualBatchPCA(
        rank=None,
        variance_threshold=0.90,
        ridge_alpha=1.0,
        crossfit="group_kfold",
        n_splits=5,
        random_state=args.split_seed,
    )

    rbpca.fit(
        Y=data.train.Y,
        X_forward=X_forward_train,
        batch=data.train.engine_id,
    )

    print(
        "Residual Batch PCA diagnostics:"
    )
    print(
        rbpca.diagnostics()
    )

    # ========================================================
    # 4. Construct raw and quotient representations
    #
    # RB-PCA only acts on the 63 sensor features.
    # Settings are added back afterwards.
    # ========================================================

    representations = {
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

    # Append standardized operating settings.
    #
    # 63 sensor features + 3 settings = 66 dimensions.

    for name, (
        Y_train,
        Y_valid,
        Y_test,
    ) in list(
        representations.items()
    ):

        representations[name] = (

            append_standardized_settings(
                Y_train,
                data.train.settings,
                design,
            ),

            append_standardized_settings(
                Y_valid,
                data.valid.settings,
                design,
            ),

            append_standardized_settings(
                Y_test,
                data.test.settings,
                design,
            ),
        )

    # ========================================================
    # 5. Choose representations
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
    # 6. Choose backbones
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
    # 7. Choose seeds
    # ========================================================

    seeds = parse_list_or_all(
        args.seed,
        SEEDS,
        cast=int,
    )

    # Smoke test
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
    # 8. Backbone configurations
    # ========================================================

    configs = configs_for_dataset(
        "fd004"
    )

    # Test endpoint for each engine
    end_mask = endpoint_mask(
        data.test.engine_id,
        data.test.cycle,
    )

    # ========================================================
    # 9. Run experiments
    # ========================================================

    rows = []

    for backbone in backbones:

        for seed in seeds:

            for representation in representation_names:

                print()
                print(
                    "[FD004] "
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

                    model = build_torch_model(
                        backbone,
                        configs[backbone],
                    )

                    train_loader = (
                        make_loader(
                            X_train,
                            data.train.q,
                            args.batch_size,
                            True,
                        )
                    )

                    valid_loader = (
                        make_loader(
                            X_valid,
                            data.valid.q,
                            args.batch_size,
                            False,
                        )
                    )

                    test_loader = (
                        make_loader(
                            X_test,
                            data.test.q,
                            args.batch_size,
                            False,
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
                # Test metrics
                # ============================================

                y_true = np.asarray(
                    test_metrics["y_true"]
                )

                y_pred = np.asarray(
                    test_metrics["y_pred"]
                )

                endpoint_mse = float(
                    np.mean(
                        (
                            y_pred[end_mask]
                            - y_true[end_mask]
                        )
                        ** 2
                    )
                )

                endpoint_rmse = float(
                    np.sqrt(
                        endpoint_mse
                    )
                )

                row = {
                    "dataset":
                        "FD004",

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

                    "test_all_mse":
                        float(
                            test_metrics["mse"]
                        ),

                    "test_all_rmse":
                        float(
                            test_metrics["rmse"]
                        ),

                    "test_endpoint_mse":
                        endpoint_mse,

                    "test_endpoint_rmse":
                        endpoint_rmse,
                }

                print(row)

                rows.append(row)

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
