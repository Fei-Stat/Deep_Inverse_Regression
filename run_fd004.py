from __future__ import annotations

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from datasets.fd004 import (
    load_fd004,
    read_engine_ids_file,
    build_fd004_forward_design,
    append_standardized_settings,
    describe_fd004,
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

PAPER_WINDOW_COUNTS = {
    "train_windows": 23034,
    "valid_windows": 5417,
    "test_windows": 18429,
}


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
        pin_memory=(
            torch.cuda.is_available()
        ),
    )


def endpoint_mask(
    engine_id,
    cycle,
):
    engine_id = np.asarray(
        engine_id
    )

    cycle = np.asarray(
        cycle
    )

    mask = np.zeros(
        len(engine_id),
        dtype=bool,
    )

    for engine in np.unique(
        engine_id
    ):
        idx = np.flatnonzero(
            engine_id == engine
        )

        last = idx[
            np.argmax(
                cycle[idx]
            )
        ]

        mask[last] = True

    return mask


def parse_list_or_all(
    value,
    allowed,
    cast=str,
):
    if str(value).lower() == "all":
        return list(allowed)

    return [
        cast(item.strip())
        for item
        in str(value).split(",")
        if item.strip()
    ]


def save_row(
    path,
    row,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not path.exists()
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            ),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            row
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "C-MAPSS FD004 raw-versus-quotient RUL regression."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
    )

    parser.add_argument(
        "--valid-engine-ids-file",
        default=None,
        help=(
            "Text file containing the fixed 50 validation engine IDs. "
            "Use this for exact paper reproduction."
        ),
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help=(
            "Exploratory fallback only when no fixed validation-ID file "
            "is supplied."
        ),
    )

    parser.add_argument(
        "--require-paper-split",
        action="store_true",
        help=(
            "Fail unless --valid-engine-ids-file is supplied and the "
            "window counts match the recorded 23034/5417/18429."
        ),
    )

    parser.add_argument(
        "--centroid-shrinkage-kappa",
        type=float,
        default=0.0,
        help=(
            "Explicit RB-PCA centroid shrinkage kappa. "
            "0 disables shrinkage. The historical paper coefficient "
            "was not preserved, so do not invent one silently."
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
        default="results/fd004_results.csv",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    valid_engine_ids = None

    if (
        args.valid_engine_ids_file
        is not None
    ):
        valid_engine_ids = (
            read_engine_ids_file(
                args.valid_engine_ids_file
            )
        )

    elif args.require_paper_split:
        parser.error(
            "--require-paper-split requires "
            "--valid-engine-ids-file."
        )

    else:
        warnings.warn(
            "No fixed validation-engine file was supplied. "
            "The run will use an exploratory random 199/50 engine split "
            "and is NOT guaranteed to reproduce the recorded FD004 paper split."
        )

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
        split_seed=(
            args.split_seed
        ),
        valid_engine_ids=(
            valid_engine_ids
        ),
        rul_cap=125.0,
        window_length=20,
        stride=2,
    )

    summary = describe_fd004(
        data
    )

    print(
        "FD004 data summary:"
    )
    print(
        summary
    )

    counts_match = all(
        summary[key]
        == value
        for key, value
        in PAPER_WINDOW_COUNTS.items()
    )

    print(
        "Recorded paper window counts match:",
        counts_match,
    )

    if (
        args.require_paper_split
        and not counts_match
    ):
        raise RuntimeError(
            "The supplied validation-engine IDs do not reproduce "
            "the recorded 23034/5417/18429 window counts."
        )

    (
        design,
        X_forward_train,
        _,
        _,
    ) = build_fd004_forward_design(
        data,
        n_regimes=6,
        n_knots=5,
        spline_degree=3,
        random_state=(
            args.split_seed
        ),
    )

    rbpca = ResidualBatchPCA(
        rank=None,
        variance_threshold=0.90,
        ridge_alpha=1.0,
        crossfit="group_kfold",
        n_splits=5,
        random_state=(
            args.split_seed
        ),
        centroid_shrinkage_kappa=(
            args.centroid_shrinkage_kappa
        ),
    )

    rbpca.fit(
        Y=data.train.Y,
        X_forward=(
            X_forward_train
        ),
        batch=(
            data.train.engine_id
        ),
    )

    print(
        "RB-PCA diagnostics:"
    )

    print(
        rbpca.diagnostics()
    )

    reps = {
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

    for name, (
        Y_train,
        Y_valid,
        Y_test,
    ) in list(
        reps.items()
    ):
        reps[name] = (
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

    representation_names = (
        [
            "raw",
            "quotient",
        ]
        if args.representation
        == "both"
        else [
            args.representation
        ]
    )

    backbones = parse_list_or_all(
        args.backbone,
        BACKBONES,
    )

    unknown = (
        set(backbones)
        - set(BACKBONES)
    )

    if unknown:
        raise ValueError(
            f"Unknown backbones: {sorted(unknown)}"
        )

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

    configs = configs_for_dataset(
        "fd004"
    )

    end_mask = endpoint_mask(
        data.test.engine_id,
        data.test.cycle,
    )

    if int(end_mask.sum()) != 248:
        raise RuntimeError(
            "Expected one endpoint window for each of the 248 test engines."
        )

    for backbone in backbones:
        for seed in seeds:
            for representation in representation_names:
                print(
                    "\n[FD004]",
                    f"backbone={backbone}",
                    f"seed={seed}",
                    f"representation={representation}",
                )

                set_seed(
                    seed
                )

                (
                    X_train,
                    X_valid,
                    X_test,
                ) = reps[
                    representation
                ]

                if backbone == "xgboost":
                    model = train_xgboost(
                        X_train,
                        data.train.q,
                        X_valid,
                        data.valid.q,
                        config=(
                            configs[
                                "xgboost"
                            ]
                        ),
                        seed=seed,
                        selection_metric=(
                            "rmse"
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

                else:
                    model = (
                        build_torch_model(
                            backbone,
                            configs[
                                backbone
                            ],
                        )
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
                        epochs=(
                            args.epochs
                        ),
                        patience=(
                            args.patience
                        ),
                        device=DEVICE,
                        selection_metric=(
                            "rmse"
                        ),
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

                y_true = np.asarray(
                    test_metrics[
                        "y_true"
                    ]
                )

                y_pred = np.asarray(
                    test_metrics[
                        "y_pred"
                    ]
                )

                endpoint_mse = float(
                    np.mean(
                        (
                            y_pred[
                                end_mask
                            ]
                            - y_true[
                                end_mask
                            ]
                        ) ** 2
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

                    "paper_counts_match":
                        counts_match,

                    "centroid_shrinkage_kappa":
                        args.centroid_shrinkage_kappa,

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
                            test_metrics[
                                "mse"
                            ]
                        ),

                    "test_all_rmse":
                        float(
                            test_metrics[
                                "rmse"
                            ]
                        ),

                    "test_endpoint_mse":
                        endpoint_mse,

                    "test_endpoint_rmse":
                        endpoint_rmse,
                }

                print(
                    row
                )

                save_row(
                    args.output,
                    row,
                )

    print(
        "\nSaved results to",
        args.output,
    )


if __name__ == "__main__":
    main()
