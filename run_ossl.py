from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
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
    OSSL_TABM_LR_GRID,
    OSSL_TABM_SEARCH_EPOCHS,
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


def make_loader(
    X,
    y,
    batch_size,
    shuffle,
    sample_weight=None,
):
    tensors = [
        torch.as_tensor(
            X,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            y,
            dtype=torch.float32,
        ),
    ]

    if sample_weight is not None:
        tensors.append(
            torch.as_tensor(
                sample_weight,
                dtype=torch.float32,
            )
        )

    dataset = TensorDataset(
        *tensors
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


def train_torch_once(
    backbone,
    config,
    seed,
    X_train,
    y_train,
    train_weight,
    X_valid,
    y_valid,
    valid_source,
    batch_size,
    epochs,
    patience,
):
    # Reset the same seed for every candidate so that LR comparisons are
    # not confounded by different random initializations.
    set_seed(
        seed
    )

    model = build_torch_model(
        backbone,
        config,
    )

    train_loader = make_loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True,
        sample_weight=(
            train_weight
        ),
    )

    valid_loader = make_loader(
        X_valid,
        y_valid,
        batch_size,
        shuffle=False,
    )

    fit_torch_regressor(
        backbone,
        model,
        train_loader,
        valid_loader,
        epochs=epochs,
        patience=patience,
        device=DEVICE,
        selection_metric=(
            "macro_mse"
        ),
        valid_groups=(
            valid_source
        ),
    )

    valid_metrics = evaluate_torch(
        backbone,
        model,
        valid_loader,
        device=DEVICE,
        groups=(
            valid_source
        ),
        return_predictions=False,
    )

    return (
        model,
        valid_metrics,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OSSL MIR organic-carbon raw-versus-quotient regression."
        )
    )

    source_group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    source_group.add_argument(
        "--all-l1",
    )

    source_group.add_argument(
        "--mir-l0",
    )

    parser.add_argument(
        "--soillab-l1",
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
        default=(
            train_config.epochs
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=(
            train_config.patience
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=(
            train_config.batch_size
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/ossl_results.csv"
        ),
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
            "--soillab-l1 is required when --mir-l0 is used."
        )

    data = load_ossl(
        all_l1_path=(
            args.all_l1
        ),
        mir_l0_path=(
            args.mir_l0
        ),
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

    (
        _,
        X_forward_train,
        _,
        _,
    ) = build_ossl_forward_design(
        data,
        n_knots=7,
        degree=3,
    )

    source_weights = (
        make_equal_source_weights(
            data.train.source
        )
    )

    rbpca = ResidualBatchPCA(
        rank=None,
        variance_threshold=0.90,
        ridge_alpha=1.0,
        crossfit="none",
        centroid_shrinkage_kappa=0.0,
    )

    rbpca.fit(
        Y=data.train.Y,
        X_forward=(
            X_forward_train
        ),
        batch=(
            data.train.source
        ),
        sample_weight=(
            source_weights
        ),
    )

    print(
        "RB-PCA diagnostics:"
    )

    print(
        rbpca.diagnostics()
    )

    reps_426 = {
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

    # Projection first, deterministic pooling second.
    reps = {
        name: tuple(
            mean_pool_mir_426_to_71(
                X
            )
            for X in values
        )
        for name, values
        in reps_426.items()
    }

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
        "ossl"
    )

    for backbone in backbones:
        for seed in seeds:
            for representation in representation_names:
                print(
                    "\n[OSSL]",
                    f"backbone={backbone}",
                    f"seed={seed}",
                    f"representation={representation}",
                )

                (
                    X_train,
                    X_valid,
                    X_test,
                ) = reps[
                    representation
                ]

                selected_lr = None
                valid_macro_mse = None

                # --------------------------------------------------
                # XGBoost:
                # custom equal-source validation macro-MSE controls
                # early stopping.
                # --------------------------------------------------
                if backbone == "xgboost":
                    set_seed(
                        seed
                    )

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
                        sample_weight=(
                            source_weights
                        ),
                        valid_groups=(
                            data.valid.source
                        ),
                        selection_metric=(
                            "macro_mse"
                        ),
                    )

                    valid_metrics = (
                        evaluate_xgboost(
                            model,
                            X_valid,
                            data.valid.q,
                            groups=(
                                data.valid.source
                            ),
                        )
                    )

                    valid_macro_mse = float(
                        valid_metrics[
                            "macro_mse"
                        ]
                    )

                    test_metrics = (
                        evaluate_xgboost(
                            model,
                            X_test,
                            data.test.q,
                            return_predictions=True,
                            groups=(
                                data.test.source
                            ),
                        )
                    )

                # --------------------------------------------------
                # TabM:
                # identical LR grid for raw and quotient.
                # Every candidate gets 60 epochs in the formal run.
                # Checkpoint selection uses validation macro-MSE.
                # --------------------------------------------------
                elif backbone == "tabm":
                    lr_grid = list(
                        OSSL_TABM_LR_GRID
                    )

                    search_epochs = (
                        OSSL_TABM_SEARCH_EPOCHS
                    )

                    search_patience = None

                    if args.smoke:
                        lr_grid = (
                            lr_grid[:1]
                        )
                        search_epochs = min(
                            3,
                            search_epochs,
                        )

                    best_value = float(
                        "inf"
                    )

                    best_model = None
                    best_lr = None

                    for lr in lr_grid:
                        print(
                            "  TabM candidate lr=",
                            lr,
                        )

                        candidate_config = (
                            replace(
                                configs[
                                    "tabm"
                                ],
                                lr=float(lr),
                            )
                        )

                        (
                            candidate_model,
                            candidate_valid,
                        ) = train_torch_once(
                            backbone="tabm",
                            config=(
                                candidate_config
                            ),
                            seed=seed,
                            X_train=X_train,
                            y_train=(
                                data.train.q
                            ),
                            train_weight=(
                                source_weights
                            ),
                            X_valid=X_valid,
                            y_valid=(
                                data.valid.q
                            ),
                            valid_source=(
                                data.valid.source
                            ),
                            batch_size=(
                                args.batch_size
                            ),
                            epochs=(
                                search_epochs
                            ),
                            patience=(
                                search_patience
                            ),
                        )

                        value = float(
                            candidate_valid[
                                "macro_mse"
                            ]
                        )

                        print(
                            "    valid macro-MSE =",
                            value,
                        )

                        if value < best_value:
                            best_value = value
                            best_model = (
                                candidate_model
                            )
                            best_lr = float(
                                lr
                            )

                    model = best_model
                    selected_lr = (
                        best_lr
                    )
                    valid_macro_mse = (
                        best_value
                    )

                    test_loader = (
                        make_loader(
                            X_test,
                            data.test.q,
                            args.batch_size,
                            shuffle=False,
                        )
                    )

                    test_metrics = (
                        evaluate_torch(
                            "tabm",
                            model,
                            test_loader,
                            device=DEVICE,
                            return_predictions=True,
                            groups=(
                                data.test.source
                            ),
                        )
                    )

                # --------------------------------------------------
                # ResNet / FT-Transformer:
                # validation checkpoint selected by equal-source
                # macro-MSE.
                # --------------------------------------------------
                else:
                    (
                        model,
                        valid_metrics,
                    ) = train_torch_once(
                        backbone=backbone,
                        config=(
                            configs[
                                backbone
                            ]
                        ),
                        seed=seed,
                        X_train=X_train,
                        y_train=(
                            data.train.q
                        ),
                        train_weight=(
                            source_weights
                        ),
                        X_valid=X_valid,
                        y_valid=(
                            data.valid.q
                        ),
                        valid_source=(
                            data.valid.source
                        ),
                        batch_size=(
                            args.batch_size
                        ),
                        epochs=(
                            args.epochs
                        ),
                        patience=(
                            args.patience
                        ),
                    )

                    selected_lr = float(
                        configs[
                            backbone
                        ].lr
                    )

                    valid_macro_mse = float(
                        valid_metrics[
                            "macro_mse"
                        ]
                    )

                    test_loader = (
                        make_loader(
                            X_test,
                            data.test.q,
                            args.batch_size,
                            shuffle=False,
                        )
                    )

                    test_metrics = (
                        evaluate_torch(
                            backbone,
                            model,
                            test_loader,
                            device=DEVICE,
                            return_predictions=True,
                            groups=(
                                data.test.source
                            ),
                        )
                    )

                row = {
                    "dataset":
                        "OSSL_OC",

                    "backbone":
                        backbone,

                    "seed":
                        seed,

                    "representation":
                        representation,

                    "selected_lr":
                        selected_lr,

                    "valid_macro_mse":
                        valid_macro_mse,

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
                            test_metrics[
                                "mse"
                            ]
                        ),

                    "test_rmse":
                        float(
                            test_metrics[
                                "rmse"
                            ]
                        ),

                    "test_mae":
                        float(
                            test_metrics[
                                "mae"
                            ]
                        ),
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
