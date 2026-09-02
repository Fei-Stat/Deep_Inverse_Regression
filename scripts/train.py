from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb

from scripts.config import (
    DEVICE,
    resnet_config,
    ft_config,
    tabm_config,
)
from scripts.models import (
    ResNetMLP,
    FTTransformer,
    TabM,
)


def set_seed(seed: int):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def build_torch_model(
    name: str,
    config=None,
):
    name = name.lower()

    if name in {
        "resnet",
        "resnet_mlp",
        "resnet-mlp",
    }:
        return ResNetMLP(
            config
        ).to(DEVICE)

    if name in {
        "ft",
        "ft_transformer",
        "ft-transformer",
    }:
        return FTTransformer(
            config
        ).to(DEVICE)

    if name == "tabm":
        return TabM(
            config
        ).to(DEVICE)

    raise ValueError(
        f"Unknown torch backbone: {name}"
    )


def build_optimizer(
    name: str,
    model,
):
    name = name.lower()

    config = getattr(
        model,
        "config",
        None,
    )

    if config is None:
        if name in {
            "resnet",
            "resnet_mlp",
            "resnet-mlp",
        }:
            config = resnet_config

        elif name in {
            "ft",
            "ft_transformer",
            "ft-transformer",
        }:
            config = ft_config

        elif name == "tabm":
            config = tabm_config

        else:
            raise ValueError(
                f"Unknown backbone: {name}"
            )

    if name in {
        "resnet",
        "resnet_mlp",
        "resnet-mlp",
    }:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=(
                config.weight_decay
            ),
        )

    if name in {
        "ft",
        "ft_transformer",
        "ft-transformer",
    }:
        return torch.optim.AdamW(
            model.make_parameter_groups(),
            lr=config.lr,
            weight_decay=(
                config.weight_decay
            ),
        )

    if name == "tabm":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=(
                config.weight_decay
            ),
        )

    raise ValueError(
        f"Unknown backbone: {name}"
    )


def regression_prediction(
    name,
    output,
):
    name = name.lower()

    if name == "tabm":
        if output.ndim != 2:
            raise ValueError(
                "TabM output must have shape (B,K)."
            )

        return output.mean(
            dim=1
        )

    return output.reshape(-1)


def regression_loss(
    name,
    output,
    target,
    sample_weight=None,
):
    """
    Weighted MSE.

    For TabM the member-wise squared errors are first averaged over
    ensemble members for each observation, then observation weights are
    applied. This preserves the TabM ensemble training objective while
    allowing exact source weighting.
    """
    name = name.lower()

    target = (
        target.float()
        .reshape(-1)
    )

    if sample_weight is None:
        sample_weight = torch.ones_like(
            target
        )
    else:
        sample_weight = (
            sample_weight.float()
            .reshape(-1)
        )

    if name == "tabm":
        if output.ndim != 2:
            raise ValueError(
                "TabM output must have shape (B,K)."
            )

        target_k = (
            target[:, None]
            .expand_as(output)
        )

        per_member_sq = (
            output - target_k
        ) ** 2

        per_observation = (
            per_member_sq.mean(
                dim=1
            )
        )

    else:
        pred = output.reshape(-1)

        per_observation = (
            pred - target
        ) ** 2

    denom = sample_weight.sum()

    if denom <= 0:
        raise ValueError(
            "Nonpositive total sample weight."
        )

    return (
        sample_weight
        * per_observation
    ).sum() / denom


def macro_group_mse(
    y_true,
    y_pred,
    groups,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    groups = np.asarray(
        groups
    )

    if not (
        len(y_true)
        == len(y_pred)
        == len(groups)
    ):
        raise ValueError(
            "y_true, y_pred and groups must align."
        )

    values = []

    for group in np.unique(
        groups
    ):
        idx = groups == group

        values.append(
            np.mean(
                (
                    y_pred[idx]
                    - y_true[idx]
                ) ** 2
            )
        )

    return float(
        np.mean(values)
    )


def regression_metrics(
    y_true,
    y_pred,
    groups=None,
):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "Shapes do not match."
        )

    error = (
        y_pred - y_true
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    result = {
        "mse": mse,
        "rmse": float(
            np.sqrt(mse)
        ),
        "mae": float(
            np.mean(
                np.abs(error)
            )
        ),
    }

    if groups is not None:
        macro_mse = (
            macro_group_mse(
                y_true,
                y_pred,
                groups,
            )
        )

        result[
            "macro_mse"
        ] = macro_mse

        result[
            "macro_rmse"
        ] = float(
            np.sqrt(
                macro_mse
            )
        )

    return result


def _unpack_batch(
    batch,
    device,
):
    if len(batch) == 2:
        x, y = batch
        w = None

    elif len(batch) == 3:
        x, y, w = batch

    else:
        raise ValueError(
            "DataLoader batch must contain (X,y) or (X,y,w)."
        )

    x = (
        x.to(
            device,
            non_blocking=True,
        )
        .float()
    )

    y = (
        y.to(
            device,
            non_blocking=True,
        )
        .float()
        .reshape(-1)
    )

    if w is not None:
        w = (
            w.to(
                device,
                non_blocking=True,
            )
            .float()
            .reshape(-1)
        )

    return x, y, w


def train_one_epoch(
    name,
    model,
    loader,
    optimizer,
    device=DEVICE,
):
    model.train()

    weighted_loss_sum = 0.0
    weight_sum = 0.0

    y_true_all = []
    y_pred_all = []

    for batch in loader:
        x, y, w = _unpack_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(x)

        loss = regression_loss(
            name,
            output,
            y,
            sample_weight=w,
        )

        loss.backward()
        optimizer.step()

        pred = regression_prediction(
            name,
            output.detach(),
        )

        if w is None:
            local_weight = float(
                len(y)
            )
        else:
            local_weight = float(
                w.sum().item()
            )

        weighted_loss_sum += (
            float(loss.item())
            * local_weight
        )

        weight_sum += (
            local_weight
        )

        y_true_all.append(
            y.detach()
            .cpu()
            .numpy()
        )

        y_pred_all.append(
            pred.detach()
            .cpu()
            .numpy()
        )

    y_true = np.concatenate(
        y_true_all
    )

    y_pred = np.concatenate(
        y_pred_all
    )

    metrics = regression_metrics(
        y_true,
        y_pred,
    )

    metrics["loss"] = (
        weighted_loss_sum
        / max(
            weight_sum,
            1e-12,
        )
    )

    return metrics


@torch.no_grad()
def evaluate_torch(
    name,
    model,
    loader,
    device=DEVICE,
    return_predictions=False,
    groups=None,
):
    model.eval()

    weighted_loss_sum = 0.0
    weight_sum = 0.0

    y_true_all = []
    y_pred_all = []

    for batch in loader:
        x, y, w = _unpack_batch(
            batch,
            device,
        )

        output = model(x)

        loss = regression_loss(
            name,
            output,
            y,
            sample_weight=w,
        )

        pred = regression_prediction(
            name,
            output,
        )

        if w is None:
            local_weight = float(
                len(y)
            )
        else:
            local_weight = float(
                w.sum().item()
            )

        weighted_loss_sum += (
            float(loss.item())
            * local_weight
        )

        weight_sum += (
            local_weight
        )

        y_true_all.append(
            y.detach()
            .cpu()
            .numpy()
        )

        y_pred_all.append(
            pred.detach()
            .cpu()
            .numpy()
        )

    if not y_true_all:
        raise ValueError(
            "Evaluation loader is empty."
        )

    y_true = np.concatenate(
        y_true_all
    )

    y_pred = np.concatenate(
        y_pred_all
    )

    metrics = regression_metrics(
        y_true,
        y_pred,
        groups=groups,
    )

    metrics["loss"] = (
        weighted_loss_sum
        / max(
            weight_sum,
            1e-12,
        )
    )

    if return_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred

    return metrics


def fit_torch_regressor(
    name,
    model,
    train_loader,
    valid_loader,
    epochs=200,
    patience=20,
    min_delta=0.0,
    device=DEVICE,
    selection_metric="rmse",
    valid_groups=None,
):
    """
    Train a PyTorch regressor and restore the best checkpoint.

    selection_metric:
        "rmse", "mse", "macro_mse", or "macro_rmse".

    For OSSL use:
        selection_metric="macro_mse"
        valid_groups=validation source labels
    """
    allowed_metrics = {
        "rmse",
        "mse",
        "macro_mse",
        "macro_rmse",
    }

    if selection_metric not in allowed_metrics:
        raise ValueError(
            f"Unknown selection metric: {selection_metric}"
        )

    if (
        selection_metric.startswith(
            "macro_"
        )
        and valid_groups is None
    ):
        raise ValueError(
            "valid_groups are required for macro validation."
        )

    optimizer = build_optimizer(
        name,
        model,
    )

    best_value = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(
        1,
        int(epochs) + 1,
    ):
        train_metrics = train_one_epoch(
            name,
            model,
            train_loader,
            optimizer,
            device=device,
        )

        valid_metrics = evaluate_torch(
            name,
            model,
            valid_loader,
            device=device,
            return_predictions=False,
            groups=valid_groups,
        )

        current = float(
            valid_metrics[
                selection_metric
            ]
        )

        row = {
            "epoch": epoch,
            "train_loss":
                train_metrics["loss"],
            "train_mse":
                train_metrics["mse"],
            "train_rmse":
                train_metrics["rmse"],
            "valid_loss":
                valid_metrics["loss"],
            "valid_mse":
                valid_metrics["mse"],
            "valid_rmse":
                valid_metrics["rmse"],
            "selection_metric":
                selection_metric,
            "selection_value":
                current,
        }

        if "macro_mse" in valid_metrics:
            row[
                "valid_macro_mse"
            ] = valid_metrics[
                "macro_mse"
            ]

            row[
                "valid_macro_rmse"
            ] = valid_metrics[
                "macro_rmse"
            ]

        history.append(row)

        if (
            current
            < best_value
            - float(min_delta)
        ):
            best_value = current

            best_state_dict = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict()
                .items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            patience is not None
            and int(patience) > 0
            and epochs_without_improvement
            >= int(patience)
        ):
            break

    if best_state_dict is None:
        raise RuntimeError(
            "Training did not produce a checkpoint."
        )

    model.load_state_dict(
        best_state_dict
    )

    return (
        history,
        best_state_dict,
    )


def _xgb_params(
    config,
    seed,
):
    return {
        "objective":
            config.objective,

        "max_depth":
            config.max_depth,

        "eta":
            config.learning_rate,

        "subsample":
            config.subsample,

        "colsample_bytree":
            config.colsample_bytree,

        "min_child_weight":
            config.min_child_weight,

        "lambda":
            config.reg_lambda,

        "alpha":
            config.reg_alpha,

        "tree_method":
            config.tree_method,

        "device":
            config.device,

        "seed":
            int(seed),

        "nthread":
            config.n_jobs,
    }


def train_xgboost(
    X_train,
    y_train,
    X_valid,
    y_valid,
    config,
    seed=42,
    sample_weight=None,
    valid_groups=None,
    selection_metric="rmse",
):
    """
    Train XGBoost with validation-based early stopping.

    For OSSL, selection_metric="macro_mse" makes early stopping use
    equal-source validation macro-MSE rather than ordinary micro-RMSE.
    """
    X_train = np.asarray(
        X_train,
        dtype=float,
    )

    X_valid = np.asarray(
        X_valid,
        dtype=float,
    )

    y_train = np.asarray(
        y_train,
        dtype=float,
    ).reshape(-1)

    y_valid = np.asarray(
        y_valid,
        dtype=float,
    ).reshape(-1)

    if sample_weight is None:
        sample_weight = np.ones(
            len(y_train),
            dtype=float,
        )
    else:
        sample_weight = np.asarray(
            sample_weight,
            dtype=float,
        ).reshape(-1)

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        weight=sample_weight,
    )

    dvalid = xgb.DMatrix(
        X_valid,
        label=y_valid,
    )

    params = _xgb_params(
        config,
        seed,
    )

    if selection_metric == "rmse":
        params[
            "eval_metric"
        ] = "rmse"

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=(
                config.n_estimators
            ),
            evals=[
                (
                    dvalid,
                    "valid",
                )
            ],
            maximize=False,
            early_stopping_rounds=(
                config.early_stopping_rounds
            ),
            verbose_eval=False,
        )

    elif selection_metric == "macro_mse":
        if valid_groups is None:
            raise ValueError(
                "valid_groups are required for OSSL macro-MSE."
            )

        valid_groups = np.asarray(
            valid_groups
        )

        if len(valid_groups) != len(
            y_valid
        ):
            raise ValueError(
                "valid_groups length does not match y_valid."
            )

        # Disable default metric so early stopping monitors the custom
        # equal-source macro-MSE.
        params[
            "disable_default_eval_metric"
        ] = 1

        def macro_metric(
            predt,
            dmatrix,
        ):
            y = (
                dmatrix.get_label()
            )

            value = macro_group_mse(
                y,
                predt,
                valid_groups,
            )

            return (
                "macro_mse",
                value,
            )

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=(
                config.n_estimators
            ),
            evals=[
                (
                    dvalid,
                    "valid",
                )
            ],
            custom_metric=(
                macro_metric
            ),
            maximize=False,
            early_stopping_rounds=(
                config.early_stopping_rounds
            ),
            verbose_eval=False,
        )

    else:
        raise ValueError(
            "XGBoost selection_metric must be "
            "'rmse' or 'macro_mse'."
        )

    booster.selection_metric_ = (
        selection_metric
    )

    return booster


def _predict_booster(
    model,
    X,
):
    dmatrix = xgb.DMatrix(
        np.asarray(
            X,
            dtype=float,
        )
    )

    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )

    if best_iteration is None:
        return model.predict(
            dmatrix
        )

    return model.predict(
        dmatrix,
        iteration_range=(
            0,
            int(best_iteration) + 1,
        ),
    )


def evaluate_xgboost(
    model,
    X,
    y,
    return_predictions=False,
    groups=None,
):
    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    pred = np.asarray(
        _predict_booster(
            model,
            X,
        ),
        dtype=float,
    ).reshape(-1)

    metrics = regression_metrics(
        y,
        pred,
        groups=groups,
    )

    if return_predictions:
        metrics[
            "y_true"
        ] = y

        metrics[
            "y_pred"
        ] = pred

    return metrics
