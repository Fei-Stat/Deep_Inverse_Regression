from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from scripts.config import (
    DEVICE,
    resnet_config,
    ft_config,
    tabm_config,
    xgb_config,
)
from scripts.models import (
    ResNetMLP,
    FTTransformer,
    TabM,
    make_xgboost,
)


def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_torch_model(name: str, config=None):
    name = name.lower()

    if name in {"resnet", "resnet_mlp", "resnet-mlp"}:
        return ResNetMLP(config).to(DEVICE)

    if name in {"ft", "ft_transformer", "ft-transformer"}:
        return FTTransformer(config).to(DEVICE)

    if name == "tabm":
        return TabM(config).to(DEVICE)

    raise ValueError(f"Unknown torch backbone: {name}")


def build_optimizer(name: str, model):
    name = name.lower()
    config = getattr(model, "config", None)

    if config is None:
        if name in {"resnet", "resnet_mlp", "resnet-mlp"}:
            config = resnet_config
        elif name in {"ft", "ft_transformer", "ft-transformer"}:
            config = ft_config
        elif name == "tabm":
            config = tabm_config
        else:
            raise ValueError(f"Unknown torch backbone: {name}")

    if name in {"resnet", "resnet_mlp", "resnet-mlp"}:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    if name in {"ft", "ft_transformer", "ft-transformer"}:
        return torch.optim.AdamW(
            model.make_parameter_groups(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    if name == "tabm":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    raise ValueError(f"Unknown torch backbone: {name}")


def regression_prediction(name: str, output: torch.Tensor) -> torch.Tensor:
    """
    Return one scalar prediction per observation.

    ResNet / FT-Transformer:
        output shape (B,)

    TabM:
        output shape (B, K), averaged over K members.
    """
    name = name.lower()

    if name == "tabm":
        if output.ndim != 2:
            raise ValueError(
                f"TabM regression output must have shape (B,K), "
                f"got {tuple(output.shape)}"
            )
        return output.mean(dim=1)

    if output.ndim != 1:
        raise ValueError(
            f"Regression output must have shape (B,), got {tuple(output.shape)}"
        )

    return output


def regression_loss(
    name: str,
    output: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-squared-error objective.

    For TabM, MSE is averaged over all ensemble members rather than
    computing MSE after first averaging member predictions.
    """
    name = name.lower()
    target = target.float().reshape(-1)

    if name == "tabm":
        if output.ndim != 2:
            raise ValueError(
                f"TabM regression output must have shape (B,K), "
                f"got {tuple(output.shape)}"
            )

        if output.shape[0] != target.shape[0]:
            raise ValueError("Target batch size does not match TabM output.")

        target_k = target[:, None].expand_as(output)
        return F.mse_loss(output, target_k, reduction="mean")

    pred = output.reshape(-1)

    if pred.shape[0] != target.shape[0]:
        raise ValueError("Target batch size does not match model output.")

    return F.mse_loss(pred, target, reduction="mean")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def train_one_epoch(
    name,
    model,
    loader,
    optimizer,
    device=DEVICE,
):
    model.train()

    total_member_loss = 0.0
    total_squared_error = 0.0
    n_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float().reshape(-1)

        optimizer.zero_grad(set_to_none=True)

        output = model(x)
        loss = regression_loss(name, output, y)

        loss.backward()
        optimizer.step()

        pred = regression_prediction(name, output.detach())

        batch_size = x.shape[0]
        total_member_loss += loss.item() * batch_size
        total_squared_error += torch.sum((pred - y) ** 2).item()
        n_samples += batch_size

    if n_samples == 0:
        raise ValueError("Training loader is empty.")

    mse = total_squared_error / n_samples

    return {
        "loss": total_member_loss / n_samples,
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


@torch.no_grad()
def evaluate_torch(
    name,
    model,
    loader,
    device=DEVICE,
    return_predictions: bool = False,
):
    model.eval()

    total_member_loss = 0.0
    n_samples = 0

    y_true_all = []
    y_pred_all = []

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float().reshape(-1)

        output = model(x)
        loss = regression_loss(name, output, y)
        pred = regression_prediction(name, output)

        batch_size = x.shape[0]
        total_member_loss += loss.item() * batch_size
        n_samples += batch_size

        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    if n_samples == 0:
        raise ValueError("Evaluation loader is empty.")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = total_member_loss / n_samples

    if return_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred

    return metrics


def fit_torch_regressor(
    name,
    model,
    train_loader,
    valid_loader,
    epochs: int = 200,
    patience: int = 20,
    min_delta: float = 0.0,
    device=DEVICE,
):
    """
    Train a PyTorch regression backbone with validation-RMSE early stopping.

    The model is restored to the best validation-RMSE checkpoint.
    """
    optimizer = build_optimizer(name, model)

    best_rmse = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(epochs) + 1):
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
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_rmse": train_metrics["rmse"],
                "valid_loss": valid_metrics["loss"],
                "valid_mse": valid_metrics["mse"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
            }
        )

        current = valid_metrics["rmse"]

        if current < best_rmse - float(min_delta):
            best_rmse = current
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(patience):
            break

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state_dict)

    return history, best_state_dict


def train_xgboost(
    X_train,
    y_train,
    X_valid,
    y_valid,
    config=xgb_config,
    seed: int = 42,
    sample_weight=None,
):
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_valid = np.asarray(y_valid, dtype=float).reshape(-1)

    model = make_xgboost(config=config, seed=seed)

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_valid, y_valid)],
        "verbose": False,
    }

    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(
            sample_weight,
            dtype=float,
        ).reshape(-1)

    model.fit(**fit_kwargs)
    return model


def evaluate_xgboost(
    model,
    X,
    y,
    return_predictions: bool = False,
):
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(model.predict(X), dtype=float).reshape(-1)

    metrics = regression_metrics(y, pred)

    if return_predictions:
        metrics["y_true"] = y
        metrics["y_pred"] = pred

    return metrics
