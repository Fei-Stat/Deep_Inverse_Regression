from __future__ import annotations

import torch
import torch.nn as nn

from scripts.config import (
    resnet_config,
    ft_config,
    tabm_config,
    xgb_config,
)


try:
    from rtdl_revisiting_models import (
        ResNet as RTDLResNet,
        FTTransformer as RTDLFTTransformer,
    )
except ImportError as e:
    RTDLResNet = None
    RTDLFTTransformer = None
    _RTDL_IMPORT_ERROR = e
else:
    _RTDL_IMPORT_ERROR = None


try:
    from tabm import TabM as YandexTabM
except ImportError as e:
    YandexTabM = None
    _TABM_IMPORT_ERROR = e
else:
    _TABM_IMPORT_ERROR = None


try:
    import xgboost as xgb
except ImportError as e:
    xgb = None
    _XGB_IMPORT_ERROR = e
else:
    _XGB_IMPORT_ERROR = None


def _require_rtdl():
    if RTDLResNet is None or RTDLFTTransformer is None:
        raise ImportError(
            "rtdl_revisiting_models is required. Install it with:\n"
            "  pip install rtdl_revisiting_models"
        ) from _RTDL_IMPORT_ERROR


def _require_tabm():
    if YandexTabM is None:
        raise ImportError(
            "The official Yandex TabM package is required. Install it with:\n"
            "  pip install tabm"
        ) from _TABM_IMPORT_ERROR


def _require_xgboost():
    if xgb is None:
        raise ImportError(
            "xgboost is required. Install it with:\n"
            "  pip install xgboost"
        ) from _XGB_IMPORT_ERROR


class ResNetMLP(nn.Module):
    """Yandex/RTDL tabular ResNet for scalar regression."""

    def __init__(self, config=None):
        super().__init__()
        _require_rtdl()

        if config is None:
            config = resnet_config

        self.config = config

        if config.output_dim != 1:
            raise ValueError("Current experiments require output_dim=1.")

        self.model = RTDLResNet(
            d_in=config.input_dim,
            d_out=config.output_dim,
            n_blocks=config.n_blocks,
            d_block=config.d_block,
            d_hidden=None,
            d_hidden_multiplier=config.d_hidden_multiplier,
            dropout1=config.dropout1,
            dropout2=config.dropout2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"ResNetMLP expects x with shape (B,D), got {tuple(x.shape)}"
            )

        out = self.model(x)

        if out.ndim != 2 or out.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected ResNet output shape: {tuple(out.shape)}"
            )

        return out.squeeze(-1)


class FTTransformer(nn.Module):
    """Yandex/RTDL FT-Transformer for numerical-only scalar regression."""

    def __init__(self, config=None):
        super().__init__()
        _require_rtdl()

        if config is None:
            config = ft_config

        self.config = config

        if config.output_dim != 1:
            raise ValueError("Current experiments require output_dim=1.")

        self.model = RTDLFTTransformer(
            n_cont_features=config.input_dim,
            cat_cardinalities=[],
            d_out=config.output_dim,
            n_blocks=config.n_blocks,
            d_block=config.d_block,
            attention_n_heads=config.n_heads,
            attention_dropout=config.attention_dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=config.ffn_d_hidden_multiplier,
            ffn_dropout=config.ffn_dropout,
            residual_dropout=config.residual_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"FTTransformer expects x with shape (B,D), got {tuple(x.shape)}"
            )

        out = self.model(x, None)

        if out.ndim != 2 or out.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected FT-Transformer output shape: {tuple(out.shape)}"
            )

        return out.squeeze(-1)

    def make_parameter_groups(self):
        return self.model.make_parameter_groups()


class TabM(nn.Module):
    """
    Official Yandex TabM wrapper for scalar regression.

    Forward output has shape (B, K). Training loss is averaged over members;
    inference averages member predictions.
    """

    def __init__(self, config=None):
        super().__init__()
        _require_tabm()

        if config is None:
            config = tabm_config

        self.config = config

        if config.output_dim != 1:
            raise ValueError("Current experiments require output_dim=1.")

        self.model = YandexTabM.make(
            n_num_features=config.input_dim,
            d_out=config.output_dim,
            arch_type=config.arch_type,
            k=config.k,
            n_blocks=config.n_blocks,
            d_block=config.d_block,
            dropout=config.dropout,
        )

    @property
    def k(self) -> int:
        return self.model.k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"TabM expects x with shape (B,D), got {tuple(x.shape)}"
            )

        out = self.model(x)

        if out.ndim != 3 or out.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected TabM output shape: {tuple(out.shape)}"
            )

        return out.squeeze(-1)


def make_xgboost(config=None, seed: int = 42):
    """Construct XGBRegressor for FD004 or OSSL organic-carbon prediction."""
    _require_xgboost()

    if config is None:
        config = xgb_config

    return xgb.XGBRegressor(
        objective=config.objective,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        tree_method=config.tree_method,
        device=config.device,
        eval_metric=config.eval_metric,
        early_stopping_rounds=config.early_stopping_rounds,
        n_jobs=config.n_jobs,
        random_state=seed,
    )
