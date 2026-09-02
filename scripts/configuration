from __future__ import annotations

from dataclasses import dataclass, replace

import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_device()

# Fixed seeds used by both FD004 and OSSL paired experiments.
SEEDS = [7523, 4996, 4292, 4494, 365]

# Dataset-specific downstream dimensions.
# FD004: 63 sensor-window features + 3 operating settings = 66.
# OSSL: 426 MIR ordinates -> quotient first -> mean pooling -> 71.
FD004_INPUT_DIM = 66
OSSL_INPUT_DIM = 71


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 200
    patience: int = 20
    min_delta: float = 0.0
    criterion: str = "mse"
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class ResNetMLPConfig:
    input_dim: int = FD004_INPUT_DIM
    output_dim: int = 1
    n_blocks: int = 3
    d_block: int = 192
    d_hidden_multiplier: float = 2.0
    dropout1: float = 0.15
    dropout2: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class FTTransformerConfig:
    input_dim: int = FD004_INPUT_DIM
    output_dim: int = 1
    n_blocks: int = 3
    d_block: int = 192
    n_heads: int = 8
    attention_dropout: float = 0.2
    ffn_d_hidden_multiplier: float = 4.0 / 3.0
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    lr: float = 1e-4
    weight_decay: float = 1e-5


@dataclass
class TabMConfig:
    input_dim: int = FD004_INPUT_DIM
    output_dim: int = 1
    k: int = 32
    arch_type: str = "tabm"
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 3e-4

    @property
    def n_views(self) -> int:
        return self.k

    @property
    def hidden_dims(self):
        return [self.d_block] * self.n_blocks


@dataclass
class XGBoostConfig:
    n_estimators: int = 2000
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    tree_method: str = "hist"
    device: str = "cpu"
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    early_stopping_rounds: int = 50
    n_jobs: int = -1


train_config = TrainingConfig()
resnet_config = ResNetMLPConfig()
ft_config = FTTransformerConfig()
tabm_config = TabMConfig()
xgb_config = XGBoostConfig()


def configs_for_input_dim(input_dim: int):
    input_dim = int(input_dim)

    if input_dim < 1:
        raise ValueError("input_dim must be positive.")

    return {
        "resnet": replace(resnet_config, input_dim=input_dim),
        "ft_transformer": replace(ft_config, input_dim=input_dim),
        "tabm": replace(tabm_config, input_dim=input_dim),
        "xgboost": xgb_config,
    }


def configs_for_dataset(dataset: str):
    dataset = dataset.lower()

    if dataset in {"fd004", "cmapss_fd004", "c-mapss_fd004"}:
        return configs_for_input_dim(FD004_INPUT_DIM)

    if dataset in {"ossl", "ossl_oc", "ossl-organic-carbon"}:
        return configs_for_input_dim(OSSL_INPUT_DIM)

    raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    print("SEEDS:", SEEDS)
    print("FD004 configs:", configs_for_dataset("fd004"))
    print("OSSL configs:", configs_for_dataset("ossl"))
