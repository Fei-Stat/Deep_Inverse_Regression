from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import SplineTransformer, StandardScaler


FD004_COLUMNS = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{j}" for j in range(1, 22)]
)

SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{j}" for j in range(1, 22)]


@dataclass
class FD004Partition:
    """
    Window-level FD004 data.

    Y:
        63-dimensional sensor representation:
        21 sensors x {mean, std, slope}.

    q:
        Capped RUL target at the end of each window.

    settings:
        Three operating settings at the end of each window.
        These are retained as known covariates and are NOT quotient-projected.

    engine_id:
        Nuisance-group label used by Residual Batch PCA.

    cycle:
        End cycle of the window.
    """
    Y: np.ndarray
    q: np.ndarray
    settings: np.ndarray
    engine_id: np.ndarray
    cycle: np.ndarray


@dataclass
class FD004Data:
    train: FD004Partition
    valid: FD004Partition
    test: FD004Partition
    train_engine_ids: np.ndarray
    valid_engine_ids: np.ndarray


class FD004ForwardDesign:
    """
    Training-only forward-design builder for

        E[Y | RUL, operating settings].

    The design contains:
        1. intercept
        2. spline basis of capped RUL
        3. standardized operating settings
        4. six operating-regime indicators
        5. RUL-spline x operating-regime interactions

    Operating regimes are learned from training-window settings using KMeans.
    """

    def __init__(
        self,
        n_regimes: int = 6,
        n_knots: int = 5,
        spline_degree: int = 3,
        random_state: int = 42,
    ):
        self.n_regimes = int(n_regimes)
        self.n_knots = int(n_knots)
        self.spline_degree = int(spline_degree)
        self.random_state = int(random_state)

    def fit(self, q, settings):
        q = np.asarray(q, dtype=float).reshape(-1, 1)
        settings = np.asarray(settings, dtype=float)

        if settings.ndim != 2 or settings.shape[1] != 3:
            raise ValueError("settings must have shape (n_samples, 3).")

        if len(q) != len(settings):
            raise ValueError("q and settings must contain the same number of rows.")

        self.setting_scaler_ = StandardScaler()
        settings_z = self.setting_scaler_.fit_transform(settings)

        self.rul_spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.spline_degree,
            include_bias=False,
        )
        self.rul_spline_.fit(q)

        self.regime_model_ = KMeans(
            n_clusters=self.n_regimes,
            n_init=20,
            random_state=self.random_state,
        )
        self.regime_model_.fit(settings_z)

        return self

    def transform(self, q, settings):
        if not hasattr(self, "setting_scaler_"):
            raise RuntimeError("Call fit() before transform().")

        q = np.asarray(q, dtype=float).reshape(-1, 1)
        settings = np.asarray(settings, dtype=float)

        settings_z = self.setting_scaler_.transform(settings)
        spline = self.rul_spline_.transform(q)

        regime = self.regime_model_.predict(settings_z)

        one_hot = np.zeros(
            (len(regime), self.n_regimes),
            dtype=float,
        )
        one_hot[np.arange(len(regime)), regime] = 1.0

        interaction = (
            one_hot[:, :, None]
            * spline[:, None, :]
        ).reshape(len(regime), -1)

        return np.column_stack(
            [
                np.ones(len(q), dtype=float),
                spline,
                settings_z,
                one_hot,
                interaction,
            ]
        )

    def fit_transform(self, q, settings):
        return self.fit(q, settings).transform(q, settings)

    def transform_settings(self, settings):
        """
        Standardize the three settings for the downstream regression model.

        These standardized settings should be appended AFTER the raw/quotient
        sensor representation and should never be quotient-projected.
        """
        if not hasattr(self, "setting_scaler_"):
            raise RuntimeError("Call fit() before transform_settings().")

        settings = np.asarray(settings, dtype=float)
        return self.setting_scaler_.transform(settings)


def read_fd004_txt(path):
    """
    Read a NASA C-MAPSS FD004 train/test text file.

    The original file has no header and contains:
        engine_id, cycle,
        3 operating settings,
        21 sensor measurements.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=FD004_COLUMNS,
        engine="python",
    )

    if df.shape[1] != 26:
        raise ValueError(
            f"{path} should contain 26 columns, got {df.shape[1]}."
        )

    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df


def read_rul_file(path):
    """
    Read RUL_FD004.txt.

    Row i contains the RUL at the last observed cycle of test engine i.
    """
    path = Path(path)

    rul = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
    ).iloc[:, 0].to_numpy(dtype=float)

    return rul


def add_train_rul(df, cap=125.0):
    """
    Add true and capped RUL to complete run-to-failure training trajectories.

        RUL_{e,t} = max_cycle_e - t.
    """
    df = df.copy()

    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["rul_true"] = max_cycle - df["cycle"]
    df["rul"] = np.minimum(df["rul_true"], float(cap))

    return df


def add_test_rul(df, last_rul, cap=125.0):
    """
    Recover the RUL for every observed test cycle.

    If T_obs(e) is the last observed cycle and R_e is the official RUL label
    at that last cycle, then

        RUL_{e,t} = R_e + T_obs(e) - t.
    """
    df = df.copy()

    engines = np.sort(df["engine_id"].unique())
    last_rul = np.asarray(last_rul, dtype=float).reshape(-1)

    if len(engines) != len(last_rul):
        raise ValueError(
            "Number of test engines does not match number of RUL labels."
        )

    label_map = dict(zip(engines, last_rul))
    max_cycle_map = df.groupby("engine_id")["cycle"].max().to_dict()

    true_rul = np.empty(len(df), dtype=float)

    for pos, row in enumerate(df[["engine_id", "cycle"]].itertuples(index=False)):
        e = int(row.engine_id)
        t = int(row.cycle)

        true_rul[pos] = (
            float(label_map[e])
            + float(max_cycle_map[e] - t)
        )

    df["rul_true"] = true_rul
    df["rul"] = np.minimum(true_rul, float(cap))

    return df


def split_training_engines(
    df,
    n_valid_engines=50,
    seed=42,
    valid_engine_ids=None,
):
    """
    Split the official FD004 training engines at the ENGINE level.

    For exact paper reproduction, it is preferable to pass the stored
    valid_engine_ids explicitly rather than relying on a random seed.
    """
    engines = np.sort(df["engine_id"].unique())

    if valid_engine_ids is None:
        rng = np.random.default_rng(int(seed))

        if n_valid_engines >= len(engines):
            raise ValueError("Validation set must contain fewer engines than train.")

        valid_engine_ids = np.sort(
            rng.choice(
                engines,
                size=int(n_valid_engines),
                replace=False,
            )
        )
    else:
        valid_engine_ids = np.sort(
            np.asarray(valid_engine_ids, dtype=int)
        )

        missing = np.setdiff1d(valid_engine_ids, engines)
        if len(missing):
            raise ValueError(
                f"Unknown validation engine ids: {missing.tolist()}"
            )

    train_engine_ids = np.setdiff1d(
        engines,
        valid_engine_ids,
        assume_unique=False,
    )

    return train_engine_ids, valid_engine_ids


def _linear_slope(values):
    """
    Least-squares slope of a 1D sequence against equally spaced time points
    0,1,...,L-1.
    """
    values = np.asarray(values, dtype=float)
    L = len(values)

    if L < 2:
        return 0.0

    t = np.arange(L, dtype=float)
    t_centered = t - t.mean()
    y_centered = values - values.mean()

    denom = np.sum(t_centered ** 2)

    if denom <= 0:
        return 0.0

    return float(
        np.sum(t_centered * y_centered) / denom
    )


def _window_sensor_features(window):
    """
    Convert one L x 21 sensor window to a 63-dimensional feature vector.

    Feature order:
        sensor_1_mean, sensor_1_std, sensor_1_slope,
        sensor_2_mean, sensor_2_std, sensor_2_slope,
        ...
        sensor_21_mean, sensor_21_std, sensor_21_slope.
    """
    sensor_matrix = window[SENSOR_COLUMNS].to_numpy(dtype=float)

    features = []

    for j in range(sensor_matrix.shape[1]):
        v = sensor_matrix[:, j]

        features.extend(
            [
                float(np.mean(v)),
                float(np.std(v, ddof=0)),
                _linear_slope(v),
            ]
        )

    return np.asarray(features, dtype=float)


def make_windows(
    df,
    window_length=20,
    stride=2,
):
    """
    Construct full-length sliding windows independently within each engine.

    The label and operating settings of a window are taken from its END cycle.

    Only full windows are used. This avoids silently inventing early-cycle
    observations. If a different padding convention is required for a specific
    reproduction run, implement it explicitly rather than mixing conventions.
    """
    window_length = int(window_length)
    stride = int(stride)

    if window_length < 2:
        raise ValueError("window_length must be at least 2.")

    if stride < 1:
        raise ValueError("stride must be positive.")

    Y_rows = []
    q_rows = []
    settings_rows = []
    engine_rows = []
    cycle_rows = []

    for engine_id, g in df.groupby("engine_id", sort=True):
        g = g.sort_values("cycle").reset_index(drop=True)

        n = len(g)

        for start in range(0, n - window_length + 1, stride):
            end = start + window_length
            window = g.iloc[start:end]
            endpoint = window.iloc[-1]

            Y_rows.append(
                _window_sensor_features(window)
            )
            q_rows.append(
                float(endpoint["rul"])
            )
            settings_rows.append(
                endpoint[SETTING_COLUMNS].to_numpy(dtype=float)
            )
            engine_rows.append(int(engine_id))
            cycle_rows.append(int(endpoint["cycle"]))

    if not Y_rows:
        raise ValueError("No full windows were created.")

    return FD004Partition(
        Y=np.vstack(Y_rows),
        q=np.asarray(q_rows, dtype=float),
        settings=np.vstack(settings_rows),
        engine_id=np.asarray(engine_rows, dtype=int),
        cycle=np.asarray(cycle_rows, dtype=int),
    )


def subset_by_engines(df, engine_ids):
    engine_ids = np.asarray(engine_ids, dtype=int)
    return df[df["engine_id"].isin(engine_ids)].copy()


def load_fd004(
    train_path,
    test_path,
    rul_path,
    n_valid_engines=50,
    split_seed=42,
    valid_engine_ids=None,
    rul_cap=125.0,
    window_length=20,
    stride=2,
):
    """
    Complete FD004 data-entry pipeline.

    Returns
    -------
    FD004Data
        train / validation / official-test window-level partitions.

    Important
    ---------
    * The official test set is never used to choose the train/validation split.
    * The split is performed at the engine level before window construction.
    * Residual Batch PCA should later be fitted using ONLY data.train.
    """
    train_raw = read_fd004_txt(train_path)
    test_raw = read_fd004_txt(test_path)
    last_rul = read_rul_file(rul_path)

    train_raw = add_train_rul(
        train_raw,
        cap=rul_cap,
    )

    test_raw = add_test_rul(
        test_raw,
        last_rul=last_rul,
        cap=rul_cap,
    )

    train_engine_ids, valid_engine_ids = split_training_engines(
        train_raw,
        n_valid_engines=n_valid_engines,
        seed=split_seed,
        valid_engine_ids=valid_engine_ids,
    )

    train_df = subset_by_engines(
        train_raw,
        train_engine_ids,
    )

    valid_df = subset_by_engines(
        train_raw,
        valid_engine_ids,
    )

    train = make_windows(
        train_df,
        window_length=window_length,
        stride=stride,
    )

    valid = make_windows(
        valid_df,
        window_length=window_length,
        stride=stride,
    )

    test = make_windows(
        test_raw,
        window_length=window_length,
        stride=stride,
    )

    return FD004Data(
        train=train,
        valid=valid,
        test=test,
        train_engine_ids=train_engine_ids,
        valid_engine_ids=valid_engine_ids,
    )


def build_fd004_forward_design(
    data,
    n_regimes=6,
    n_knots=5,
    spline_degree=3,
    random_state=42,
):
    """
    Fit the FD004 forward-design transformations on training windows only.

    Returns
    -------
    design_builder
    X_train_forward
    X_valid_forward
    X_test_forward

    Only X_train_forward is needed by ResidualBatchPCA.fit().
    Validation/test transforms are returned because later components such as
    diagnostics or a Fisher discriminator may need the same design.
    """
    design = FD004ForwardDesign(
        n_regimes=n_regimes,
        n_knots=n_knots,
        spline_degree=spline_degree,
        random_state=random_state,
    )

    X_train = design.fit_transform(
        data.train.q,
        data.train.settings,
    )

    X_valid = design.transform(
        data.valid.q,
        data.valid.settings,
    )

    X_test = design.transform(
        data.test.q,
        data.test.settings,
    )

    return design, X_train, X_valid, X_test


def append_standardized_settings(
    sensor_representation,
    settings,
    design_builder,
):
    """
    Append standardized operating settings to a 63D raw/quotient sensor matrix.

    This produces the 66-dimensional downstream input used by FD004.
    """
    sensor_representation = np.asarray(
        sensor_representation,
        dtype=float,
    )

    settings_z = design_builder.transform_settings(settings)

    if len(sensor_representation) != len(settings_z):
        raise ValueError(
            "sensor_representation and settings must have the same number of rows."
        )

    return np.column_stack(
        [
            sensor_representation,
            settings_z,
        ]
    )


def describe_fd004(data):
    """
    Return a compact summary useful for logging/reproducibility.
    """
    return {
        "train_windows": int(len(data.train.q)),
        "valid_windows": int(len(data.valid.q)),
        "test_windows": int(len(data.test.q)),
        "train_engines": int(len(np.unique(data.train.engine_id))),
        "valid_engines": int(len(np.unique(data.valid.engine_id))),
        "test_engines": int(len(np.unique(data.test.engine_id))),
        "sensor_feature_dim": int(data.train.Y.shape[1]),
    }
