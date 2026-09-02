from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import (
    SplineTransformer,
    StandardScaler,
)


FD004_COLUMNS = (
    [
        "engine_id",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
    ]
    + [
        f"sensor_{j}"
        for j in range(1, 22)
    ]
)

SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]

SENSOR_COLUMNS = [
    f"sensor_{j}"
    for j in range(1, 22)
]


@dataclass
class FD004Partition:
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
    Training-only forward design for E[Y | RUL, settings].

    Columns:
        intercept
        spline(capped RUL)
        standardized operating settings
        six operating-regime indicators
        spline(RUL) x regime interactions
    """

    def __init__(
        self,
        n_regimes=6,
        n_knots=5,
        spline_degree=3,
        random_state=42,
    ):
        self.n_regimes = int(n_regimes)
        self.n_knots = int(n_knots)
        self.spline_degree = int(
            spline_degree
        )
        self.random_state = int(
            random_state
        )

    def fit(
        self,
        q,
        settings,
    ):
        q = np.asarray(
            q,
            dtype=float,
        ).reshape(-1, 1)

        settings = np.asarray(
            settings,
            dtype=float,
        )

        if (
            settings.ndim != 2
            or settings.shape[1] != 3
        ):
            raise ValueError(
                "settings must have shape (n, 3)."
            )

        self.setting_scaler_ = (
            StandardScaler()
        )

        settings_z = (
            self.setting_scaler_
            .fit_transform(settings)
        )

        self.rul_spline_ = (
            SplineTransformer(
                n_knots=self.n_knots,
                degree=self.spline_degree,
                include_bias=False,
            )
        )

        self.rul_spline_.fit(q)

        self.regime_model_ = KMeans(
            n_clusters=self.n_regimes,
            n_init=20,
            random_state=self.random_state,
        )

        self.regime_model_.fit(
            settings_z
        )

        return self

    def transform(
        self,
        q,
        settings,
    ):
        if not hasattr(
            self,
            "setting_scaler_",
        ):
            raise RuntimeError(
                "Call fit() first."
            )

        q = np.asarray(
            q,
            dtype=float,
        ).reshape(-1, 1)

        settings = np.asarray(
            settings,
            dtype=float,
        )

        settings_z = (
            self.setting_scaler_
            .transform(settings)
        )

        spline = (
            self.rul_spline_
            .transform(q)
        )

        regime = (
            self.regime_model_
            .predict(settings_z)
        )

        one_hot = np.zeros(
            (
                len(regime),
                self.n_regimes,
            ),
            dtype=float,
        )

        one_hot[
            np.arange(len(regime)),
            regime,
        ] = 1.0

        interaction = (
            one_hot[:, :, None]
            * spline[:, None, :]
        ).reshape(
            len(regime),
            -1,
        )

        return np.column_stack(
            [
                np.ones(
                    len(q),
                    dtype=float,
                ),
                spline,
                settings_z,
                one_hot,
                interaction,
            ]
        )

    def fit_transform(
        self,
        q,
        settings,
    ):
        return (
            self.fit(q, settings)
            .transform(q, settings)
        )

    def transform_settings(
        self,
        settings,
    ):
        if not hasattr(
            self,
            "setting_scaler_",
        ):
            raise RuntimeError(
                "Call fit() first."
            )

        return (
            self.setting_scaler_
            .transform(
                np.asarray(
                    settings,
                    dtype=float,
                )
            )
        )


def read_fd004_txt(path):
    df = pd.read_csv(
        Path(path),
        sep=r"\s+",
        header=None,
        names=FD004_COLUMNS,
        engine="python",
    )

    if df.shape[1] != 26:
        raise ValueError(
            "FD004 file must contain 26 columns."
        )

    df["engine_id"] = (
        df["engine_id"].astype(int)
    )

    df["cycle"] = (
        df["cycle"].astype(int)
    )

    return df


def read_rul_file(path):
    return (
        pd.read_csv(
            Path(path),
            sep=r"\s+",
            header=None,
            engine="python",
        )
        .iloc[:, 0]
        .to_numpy(dtype=float)
    )


def read_engine_ids_file(path):
    """
    Read a fixed validation-engine list.

    Accepted formats include one ID per line or any whitespace-separated
    collection of integer IDs.
    """
    text = Path(path).read_text(
        encoding="utf-8"
    )

    ids = np.asarray(
        [
            int(token)
            for token in text.split()
        ],
        dtype=int,
    )

    if len(ids) == 0:
        raise ValueError(
            "Engine-ID file is empty."
        )

    if len(np.unique(ids)) != len(ids):
        raise ValueError(
            "Engine-ID file contains duplicates."
        )

    return np.sort(ids)


def add_train_rul(
    df,
    cap=125.0,
):
    df = df.copy()

    max_cycle = (
        df.groupby("engine_id")["cycle"]
        .transform("max")
    )

    df["rul_true"] = (
        max_cycle
        - df["cycle"]
    )

    df["rul"] = np.minimum(
        df["rul_true"],
        float(cap),
    )

    return df


def add_test_rul(
    df,
    last_rul,
    cap=125.0,
):
    df = df.copy()

    engines = np.sort(
        df["engine_id"].unique()
    )

    last_rul = np.asarray(
        last_rul,
        dtype=float,
    ).reshape(-1)

    if len(engines) != len(last_rul):
        raise ValueError(
            "Number of test engines does not match RUL labels."
        )

    last_rul_map = dict(
        zip(
            engines,
            last_rul,
        )
    )

    max_cycle_map = (
        df.groupby("engine_id")["cycle"]
        .max()
        .to_dict()
    )

    true_rul = np.empty(
        len(df),
        dtype=float,
    )

    for pos, row in enumerate(
        df[
            [
                "engine_id",
                "cycle",
            ]
        ].itertuples(
            index=False
        )
    ):
        e = int(row.engine_id)
        t = int(row.cycle)

        true_rul[pos] = (
            float(last_rul_map[e])
            + float(
                max_cycle_map[e] - t
            )
        )

    df["rul_true"] = true_rul

    df["rul"] = np.minimum(
        true_rul,
        float(cap),
    )

    return df


def split_training_engines(
    df,
    n_valid_engines=50,
    seed=42,
    valid_engine_ids=None,
):
    engines = np.sort(
        df["engine_id"].unique()
    )

    if valid_engine_ids is None:
        rng = np.random.default_rng(
            int(seed)
        )

        valid_engine_ids = np.sort(
            rng.choice(
                engines,
                size=int(
                    n_valid_engines
                ),
                replace=False,
            )
        )

    else:
        valid_engine_ids = np.sort(
            np.asarray(
                valid_engine_ids,
                dtype=int,
            )
        )

        if (
            len(valid_engine_ids)
            != int(n_valid_engines)
        ):
            raise ValueError(
                "Expected exactly "
                f"{n_valid_engines} validation engines, "
                f"got {len(valid_engine_ids)}."
            )

        missing = np.setdiff1d(
            valid_engine_ids,
            engines,
        )

        if len(missing):
            raise ValueError(
                "Unknown validation engine IDs: "
                f"{missing.tolist()}"
            )

    train_engine_ids = np.setdiff1d(
        engines,
        valid_engine_ids,
    )

    return (
        train_engine_ids,
        valid_engine_ids,
    )


def _linear_slope(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    L = len(values)

    if L < 2:
        return 0.0

    t = np.arange(
        L,
        dtype=float,
    )

    t_centered = (
        t - t.mean()
    )

    y_centered = (
        values
        - values.mean()
    )

    denom = np.sum(
        t_centered ** 2
    )

    if denom <= 0:
        return 0.0

    return float(
        np.sum(
            t_centered
            * y_centered
        )
        / denom
    )



def _window_sensor_features(
    sensor_matrix,
):
    """
    Vectorized 21-sensor summary:
        mean, population std, least-squares slope
    for each sensor, yielding 63 features.
    """
    sensor_matrix = np.asarray(
        sensor_matrix,
        dtype=float,
    )

    if (
        sensor_matrix.ndim != 2
        or sensor_matrix.shape[1] != 21
    ):
        raise ValueError(
            "sensor_matrix must have shape (L, 21)."
        )

    L = sensor_matrix.shape[0]

    means = sensor_matrix.mean(
        axis=0
    )

    stds = sensor_matrix.std(
        axis=0,
        ddof=0,
    )

    if L < 2:
        slopes = np.zeros(
            21,
            dtype=float,
        )
    else:
        t = np.arange(
            L,
            dtype=float,
        )
        tc = t - t.mean()
        denom = np.sum(
            tc ** 2
        )

        slopes = (
            tc @ (
                sensor_matrix
                - means[None, :]
            )
        ) / denom

    return np.column_stack(
        [
            means,
            stds,
            slopes,
        ]
    ).reshape(-1)


def _left_pad_sensor_matrix(
    sensor_matrix,
    target_length,
):
    sensor_matrix = np.asarray(
        sensor_matrix,
        dtype=float,
    )

    n = len(sensor_matrix)

    if n >= target_length:
        return sensor_matrix

    pad_n = (
        target_length - n
    )

    padding = np.repeat(
        sensor_matrix[[0]],
        pad_n,
        axis=0,
    )

    return np.vstack(
        [
            padding,
            sensor_matrix,
        ]
    )


def make_windows(
    df,
    window_length=20,
    stride=2,
    pad_short_engines=False,
    force_endpoint=True,
):
    """
    Build engine-wise sliding windows.

    Rules
    -----
    * full windows use length 20 and stride 2;
    * the final observed cycle of every engine is always represented;
    * for official test engines shorter than 20 cycles, the first
      observation is repeated on the left until length 20.

    On the standard NASA FD004 test files these rules yield exactly
    18,429 official-test windows.
    """
    window_length = int(
        window_length
    )
    stride = int(
        stride
    )

    if window_length < 2:
        raise ValueError(
            "window_length must be at least 2."
        )

    if stride < 1:
        raise ValueError(
            "stride must be positive."
        )

    Y_rows = []
    q_rows = []
    settings_rows = []
    engine_rows = []
    cycle_rows = []

    for engine_id, g in df.groupby(
        "engine_id",
        sort=True,
    ):
        g = g.sort_values(
            "cycle"
        )

        sensors_all = g[
            SENSOR_COLUMNS
        ].to_numpy(
            dtype=float
        )

        settings_all = g[
            SETTING_COLUMNS
        ].to_numpy(
            dtype=float
        )

        q_all = g[
            "rul"
        ].to_numpy(
            dtype=float
        )

        cycles_all = g[
            "cycle"
        ].to_numpy(
            dtype=int
        )

        n = len(g)

        # Entire observed trajectory shorter than one full window.
        if n < window_length:
            if not pad_short_engines:
                continue

            padded = _left_pad_sensor_matrix(
                sensors_all,
                window_length,
            )

            Y_rows.append(
                _window_sensor_features(
                    padded
                )
            )

            q_rows.append(
                float(q_all[-1])
            )

            settings_rows.append(
                settings_all[-1]
            )

            engine_rows.append(
                int(engine_id)
            )

            cycle_rows.append(
                int(cycles_all[-1])
            )

            continue

        last_start = (
            n - window_length
        )

        starts = list(
            range(
                0,
                last_start + 1,
                stride,
            )
        )

        # With stride 2, the parity of the trajectory length can otherwise
        # cause the true final observed cycle to be absent from all windows.
        if (
            force_endpoint
            and starts[-1] != last_start
        ):
            starts.append(
                last_start
            )

        for start in starts:
            end = (
                start
                + window_length
            )

            window_sensors = (
                sensors_all[
                    start:end
                ]
            )

            endpoint_idx = (
                end - 1
            )

            Y_rows.append(
                _window_sensor_features(
                    window_sensors
                )
            )

            q_rows.append(
                float(
                    q_all[
                        endpoint_idx
                    ]
                )
            )

            settings_rows.append(
                settings_all[
                    endpoint_idx
                ]
            )

            engine_rows.append(
                int(engine_id)
            )

            cycle_rows.append(
                int(
                    cycles_all[
                        endpoint_idx
                    ]
                )
            )

    if not Y_rows:
        raise ValueError(
            "No windows were created."
        )

    return FD004Partition(
        Y=np.vstack(
            Y_rows
        ),
        q=np.asarray(
            q_rows,
            dtype=float,
        ),
        settings=np.vstack(
            settings_rows
        ),
        engine_id=np.asarray(
            engine_rows,
            dtype=int,
        ),
        cycle=np.asarray(
            cycle_rows,
            dtype=int,
        ),
    )


def subset_by_engines(
    df,
    engine_ids,
):
    engine_ids = np.asarray(
        engine_ids,
        dtype=int,
    )

    return df[
        df["engine_id"]
        .isin(engine_ids)
    ].copy()


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
    train_raw = read_fd004_txt(
        train_path
    )

    test_raw = read_fd004_txt(
        test_path
    )

    last_rul = read_rul_file(
        rul_path
    )

    train_raw = add_train_rul(
        train_raw,
        cap=rul_cap,
    )

    test_raw = add_test_rul(
        test_raw,
        last_rul=last_rul,
        cap=rul_cap,
    )

    (
        train_engine_ids,
        valid_engine_ids,
    ) = split_training_engines(
        train_raw,
        n_valid_engines=n_valid_engines,
        seed=split_seed,
        valid_engine_ids=(
            valid_engine_ids
        ),
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
        pad_short_engines=False,
        force_endpoint=True,
    )

    valid = make_windows(
        valid_df,
        window_length=window_length,
        stride=stride,
        pad_short_engines=False,
        force_endpoint=True,
    )

    test = make_windows(
        test_raw,
        window_length=window_length,
        stride=stride,
        pad_short_engines=True,
        force_endpoint=True,
    )

    return FD004Data(
        train=train,
        valid=valid,
        test=test,
        train_engine_ids=(
            train_engine_ids
        ),
        valid_engine_ids=(
            valid_engine_ids
        ),
    )


def build_fd004_forward_design(
    data,
    n_regimes=6,
    n_knots=5,
    spline_degree=3,
    random_state=42,
):
    design = FD004ForwardDesign(
        n_regimes=n_regimes,
        n_knots=n_knots,
        spline_degree=(
            spline_degree
        ),
        random_state=(
            random_state
        ),
    )

    X_train = (
        design.fit_transform(
            data.train.q,
            data.train.settings,
        )
    )

    X_valid = (
        design.transform(
            data.valid.q,
            data.valid.settings,
        )
    )

    X_test = (
        design.transform(
            data.test.q,
            data.test.settings,
        )
    )

    return (
        design,
        X_train,
        X_valid,
        X_test,
    )


def append_standardized_settings(
    sensor_representation,
    settings,
    design_builder,
):
    sensor_representation = (
        np.asarray(
            sensor_representation,
            dtype=float,
        )
    )

    settings_z = (
        design_builder
        .transform_settings(
            settings
        )
    )

    if (
        len(sensor_representation)
        != len(settings_z)
    ):
        raise ValueError(
            "Row counts do not match."
        )

    return np.column_stack(
        [
            sensor_representation,
            settings_z,
        ]
    )


def describe_fd004(data):
    return {
        "train_windows":
            int(len(data.train.q)),

        "valid_windows":
            int(len(data.valid.q)),

        "test_windows":
            int(len(data.test.q)),

        "train_engines":
            int(
                len(
                    np.unique(
                        data.train.engine_id
                    )
                )
            ),

        "valid_engines":
            int(
                len(
                    np.unique(
                        data.valid.engine_id
                    )
                )
            ),

        "test_engines":
            int(
                len(
                    np.unique(
                        data.test.engine_id
                    )
                )
            ),

        "sensor_feature_dim":
            int(
                data.train.Y.shape[1]
            ),
    }
