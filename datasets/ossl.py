from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.preprocessing import SplineTransformer


SOURCE_COLUMN = "dataset.code_ascii_txt"
UUID_COLUMN = "id.layer_uuid_txt"
OC_COLUMN = "oc_usda.c729_w.pct"

TRAIN_SOURCES = (
    "KSSL",
    "AFSIS1",
    "LUCAS.WOODWELL",
)

VALID_SOURCES = (
    "AFSIS2",
    "SCHIEDUNG",
    "SERBIA",
)

TEST_SOURCES = (
    "ICRAF.ISRIC",
)


@dataclass
class OSSLPartition:
    """
    OSSL MIR organic-carbon partition.

    Y:
        426 MIR absorbance features from 600 to 4000 cm^-1 in 8 cm^-1 steps.

    q:
        Organic carbon target in weight percent.

    source:
        Normalized source/laboratory label. This is the nuisance-group label.

    sample_id:
        OSSL global soil-layer UUID.
    """
    Y: np.ndarray
    q: np.ndarray
    source: np.ndarray
    sample_id: np.ndarray


@dataclass
class OSSLData:
    train: OSSLPartition
    valid: OSSLPartition
    test: OSSLPartition
    mir_columns: list[str]


class OSSLForwardDesign:
    """
    Cubic-spline forward design for

        E[MIR spectrum | organic carbon].

    The spline is fitted using training observations only.
    """

    def __init__(
        self,
        n_knots=7,
        degree=3,
    ):
        self.n_knots = int(n_knots)
        self.degree = int(degree)

    def fit(self, q):
        q = np.asarray(q, dtype=float).reshape(-1, 1)

        self.spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=False,
        )

        self.spline_.fit(q)
        return self

    def transform(self, q):
        if not hasattr(self, "spline_"):
            raise RuntimeError("Call fit() before transform().")

        q = np.asarray(q, dtype=float).reshape(-1, 1)
        spline = self.spline_.transform(q)

        return np.column_stack(
            [
                np.ones(len(q), dtype=float),
                spline,
            ]
        )

    def fit_transform(self, q):
        return self.fit(q).transform(q)


def normalize_source_name(value):
    """
    Normalize OSSL source strings such as

        AFSIS1.SSL -> AFSIS1
        ICRAF.ISRIC.SSL -> ICRAF.ISRIC

    Only a terminal '.SSL' suffix is removed.
    """
    value = str(value).strip().upper()
    value = re.sub(r"\.SSL$", "", value)
    return value


def _parse_mir_wavenumber(column):
    """
    Extract the wavenumber from a column like:
        scan_mir.600_abs
        scan_mir.602_abs
        ...
    """
    match = re.fullmatch(
        r"scan_mir\.(\d+)_abs",
        str(column),
    )

    if match is None:
        return None

    return int(match.group(1))


def select_mir_columns(
    columns,
    start=600,
    stop=4000,
    step=8,
):
    """
    Select the exact 426-column MIR grid used by the experiment:

        600, 608, 616, ..., 4000 cm^-1.

    The OSSL database may contain a denser MIR grid. This function explicitly
    subsamples to the experiment grid.
    """
    start = int(start)
    stop = int(stop)
    step = int(step)

    desired = set(range(start, stop + 1, step))

    found = {}

    for col in columns:
        wn = _parse_mir_wavenumber(col)

        if wn is not None and wn in desired:
            found[wn] = col

    missing = sorted(desired.difference(found))

    if missing:
        raise ValueError(
            "Missing required MIR wavenumbers. "
            f"First missing values: {missing[:20]}"
        )

    ordered = [
        found[wn]
        for wn in range(start, stop + 1, step)
    ]

    if len(ordered) != 426:
        raise RuntimeError(
            f"Expected 426 MIR columns, found {len(ordered)}."
        )

    return ordered


def read_ossl_all_l1(path):
    """
    Read the integrated OSSL all-L1 csv or csv.gz file.

    pandas automatically handles .gz compression.
    """
    path = Path(path)
    return pd.read_csv(path, low_memory=False)


def read_ossl_joined(
    mir_path,
    soillab_l1_path,
):
    """
    Read separate MIR-L0 and soil-lab-L1 files and join them using the two
    official OSSL joining keys.

    This is useful when the user downloads:
        ossl_mir_L0_v1.2.csv.gz
        ossl_soillab_L1_v1.2.csv.gz
    """
    mir = pd.read_csv(
        Path(mir_path),
        low_memory=False,
    )

    soil = pd.read_csv(
        Path(soillab_l1_path),
        low_memory=False,
    )

    keys = [
        SOURCE_COLUMN,
        UUID_COLUMN,
    ]

    missing_mir = [k for k in keys if k not in mir.columns]
    missing_soil = [k for k in keys if k not in soil.columns]

    if missing_mir:
        raise ValueError(
            f"MIR file is missing join columns: {missing_mir}"
        )

    if missing_soil:
        raise ValueError(
            f"Soil-lab file is missing join columns: {missing_soil}"
        )

    return mir.merge(
        soil,
        on=keys,
        how="left",
        suffixes=("", "_soil"),
        validate="many_to_one",
    )


def prepare_ossl_dataframe(
    df,
    oc_column=OC_COLUMN,
):
    """
    Keep only the fields needed by the OSSL-MIR organic-carbon experiment.

    Rows are retained only when:
        * source is known,
        * UUID is known,
        * organic carbon is observed and finite,
        * all 426 required MIR ordinates are observed and finite.
    """
    required_metadata = [
        SOURCE_COLUMN,
        UUID_COLUMN,
        oc_column,
    ]

    missing = [
        c for c in required_metadata
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"OSSL table is missing required columns: {missing}"
        )

    mir_columns = select_mir_columns(df.columns)

    keep = required_metadata + mir_columns
    out = df.loc[:, keep].copy()

    out["source_norm"] = (
        out[SOURCE_COLUMN]
        .map(normalize_source_name)
    )

    # Coerce target and MIR values to numeric.
    out[oc_column] = pd.to_numeric(
        out[oc_column],
        errors="coerce",
    )

    out[mir_columns] = out[mir_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Remove rows without the requested MIR range or OC target.
    finite_oc = np.isfinite(
        out[oc_column].to_numpy(dtype=float)
    )

    finite_mir = np.isfinite(
        out[mir_columns].to_numpy(dtype=float)
    ).all(axis=1)

    valid_uuid = out[UUID_COLUMN].notna().to_numpy()
    valid_source = out["source_norm"].notna().to_numpy()

    mask = (
        finite_oc
        & finite_mir
        & valid_uuid
        & valid_source
    )

    out = out.loc[mask].reset_index(drop=True)

    return out, mir_columns


def _make_partition(
    df,
    mir_columns,
    oc_column=OC_COLUMN,
):
    return OSSLPartition(
        Y=df[mir_columns].to_numpy(dtype=float),
        q=df[oc_column].to_numpy(dtype=float),
        source=df["source_norm"].to_numpy(dtype=str),
        sample_id=df[UUID_COLUMN].astype(str).to_numpy(),
    )


def load_ossl(
    all_l1_path=None,
    mir_l0_path=None,
    soillab_l1_path=None,
    oc_column=OC_COLUMN,
    train_sources=TRAIN_SOURCES,
    valid_sources=VALID_SOURCES,
    test_sources=TEST_SOURCES,
):
    """
    Complete OSSL-MIR organic-carbon data-entry pipeline.

    Supply either:
        all_l1_path

    OR:
        mir_l0_path + soillab_l1_path.

    Source-level split
    ------------------
    train:
        KSSL, AFSIS1, LUCAS.WOODWELL

    validation:
        AFSIS2, SCHIEDUNG, SERBIA

    test:
        ICRAF.ISRIC

    The split is source-level: no source may appear in more than one partition.
    """
    if all_l1_path is not None:
        if mir_l0_path is not None or soillab_l1_path is not None:
            raise ValueError(
                "Use either all_l1_path OR separate MIR/soil-lab files, not both."
            )

        df = read_ossl_all_l1(all_l1_path)

    else:
        if mir_l0_path is None or soillab_l1_path is None:
            raise ValueError(
                "Provide all_l1_path or both mir_l0_path and soillab_l1_path."
            )

        df = read_ossl_joined(
            mir_path=mir_l0_path,
            soillab_l1_path=soillab_l1_path,
        )

    df, mir_columns = prepare_ossl_dataframe(
        df,
        oc_column=oc_column,
    )

    train_sources = {
        normalize_source_name(s)
        for s in train_sources
    }
    valid_sources = {
        normalize_source_name(s)
        for s in valid_sources
    }
    test_sources = {
        normalize_source_name(s)
        for s in test_sources
    }

    if (
        train_sources & valid_sources
        or train_sources & test_sources
        or valid_sources & test_sources
    ):
        raise ValueError(
            "Train/validation/test source sets must be disjoint."
        )

    train_df = df[
        df["source_norm"].isin(train_sources)
    ].copy()

    valid_df = df[
        df["source_norm"].isin(valid_sources)
    ].copy()

    test_df = df[
        df["source_norm"].isin(test_sources)
    ].copy()

    if len(train_df) == 0:
        raise ValueError("No usable OSSL training spectra were found.")

    if len(valid_df) == 0:
        raise ValueError("No usable OSSL validation spectra were found.")

    if len(test_df) == 0:
        raise ValueError("No usable OSSL test spectra were found.")

    return OSSLData(
        train=_make_partition(
            train_df,
            mir_columns,
            oc_column=oc_column,
        ),
        valid=_make_partition(
            valid_df,
            mir_columns,
            oc_column=oc_column,
        ),
        test=_make_partition(
            test_df,
            mir_columns,
            oc_column=oc_column,
        ),
        mir_columns=list(mir_columns),
    )


def make_equal_source_weights(source):
    """
    Observation weights inversely proportional to source sample size.

    The weights are normalized so that:
        * every source has the same total weight;
        * the mean observation weight equals 1.

    These weights are intended for the OSSL forward ridge regression.
    """
    source = np.asarray(source).astype(str)

    unique, counts = np.unique(
        source,
        return_counts=True,
    )

    count_map = dict(zip(unique, counts))

    weights = np.asarray(
        [
            1.0 / count_map[s]
            for s in source
        ],
        dtype=float,
    )

    # Normalize to mean weight = 1 while preserving equal total source weight.
    weights *= len(weights) / weights.sum()

    return weights


def build_ossl_forward_design(
    data,
    n_knots=7,
    degree=3,
):
    """
    Fit the organic-carbon spline design using TRAINING observations only.

    Returns
    -------
    design_builder
    X_train_forward
    X_valid_forward
    X_test_forward
    """
    design = OSSLForwardDesign(
        n_knots=n_knots,
        degree=degree,
    )

    X_train = design.fit_transform(
        data.train.q
    )

    X_valid = design.transform(
        data.valid.q
    )

    X_test = design.transform(
        data.test.q
    )

    return design, X_train, X_valid, X_test


def mean_pool_mir_426_to_71(X):
    """
    Deterministically pool the 426 MIR ordinates into 71 downstream features.

    The operation is applied AFTER raw/quotient representation construction:

        426 -> reshape (71, 6) -> mean over each consecutive block of 6.

    There are no learned parameters and no padding because:
        426 = 71 * 6.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[1] != 426:
        raise ValueError(
            f"Expected X with shape (n_samples, 426), got {X.shape}."
        )

    return X.reshape(
        len(X),
        71,
        6,
    ).mean(axis=2)


def describe_ossl(data):
    """
    Compact split/source summary useful for experiment logs.
    """
    return {
        "train_samples": int(len(data.train.q)),
        "valid_samples": int(len(data.valid.q)),
        "test_samples": int(len(data.test.q)),
        "train_sources": sorted(np.unique(data.train.source).tolist()),
        "valid_sources": sorted(np.unique(data.valid.source).tolist()),
        "test_sources": sorted(np.unique(data.test.source).tolist()),
        "mir_feature_dim": int(data.train.Y.shape[1]),
    }
