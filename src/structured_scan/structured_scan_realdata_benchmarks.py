
"""
Final real-data benchmark pipeline for Structured Scan Repair
=============================================================

Datasets
--------
1. Adult Income
   Methodological role:
   - mixed continuous/categorical tabular data
   - interpretable original features
   - established strong tree benchmark
   - standard UCI train/test split available

2. APS Failure at Scania Trucks
   Methodological role:
   - rare positive class
   - high-dimensional numeric operational data
   - substantial missingness
   - standard UCI train/test split
   - very strong published tree discrimination

3. Diabetes 130-US Hospitals
   Methodological role:
   - clinically meaningful probability estimation
   - repeated patient encounters
   - mixed and high-cardinality clinical predictors
   - missingness
   - grouped patient-level splitting required to prevent leakage

Scientific comparison
---------------------
For each dataset:

    tuned XGBoost
        versus
    the SAME tuned XGBoost + frozen Structured Scan Repair

The real-data study is not intended to rank model architectures. The synthetic
backbone study already establishes model generality. The purpose here is to ask
whether a competitive trained model leaves stable, interpretable, scan-
correctable residual structure in real tabular data.

Important scanner representation
--------------------------------
The base predictor may use one-hot encoding, but the scanner receives ORIGINAL
variables:
    numeric variable -> one numeric scan dimension
    categorical variable -> one categorical scan dimension

Thus, several one-hot indicators from one original categorical variable can
never be counted as separate scan dimensions.

For mixed datasets, the protected legacy fixed-bin candidate is disabled
because it assumes an all-numeric ordered representation. The categorical-aware
adaptive search, likelihood objective, stability selection, acceptance
criterion, repair rounds, and joint coefficient refit are otherwise unchanged.

Data access
-----------
The notebook can download public UCI datasets when internet is available, or
use already-downloaded files in DATA_DIR.

UCI sources:
Adult:
  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
APS:
  https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip
Diabetes-130:
  https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math
import os
import re
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from adaptive_structured_scanner_v2_1 import (
    AdaptiveRuleBasis,
    NumericLiteral,
    CategoricalLiteral,
)
from adaptive_structured_scanner_v2_2 import (
    AdaptiveStructuredScannerV22,
)
from adaptive_structured_scanner_v2_3 import (
    StabilitySelectedScanRepairLearner,
)
from mdss_treescan_stage1 import sigmoid


# =============================================================================
# URLs and dataset constants
# =============================================================================

ADULT_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/adult/adult.data"
)
ADULT_TEST_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/adult/adult.test"
)

APS_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/421/"
    "aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip"
)

DIABETES_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
)

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

ADULT_NUMERIC = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


# =============================================================================
# Utility
# =============================================================================

def clip_prob(
    p: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        eps,
        1.0 - eps,
    )


def logit(
    p: np.ndarray,
) -> np.ndarray:
    p = clip_prob(
        p
    )
    return np.log(
        p / (1.0 - p)
    )


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    y = np.asarray(
        y,
        dtype=float,
    )
    p = clip_prob(
        p
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    total = len(
        y
    )
    ece = 0.0

    for lo, hi in zip(
        edges[:-1],
        edges[1:],
    ):
        if hi == 1.0:
            mask = (
                (p >= lo)
                & (p <= hi)
            )
        else:
            mask = (
                (p >= lo)
                & (p < hi)
            )

        if not np.any(
            mask
        ):
            continue

        ece += (
            np.mean(
                mask
            )
            * abs(
                np.mean(
                    y[
                        mask
                    ]
                )
                - np.mean(
                    p[
                        mask
                    ]
                )
            )
        )

    return float(
        ece
    )


def binary_metrics(
    y: np.ndarray,
    p: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(
        y,
        dtype=int,
    )
    p = clip_prob(
        p
    )

    return {
        "logloss": float(
            log_loss(
                y,
                p,
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                p,
            )
        ),
        "auroc": float(
            roc_auc_score(
                y,
                p,
            )
        ),
        "auprc": float(
            average_precision_score(
                y,
                p,
            )
        ),
        "ece_15": expected_calibration_error(
            y,
            p,
            n_bins=15,
        ),
    }


def _download(
    url: str,
    destination: Path,
) -> Path:
    destination = Path(
        destination
    )
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        return destination

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "Structured-Scan-Research/1.0"
            )
        },
    )

    with (
        urllib.request.urlopen(
            request,
            timeout=180,
        )
        as response,
        open(
            destination,
            "wb",
        )
        as f,
    ):
        while True:
            block = response.read(
                1024 * 1024
            )
            if not block:
                break
            f.write(
                block
            )

    return destination


def _download_and_extract_zip(
    url: str,
    destination_dir: Path,
    zip_name: str,
) -> Path:
    destination_dir = Path(
        destination_dir
    )
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_path = (
        destination_dir
        / zip_name
    )

    _download(
        url,
        zip_path,
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zf:
        zf.extractall(
            destination_dir
        )

    return destination_dir


def _find_file(
    root: Path,
    filename: str,
) -> Path:
    root = Path(
        root
    )

    exact = list(
        root.rglob(
            filename
        )
    )

    if not exact:
        raise FileNotFoundError(
            f"Could not find {filename} under {root}."
        )

    return exact[
        0
    ]


# =============================================================================
# Dataset bundle
# =============================================================================

@dataclass
class DatasetBundle:
    name: str
    methodological_role: str

    X: pd.DataFrame
    y: np.ndarray

    numeric_columns: List[str]
    categorical_columns: List[str]

    # Official test supplied by dataset, if applicable.
    official_train_mask: Optional[np.ndarray] = None
    official_test_mask: Optional[np.ndarray] = None

    # Group IDs when rows from the same entity must never cross partitions.
    groups: Optional[np.ndarray] = None

    source_metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# Adult
# =============================================================================

def load_adult(
    data_dir: Path,
    *,
    allow_download: bool = True,
) -> DatasetBundle:
    root = (
        Path(
            data_dir
        )
        / "adult"
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        root
        / "adult.data"
    )
    test_path = (
        root
        / "adult.test"
    )

    if allow_download:
        _download(
            ADULT_TRAIN_URL,
            train_path,
        )
        _download(
            ADULT_TEST_URL,
            test_path,
        )

    if (
        not train_path.exists()
        or not test_path.exists()
    ):
        raise FileNotFoundError(
            "Adult files are missing. Expected adult.data and adult.test "
            f"under {root}."
        )

    train = pd.read_csv(
        train_path,
        header=None,
        names=(
            ADULT_COLUMNS
            + [
                "target"
            ]
        ),
        skipinitialspace=True,
        na_values=[
            "?",
        ],
    )

    test = pd.read_csv(
        test_path,
        header=None,
        names=(
            ADULT_COLUMNS
            + [
                "target"
            ]
        ),
        skiprows=1,
        skipinitialspace=True,
        na_values=[
            "?",
        ],
    )

    train[
        "__official_split__"
    ] = "train"
    test[
        "__official_split__"
    ] = "test"

    df = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
    )

    y_clean = (
        df[
            "target"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.rstrip(
            "."
        )
    )

    valid = y_clean.isin(
        [
            "<=50K",
            ">50K",
        ]
    )

    df = df.loc[
        valid
    ].reset_index(
        drop=True
    )

    y_clean = y_clean.loc[
        valid
    ].reset_index(
        drop=True
    )

    y = (
        y_clean
        == ">50K"
    ).astype(
        int
    ).to_numpy()

    X = df[
        ADULT_COLUMNS
    ].copy()

    for col in ADULT_NUMERIC:
        X[
            col
        ] = pd.to_numeric(
            X[
                col
            ],
            errors="coerce",
        )

    categorical = [
        c
        for c in ADULT_COLUMNS
        if c not in ADULT_NUMERIC
    ]

    official_train_mask = (
        df[
            "__official_split__"
        ].to_numpy()
        == "train"
    )

    official_test_mask = ~official_train_mask

    return DatasetBundle(
        name="Adult Income",
        methodological_role=(
            "Mixed numerical/categorical tabular prediction with "
            "interpretable original features and an established strong "
            "tree benchmark."
        ),
        X=X,
        y=y,
        numeric_columns=list(
            ADULT_NUMERIC
        ),
        categorical_columns=categorical,
        official_train_mask=official_train_mask,
        official_test_mask=official_test_mask,
        groups=None,
        source_metadata={
            "source": "UCI Adult",
            "official_train_n": int(
                np.sum(
                    official_train_mask
                )
            ),
            "official_test_n": int(
                np.sum(
                    official_test_mask
                )
            ),
        },
    )


# =============================================================================
# APS Failure
# =============================================================================

def _read_aps_csv(
    path: Path,
) -> pd.DataFrame:
    """
    UCI APS files contain 20 description lines before the header.
    """
    return pd.read_csv(
        path,
        skiprows=20,
        na_values=[
            "na",
        ],
        low_memory=False,
    )


def load_aps(
    data_dir: Path,
    *,
    allow_download: bool = True,
) -> DatasetBundle:
    root = (
        Path(
            data_dir
        )
        / "aps_failure"
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_filename = (
        "aps_failure_training_set.csv"
    )
    test_filename = (
        "aps_failure_test_set.csv"
    )

    train_candidates = list(
        root.rglob(
            train_filename
        )
    )
    test_candidates = list(
        root.rglob(
            test_filename
        )
    )

    if (
        (
            not train_candidates
        )
        or (
            not test_candidates
        )
    ) and allow_download:
        _download_and_extract_zip(
            APS_ZIP_URL,
            root,
            "aps_failure.zip",
        )

        train_candidates = list(
            root.rglob(
                train_filename
            )
        )
        test_candidates = list(
            root.rglob(
                test_filename
            )
        )

    if (
        not train_candidates
        or not test_candidates
    ):
        raise FileNotFoundError(
            "APS files are missing. Expected aps_failure_training_set.csv "
            f"and aps_failure_test_set.csv under {root}."
        )

    train = _read_aps_csv(
        train_candidates[
            0
        ]
    )
    test = _read_aps_csv(
        test_candidates[
            0
        ]
    )

    train[
        "__official_split__"
    ] = "train"
    test[
        "__official_split__"
    ] = "test"

    df = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
    )

    y = (
        df[
            "class"
        ]
        .astype(
            str
        )
        .str.strip()
        .eq(
            "pos"
        )
        .astype(
            int
        )
        .to_numpy()
    )

    feature_cols = [
        c
        for c in df.columns
        if c not in {
            "class",
            "__official_split__",
        }
    ]

    X = df[
        feature_cols
    ].copy()

    for col in feature_cols:
        X[
            col
        ] = pd.to_numeric(
            X[
                col
            ],
            errors="coerce",
        )

    official_train_mask = (
        df[
            "__official_split__"
        ].to_numpy()
        == "train"
    )

    official_test_mask = ~official_train_mask

    return DatasetBundle(
        name="APS Failure at Scania Trucks",
        methodological_role=(
            "Rare-event, high-dimensional operational prediction with "
            "substantial missingness and a standard held-out test set; "
            "tests whether strong discrimination can coexist with "
            "scan-correctable probability residuals."
        ),
        X=X,
        y=y,
        numeric_columns=feature_cols,
        categorical_columns=[],
        official_train_mask=official_train_mask,
        official_test_mask=official_test_mask,
        groups=None,
        source_metadata={
            "source": "UCI APS Failure",
            "official_train_n": int(
                np.sum(
                    official_train_mask
                )
            ),
            "official_test_n": int(
                np.sum(
                    official_test_mask
                )
            ),
        },
    )


# =============================================================================
# Diabetes 130-US Hospitals
# =============================================================================

DIABETES_NUMERIC = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

DIABETES_CATEGORICAL_ID_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]


def _icd9_chapter(
    value: Any,
) -> str:
    if pd.isna(
        value
    ):
        return "__MISSING__"

    s = str(
        value
    ).strip()

    if (
        s == ""
        or s == "?"
    ):
        return "__MISSING__"

    upper = s.upper()

    if upper.startswith(
        "V"
    ):
        return "supplementary_V"

    if upper.startswith(
        "E"
    ):
        return "external_E"

    try:
        x = float(
            s
        )
    except Exception:
        return "other"

    if (
        250.0
        <= x
        < 251.0
    ):
        return "diabetes"

    code = int(
        math.floor(
            x
        )
    )

    ranges = [
        (
            1,
            139,
            "infectious",
        ),
        (
            140,
            239,
            "neoplasm",
        ),
        (
            240,
            279,
            "endocrine_metabolic",
        ),
        (
            280,
            289,
            "blood",
        ),
        (
            290,
            319,
            "mental",
        ),
        (
            320,
            389,
            "nervous_sense",
        ),
        (
            390,
            459,
            "circulatory",
        ),
        (
            460,
            519,
            "respiratory",
        ),
        (
            520,
            579,
            "digestive",
        ),
        (
            580,
            629,
            "genitourinary",
        ),
        (
            630,
            679,
            "pregnancy",
        ),
        (
            680,
            709,
            "skin",
        ),
        (
            710,
            739,
            "musculoskeletal",
        ),
        (
            740,
            759,
            "congenital",
        ),
        (
            760,
            779,
            "perinatal",
        ),
        (
            780,
            799,
            "symptoms",
        ),
        (
            800,
            999,
            "injury_poisoning",
        ),
    ]

    for lo, hi, label in ranges:
        if (
            lo
            <= code
            <= hi
        ):
            return label

    return "other"


def _parse_diabetes_discharge_exclusions(
    mapping_path: Optional[Path],
) -> set[int]:
    """
    Derive no-readmission-opportunity discharge codes from the supplied
    UCI mapping where possible.

    Descriptions containing 'expired' or 'hospice' are excluded.
    """
    if (
        mapping_path is None
        or not Path(
            mapping_path
        ).exists()
    ):
        # Conservative fallback commonly used with this dataset.
        # These values are retained only as a fallback when the mapping file
        # cannot be parsed.
        return {
            11,
            13,
            14,
            19,
            20,
            21,
        }

    lines = Path(
        mapping_path
    ).read_text(
        errors="ignore"
    ).splitlines()

    in_discharge = False
    exclusions: set[
        int
    ] = set()

    for raw in lines:
        line = raw.strip()

        if not line:
            continue

        low = line.lower()

        if low.startswith(
            "discharge_disposition_id"
        ):
            in_discharge = True
            continue

        if (
            in_discharge
            and (
                low.startswith(
                    "admission_source_id"
                )
                or low.startswith(
                    "admission_type_id"
                )
            )
        ):
            break

        if not in_discharge:
            continue

        parts = [
            p.strip()
            for p in line.split(
                ",",
                1,
            )
        ]

        if len(
            parts
        ) < 2:
            continue

        try:
            code = int(
                parts[
                    0
                ]
            )
        except Exception:
            continue

        description = parts[
            1
        ].lower()

        if (
            "expired"
            in description
            or "hospice"
            in description
        ):
            exclusions.add(
                code
            )

    if exclusions:
        return exclusions

    return {
        11,
        13,
        14,
        19,
        20,
        21,
    }


def load_diabetes130(
    data_dir: Path,
    *,
    allow_download: bool = True,
) -> DatasetBundle:
    root = (
        Path(
            data_dir
        )
        / "diabetes130"
    )
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_candidates = list(
        root.rglob(
            "diabetic_data.csv"
        )
    )

    mapping_candidates = list(
        root.rglob(
            "IDS_mapping.csv"
        )
    )

    if (
        not data_candidates
    ) and allow_download:
        _download_and_extract_zip(
            DIABETES_ZIP_URL,
            root,
            "diabetes130.zip",
        )

        data_candidates = list(
            root.rglob(
                "diabetic_data.csv"
            )
        )
        mapping_candidates = list(
            root.rglob(
                "IDS_mapping.csv"
            )
        )

    if not data_candidates:
        raise FileNotFoundError(
            "Diabetes-130 file is missing. Expected diabetic_data.csv "
            f"under {root}."
        )

    data_path = data_candidates[
        0
    ]

    mapping_path = (
        mapping_candidates[
            0
        ]
        if mapping_candidates
        else None
    )

    df = pd.read_csv(
        data_path,
        na_values=[
            "?",
        ],
        low_memory=False,
    )

    required = {
        "patient_nbr",
        "readmitted",
        "discharge_disposition_id",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            "Diabetes file is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    discharge_exclusions = (
        _parse_diabetes_discharge_exclusions(
            mapping_path
        )
    )

    discharge = pd.to_numeric(
        df[
            "discharge_disposition_id"
        ],
        errors="coerce",
    )

    eligible = ~discharge.isin(
        list(
            discharge_exclusions
        )
    )

    df = df.loc[
        eligible
    ].reset_index(
        drop=True
    )

    y = (
        df[
            "readmitted"
        ]
        .astype(
            str
        )
        .str.strip()
        .eq(
            "<30"
        )
        .astype(
            int
        )
        .to_numpy()
    )

    groups = (
        df[
            "patient_nbr"
        ]
        .astype(
            str
        )
        .to_numpy()
    )

    # Clinically meaningful coarse ICD-9 chapters improve interpretability and
    # prevent hundreds of diagnosis codes from acting as independent arbitrary
    # categories in the scan.
    for col in [
        "diag_1",
        "diag_2",
        "diag_3",
    ]:
        if col in df.columns:
            df[
                f"{col}_chapter"
            ] = df[
                col
            ].map(
                _icd9_chapter
            )

    drop_columns = {
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "diag_1",
        "diag_2",
        "diag_3",
    }

    feature_cols = [
        c
        for c in df.columns
        if c not in drop_columns
    ]

    numeric = [
        c
        for c in DIABETES_NUMERIC
        if c in feature_cols
    ]

    categorical = [
        c
        for c in feature_cols
        if c not in numeric
    ]

    X = df[
        feature_cols
    ].copy()

    for col in numeric:
        X[
            col
        ] = pd.to_numeric(
            X[
                col
            ],
            errors="coerce",
        )

    for col in categorical:
        X[
            col
        ] = (
            X[
                col
            ]
            .astype(
                "object"
            )
        )

    return DatasetBundle(
        name="Diabetes 130-US Hospitals",
        methodological_role=(
            "Clinical early-readmission risk prediction with repeated "
            "patients, mixed/high-cardinality predictors, missing data, "
            "and a probability-focused outcome; tests grouped leakage-safe "
            "residual repair."
        ),
        X=X,
        y=y,
        numeric_columns=numeric,
        categorical_columns=categorical,
        official_train_mask=None,
        official_test_mask=None,
        groups=groups,
        source_metadata={
            "source": "UCI Diabetes 130-US Hospitals",
            "excluded_discharge_codes": sorted(
                discharge_exclusions
            ),
            "eligible_encounters": int(
                len(
                    df
                )
            ),
            "unique_patients": int(
                len(
                    np.unique(
                        groups
                    )
                )
            ),
            "positive_rate": float(
                np.mean(
                    y
                )
            ),
        },
    )


# =============================================================================
# Splitting
# =============================================================================

@dataclass(frozen=True)
class FourWaySplit:
    train: np.ndarray
    discovery: np.ndarray
    selection: np.ndarray
    test: np.ndarray


def _stratified_three_way_with_fixed_test(
    y: np.ndarray,
    development_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    train_fraction_within_dev: float = 0.70,
    discovery_fraction_within_dev: float = 0.20,
) -> FourWaySplit:
    """
    Split official TRAIN into base-training / discovery / selection while
    preserving the supplied official TEST unchanged.
    """
    dev = np.asarray(
        development_idx,
        dtype=int,
    )

    y_dev = np.asarray(
        y,
        dtype=int,
    )[
        dev
    ]

    train_idx, remainder = train_test_split(
        dev,
        train_size=train_fraction_within_dev,
        random_state=seed,
        stratify=y_dev,
    )

    remaining_share = (
        1.0
        - train_fraction_within_dev
    )

    disc_share_within_remaining = (
        discovery_fraction_within_dev
        / remaining_share
    )

    disc_idx, sel_idx = train_test_split(
        remainder,
        train_size=disc_share_within_remaining,
        random_state=seed + 1,
        stratify=np.asarray(
            y,
            dtype=int,
        )[
            remainder
        ],
    )

    return FourWaySplit(
        train=np.asarray(
            train_idx,
            dtype=int,
        ),
        discovery=np.asarray(
            disc_idx,
            dtype=int,
        ),
        selection=np.asarray(
            sel_idx,
            dtype=int,
        ),
        test=np.asarray(
            test_idx,
            dtype=int,
        ),
    )


def _patient_group_stratified_four_way_split(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.50,
    discovery_fraction: float = 0.20,
    selection_fraction: float = 0.15,
) -> FourWaySplit:
    """
    Leakage-safe four-way split at PATIENT level.

    Each patient's stratification label is whether that patient has at least
    one <30-day readmission among eligible encounters.
    """
    y = np.asarray(
        y,
        dtype=int,
    )
    groups = np.asarray(
        groups
    )

    group_df = (
        pd.DataFrame(
            {
                "group": groups,
                "y": y,
            }
        )
        .groupby(
            "group",
            as_index=False,
        )
        .agg(
            group_y=(
                "y",
                "max",
            )
        )
    )

    unique_groups = group_df[
        "group"
    ].to_numpy()

    group_y = group_df[
        "group_y"
    ].to_numpy(
        dtype=int
    )

    train_groups, rest_groups = train_test_split(
        unique_groups,
        train_size=train_fraction,
        random_state=seed,
        stratify=group_y,
    )

    rest_mask = np.isin(
        unique_groups,
        rest_groups,
    )

    rest_y = group_y[
        rest_mask
    ]

    rest_fraction = (
        1.0
        - train_fraction
    )

    disc_rel = (
        discovery_fraction
        / rest_fraction
    )

    disc_groups, rest2_groups = train_test_split(
        rest_groups,
        train_size=disc_rel,
        random_state=seed + 1,
        stratify=rest_y,
    )

    rest2_mask = np.isin(
        unique_groups,
        rest2_groups,
    )

    rest2_y = group_y[
        rest2_mask
    ]

    rest2_fraction = (
        1.0
        - train_fraction
        - discovery_fraction
    )

    sel_rel = (
        selection_fraction
        / rest2_fraction
    )

    sel_groups, test_groups = train_test_split(
        rest2_groups,
        train_size=sel_rel,
        random_state=seed + 2,
        stratify=rest2_y,
    )

    def rows_for(
        selected_groups: np.ndarray,
    ) -> np.ndarray:
        return np.flatnonzero(
            np.isin(
                groups,
                selected_groups,
            )
        )

    return FourWaySplit(
        train=rows_for(
            train_groups
        ),
        discovery=rows_for(
            disc_groups
        ),
        selection=rows_for(
            sel_groups
        ),
        test=rows_for(
            test_groups
        ),
    )


def make_dataset_split(
    bundle: DatasetBundle,
    *,
    seed: int,
) -> FourWaySplit:
    if (
        bundle.official_train_mask
        is not None
        and bundle.official_test_mask
        is not None
    ):
        development_idx = np.flatnonzero(
            bundle.official_train_mask
        )

        test_idx = np.flatnonzero(
            bundle.official_test_mask
        )

        return _stratified_three_way_with_fixed_test(
            bundle.y,
            development_idx,
            test_idx,
            seed=seed,
            train_fraction_within_dev=0.70,
            discovery_fraction_within_dev=0.20,
        )

    if bundle.groups is None:
        raise ValueError(
            "Dataset has neither an official test set nor groups."
        )

    return _patient_group_stratified_four_way_split(
        bundle.y,
        bundle.groups,
        seed=seed,
    )


# =============================================================================
# Base-model preprocessing
# =============================================================================

def _onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float32,
        )


def make_base_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    transformers = []

    if numeric_columns:
        num_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                )
            ]
        )

        transformers.append(
            (
                "num",
                num_pipe,
                list(
                    numeric_columns
                ),
            )
        )

    if categorical_columns:
        cat_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                    ),
                ),
                (
                    "onehot",
                    _onehot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "cat",
                cat_pipe,
                list(
                    categorical_columns
                ),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


@dataclass
class FittedXGBBase:
    preprocessor: ColumnTransformer
    model: XGBClassifier
    selected_params: Dict[str, Any]
    fit_seconds: float

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        Z = self.preprocessor.transform(
            X
        )

        return clip_prob(
            self.model.predict_proba(
                Z
            )[
                :,
                1,
            ]
        )

    def logits(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        return logit(
            self.predict_proba(
                X
            )
        )


DEFAULT_XGB_GRID = [
    {
        "max_depth": 2,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "min_child_weight": 1,
    },
    {
        "max_depth": 3,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "min_child_weight": 1,
    },
    {
        "max_depth": 4,
        "n_estimators": 250,
        "learning_rate": 0.05,
        "min_child_weight": 1,
    },
    {
        "max_depth": 5,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "min_child_weight": 1,
    },
    {
        "max_depth": 3,
        "n_estimators": 500,
        "learning_rate": 0.03,
        "min_child_weight": 3,
    },
    {
        "max_depth": 4,
        "n_estimators": 400,
        "learning_rate": 0.03,
        "min_child_weight": 3,
    },
    {
        "max_depth": 3,
        "n_estimators": 200,
        "learning_rate": 0.10,
        "min_child_weight": 3,
    },
    {
        "max_depth": 4,
        "n_estimators": 150,
        "learning_rate": 0.10,
        "min_child_weight": 3,
    },
]


def _xgb_classifier(
    params: Mapping[str, Any],
    *,
    seed: int,
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(
            seed
        ),
        n_jobs=-1,
        tree_method="hist",
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        reg_alpha=0.0,
        **dict(
            params
        ),
    )


def select_xgb_params(
    bundle: DatasetBundle,
    split: FourWaySplit,
    *,
    seed: int,
    grid: Optional[
        Sequence[
            Mapping[
                str,
                Any,
            ]
        ]
    ] = None,
) -> Tuple[
    Dict[
        str,
        Any,
    ],
    pd.DataFrame,
]:
    """
    Dataset-specific tuning using TRAIN only.

    The outer DISCOVERY / SELECTION / TEST data are untouched.

    For grouped health data, the internal tune split is also patient-grouped.
    """
    grid = list(
        grid
        if grid is not None
        else DEFAULT_XGB_GRID
    )

    train_idx = np.asarray(
        split.train,
        dtype=int,
    )

    if bundle.groups is None:
        inner_train, inner_valid = train_test_split(
            train_idx,
            test_size=0.20,
            random_state=seed + 100,
            stratify=bundle.y[
                train_idx
            ],
        )
    else:
        # Patient-level 80/20 split inside TRAIN.
        groups_train = bundle.groups[
            train_idx
        ]

        local_split = (
            _patient_group_stratified_four_way_split(
                bundle.y[
                    train_idx
                ],
                groups_train,
                seed=seed + 100,
                train_fraction=0.80,
                discovery_fraction=0.10,
                selection_fraction=0.05,
            )
        )

        inner_train = train_idx[
            local_split.train
        ]

        inner_valid = np.concatenate(
            [
                train_idx[
                    local_split.discovery
                ],
                train_idx[
                    local_split.selection
                ],
                train_idx[
                    local_split.test
                ],
            ]
        )

    X_inner_train = bundle.X.iloc[
        inner_train
    ]

    X_inner_valid = bundle.X.iloc[
        inner_valid
    ]

    y_inner_train = bundle.y[
        inner_train
    ]

    y_inner_valid = bundle.y[
        inner_valid
    ]

    prep = make_base_preprocessor(
        bundle.numeric_columns,
        bundle.categorical_columns,
    )

    Z_train = prep.fit_transform(
        X_inner_train
    )

    Z_valid = prep.transform(
        X_inner_valid
    )

    records = []

    best_params = None
    best_logloss = np.inf

    for i, params in enumerate(
        grid
    ):
        model = _xgb_classifier(
            params,
            seed=seed + i,
        )

        t0 = time.perf_counter()

        model.fit(
            Z_train,
            y_inner_train,
        )

        fit_seconds = (
            time.perf_counter()
            - t0
        )

        p = clip_prob(
            model.predict_proba(
                Z_valid
            )[
                :,
                1,
            ]
        )

        ll = float(
            log_loss(
                y_inner_valid,
                p,
                labels=[
                    0,
                    1,
                ],
            )
        )

        records.append(
            {
                "candidate_index": i,
                "validation_logloss": ll,
                "fit_seconds": fit_seconds,
                "params_json": json.dumps(
                    dict(
                        params
                    ),
                    sort_keys=True,
                ),
            }
        )

        if ll < best_logloss:
            best_logloss = ll
            best_params = dict(
                params
            )

    if best_params is None:
        raise RuntimeError(
            "No XGBoost tuning candidate completed."
        )

    return (
        best_params,
        pd.DataFrame(
            records
        ).sort_values(
            "validation_logloss"
        ),
    )


def fit_xgb_base(
    bundle: DatasetBundle,
    split: FourWaySplit,
    params: Mapping[str, Any],
    *,
    seed: int,
) -> FittedXGBBase:
    prep = make_base_preprocessor(
        bundle.numeric_columns,
        bundle.categorical_columns,
    )

    X_train = bundle.X.iloc[
        split.train
    ]

    y_train = bundle.y[
        split.train
    ]

    t0 = time.perf_counter()

    Z_train = prep.fit_transform(
        X_train
    )

    model = _xgb_classifier(
        params,
        seed=seed,
    )

    model.fit(
        Z_train,
        y_train,
    )

    elapsed = (
        time.perf_counter()
        - t0
    )

    return FittedXGBBase(
        preprocessor=prep,
        model=model,
        selected_params=dict(
            params
        ),
        fit_seconds=float(
            elapsed
        ),
    )


# =============================================================================
# Compact ORIGINAL-feature scanner representation
# =============================================================================

class CompactScanPreprocessor:
    """
    TRAIN-fitted scanner transformation.

    Numeric:
        TRAIN median imputation; original scale.

    Categorical:
        one integer code per TRAIN category;
        missing is an explicit category;
        unseen values use one reserved unknown code.

    Integer category codes are NEVER treated as ordered because feature_types
    explicitly marks those columns as categorical.
    """

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
    ) -> None:
        self.numeric_columns = list(
            numeric_columns
        )
        self.categorical_columns = list(
            categorical_columns
        )
        self.columns = (
            self.numeric_columns
            + self.categorical_columns
        )

        self.numeric_medians_: Dict[
            str,
            float,
        ] = {}

        self.category_to_code_: Dict[
            str,
            Dict[
                str,
                int,
            ],
        ] = {}

        self.code_to_category_: Dict[
            str,
            Dict[
                int,
                str,
            ],
        ] = {}

    @staticmethod
    def _clean_category(
        value: Any,
    ) -> str:
        if pd.isna(
            value
        ):
            return "__MISSING__"

        s = str(
            value
        ).strip()

        if (
            s == ""
            or s == "?"
        ):
            return "__MISSING__"

        return s

    def fit(
        self,
        X: pd.DataFrame,
    ) -> "CompactScanPreprocessor":
        for col in self.numeric_columns:
            values = pd.to_numeric(
                X[
                    col
                ],
                errors="coerce",
            )

            median = float(
                values.median()
            )

            if not np.isfinite(
                median
            ):
                median = 0.0

            self.numeric_medians_[
                col
            ] = median

        for col in self.categorical_columns:
            cleaned = X[
                col
            ].map(
                self._clean_category
            )

            categories = sorted(
                cleaned.unique().tolist()
            )

            mapping = {
                value: i
                for i, value
                in enumerate(
                    categories
                )
            }

            reverse = {
                i: value
                for value, i
                in mapping.items()
            }

            self.category_to_code_[
                col
            ] = mapping

            self.code_to_category_[
                col
            ] = reverse

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        arrays = []

        for col in self.numeric_columns:
            values = pd.to_numeric(
                X[
                    col
                ],
                errors="coerce",
            ).fillna(
                self.numeric_medians_[
                    col
                ]
            )

            arrays.append(
                values.to_numpy(
                    dtype=float
                )
            )

        for col in self.categorical_columns:
            mapping = self.category_to_code_[
                col
            ]

            unknown_code = len(
                mapping
            )

            cleaned = X[
                col
            ].map(
                self._clean_category
            )

            codes = cleaned.map(
                lambda value: mapping.get(
                    value,
                    unknown_code,
                )
            )

            arrays.append(
                codes.to_numpy(
                    dtype=float
                )
            )

        if not arrays:
            return np.empty(
                (
                    len(
                        X
                    ),
                    0,
                ),
                dtype=float,
            )

        return np.column_stack(
            arrays
        ).astype(
            float
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        return self.fit(
            X
        ).transform(
            X
        )

    @property
    def feature_types(
        self,
    ) -> List[str]:
        return (
            [
                "numeric"
                for _ in self.numeric_columns
            ]
            + [
                "categorical"
                for _ in self.categorical_columns
            ]
        )

    @property
    def feature_names(
        self,
    ) -> List[str]:
        return list(
            self.columns
        )

    def categorical_label(
        self,
        feature_index: int,
        code: float,
    ) -> str:
        column = self.feature_names[
            feature_index
        ]

        if column not in self.code_to_category_:
            return str(
                code
            )

        reverse = self.code_to_category_[
            column
        ]

        integer = int(
            round(
                float(
                    code
                )
            )
        )

        if integer in reverse:
            return reverse[
                integer
            ]

        return "__UNKNOWN__"


# =============================================================================
# Final scanner
# =============================================================================

def make_realdata_scanner(
    feature_types: Sequence[str],
    *,
    discovery_n: int,
    selection_n: int,
    seed: int,
    repair_rounds: int = 4,
) -> StabilitySelectedScanRepairLearner:
    feature_types = list(
        feature_types
    )

    scanner = AdaptiveStructuredScannerV22(
        feature_types=feature_types,
        max_rule_features=5,
        beam_width=24,
        numeric_shortlist_per_direction=8,
        numeric_exact_pool=6,
        per_feature_exact_candidates=2,
        refine_top_n=3,
        refinement_passes=1,
        min_size=max(
            40,
            int(
                round(
                    0.01
                    * discovery_n
                )
            ),
        ),
        min_events=3,
        min_nonevents=3,
        candidate_pool_size=4,
        candidate_mask_jaccard_max=0.90,
    )

    all_numeric = all(
        t == "numeric"
        for t in feature_types
    )

    return StabilitySelectedScanRepairLearner(
        scanner,
        repair_rounds=repair_rounds,
        adaptive_candidates=4,
        # The protected legacy candidate assumes ordered numeric bins.
        # It is retained unchanged for all-numeric APS, but disabled for mixed
        # data so categorical IDs are never treated as ordered thresholds.
        include_legacy_candidate=all_numeric,
        legacy_beam_width=20,
        legacy_n_bins=5,
        coefficient_l2=1e-3,
        alpha=0.05,
        validation_split_seed=seed + 17001,
        selection_min_size=max(
            25,
            int(
                round(
                    0.01
                    * selection_n
                )
            ),
        ),
        selection_min_events=2,
        selection_min_nonevents=2,
        half_min_size=max(
            12,
            int(
                round(
                    0.005
                    * selection_n
                )
            ),
        ),
        half_min_events=1,
        half_min_nonevents=1,
    )


# =============================================================================
# Rule reporting
# =============================================================================

def describe_basis(
    basis: Any,
    scan_prep: CompactScanPreprocessor,
) -> str:
    if not hasattr(
        basis,
        "literals",
    ):
        return str(
            basis
        )

    pieces = []

    for j in sorted(
        basis.literals
    ):
        literal = basis.literals[
            j
        ]

        name = scan_prep.feature_names[
            j
        ]

        if isinstance(
            literal,
            NumericLiteral,
        ):
            op = (
                "<="
                if literal.direction
                == "le"
                else ">"
            )

            pieces.append(
                f"{name} {op} {literal.threshold:.5g}"
            )

        elif isinstance(
            literal,
            CategoricalLiteral,
        ):
            labels = [
                scan_prep.categorical_label(
                    j,
                    code,
                )
                for code
                in literal.allowed
            ]

            short = labels[
                :8
            ]

            suffix = (
                ""
                if len(
                    labels
                )
                <= 8
                else f" +{len(labels)-8} more"
            )

            pieces.append(
                f"{name} IN {{{', '.join(short)}}}{suffix}"
            )

        else:
            pieces.append(
                f"{name}: {literal}"
            )

    return " AND ".join(
        pieces
    )


def rule_diagnostics(
    learner: StabilitySelectedScanRepairLearner,
    scan_prep: CompactScanPreprocessor,
    X_test_scan: np.ndarray,
    y_test: np.ndarray,
    p_base: np.ndarray,
    p_repair: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for t, (
        basis,
        coef,
    ) in enumerate(
        zip(
            learner.bases_,
            learner.coefficients_,
        ),
        start=1,
    ):
        mask = (
            basis.transform(
                X_test_scan
            )
            .astype(
                bool
            )
        )

        n = int(
            np.sum(
                mask
            )
        )

        if n == 0:
            feature_indices = (
                sorted(
                    basis.literals
                )
                if hasattr(
                    basis,
                    "literals",
                )
                else []
            )

            feature_names = [
                scan_prep.feature_names[
                    j
                ]
                for j in feature_indices
            ]

            rows.append(
                {
                    "term": t,
                    "rule": describe_basis(
                        basis,
                        scan_prep,
                    ),
                    "rule_depth": int(
                        len(
                            feature_indices
                        )
                    ),
                    "features": ";".join(
                        feature_names
                    ),
                    "coefficient": float(
                        coef
                    ),
                    "test_n": 0,
                    "test_support": 0.0,
                    "test_event_rate": np.nan,
                    "base_mean_probability": np.nan,
                    "repair_mean_probability": np.nan,
                    "base_mean_residual": np.nan,
                    "repair_mean_residual": np.nan,
                }
            )

            continue

        event_rate = float(
            np.mean(
                y_test[
                    mask
                ]
            )
        )

        base_mean = float(
            np.mean(
                p_base[
                    mask
                ]
            )
        )

        repair_mean = float(
            np.mean(
                p_repair[
                    mask
                ]
            )
        )

        feature_indices = (
            sorted(
                basis.literals
            )
            if hasattr(
                basis,
                "literals",
            )
            else []
        )

        feature_names = [
            scan_prep.feature_names[
                j
            ]
            for j in feature_indices
        ]

        rows.append(
            {
                "term": t,
                "rule": describe_basis(
                    basis,
                    scan_prep,
                ),
                "rule_depth": int(
                    len(
                        feature_indices
                    )
                ),
                "features": ";".join(
                    feature_names
                ),
                "coefficient": float(
                    coef
                ),
                "test_n": n,
                "test_support": float(
                    np.mean(
                        mask
                    )
                ),
                "test_event_rate": event_rate,
                "base_mean_probability": base_mean,
                "repair_mean_probability": repair_mean,
                "base_mean_residual": (
                    event_rate
                    - base_mean
                ),
                "repair_mean_residual": (
                    event_rate
                    - repair_mean
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# One dataset / seed
# =============================================================================

def run_realdata_seed(
    bundle: DatasetBundle,
    selected_params: Mapping[str, Any],
    *,
    seed: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    split = make_dataset_split(
        bundle,
        seed=seed,
    )

    base = fit_xgb_base(
        bundle,
        split,
        selected_params,
        seed=seed,
    )

    Xd = bundle.X.iloc[
        split.discovery
    ]
    Xs = bundle.X.iloc[
        split.selection
    ]
    Xt = bundle.X.iloc[
        split.test
    ]

    yd = bundle.y[
        split.discovery
    ]
    ys = bundle.y[
        split.selection
    ]
    yt = bundle.y[
        split.test
    ]

    zd = base.logits(
        Xd
    )
    zs = base.logits(
        Xs
    )
    zt = base.logits(
        Xt
    )

    p_base = sigmoid(
        zt
    )

    # Fit compact scanner representation on TRAIN only.
    scan_prep = CompactScanPreprocessor(
        bundle.numeric_columns,
        bundle.categorical_columns,
    )

    scan_prep.fit(
        bundle.X.iloc[
            split.train
        ]
    )

    Xd_scan = scan_prep.transform(
        Xd
    )
    Xs_scan = scan_prep.transform(
        Xs
    )
    Xt_scan = scan_prep.transform(
        Xt
    )

    learner = make_realdata_scanner(
        scan_prep.feature_types,
        discovery_n=len(
            Xd
        ),
        selection_n=len(
            Xs
        ),
        seed=seed,
    )

    t0 = time.perf_counter()

    learner.fit(
        Xd_scan,
        yd,
        zd,
        Xs_scan,
        ys,
        zs,
    )

    repair_seconds = (
        time.perf_counter()
        - t0
    )

    z_repair = learner.repair_logits(
        Xt_scan,
        zt,
    )

    p_repair = sigmoid(
        z_repair
    )

    base_metrics = binary_metrics(
        yt,
        p_base,
    )

    repair_metrics = binary_metrics(
        yt,
        p_repair,
    )

    accepted = len(
        learner.bases_
    )

    rows = []

    for method, metrics in [
        (
            "Tuned XGBoost",
            base_metrics,
        ),
        (
            "Tuned XGBoost + Structured Scan",
            repair_metrics,
        ),
    ]:
        row = {
            "dataset": bundle.name,
            "seed": int(
                seed
            ),
            "method": method,
            "test_n": int(
                len(
                    yt
                )
            ),
            "test_events": int(
                np.sum(
                    yt
                )
            ),
            "test_event_rate": float(
                np.mean(
                    yt
                )
            ),
            "base_fit_seconds": float(
                base.fit_seconds
            ),
            "repair_seconds": (
                0.0
                if method
                == "Tuned XGBoost"
                else float(
                    repair_seconds
                )
            ),
            "accepted_repair_terms": (
                0
                if method
                == "Tuned XGBoost"
                else int(
                    accepted
                )
            ),
            "intervened": (
                0
                if method
                == "Tuned XGBoost"
                else int(
                    accepted
                    > 0
                )
            ),
            "selected_params_json": json.dumps(
                dict(
                    selected_params
                ),
                sort_keys=True,
            ),
            "mixed_categorical_scan": bool(
                len(
                    bundle.categorical_columns
                )
                > 0
            ),
            "protected_legacy_candidate_enabled": bool(
                len(
                    bundle.categorical_columns
                )
                == 0
            ),
            **metrics,
        }

        rows.append(
            row
        )

    rules = rule_diagnostics(
        learner,
        scan_prep,
        Xt_scan,
        yt,
        p_base,
        p_repair,
    )

    if not rules.empty:
        rules[
            "dataset"
        ] = bundle.name
        rules[
            "seed"
        ] = int(
            seed
        )

    return (
        pd.DataFrame(
            rows
        ),
        rules,
    )


# =============================================================================
# Matched summaries
# =============================================================================

def build_realdata_matched(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "dataset",
        "seed",
    ]

    base = (
        raw[
            raw[
                "method"
            ]
            == "Tuned XGBoost"
        ][
            keys
            + [
                "logloss",
                "brier",
                "auroc",
                "auprc",
                "ece_15",
            ]
        ]
        .rename(
            columns={
                "logloss":
                    "base_logloss",
                "brier":
                    "base_brier",
                "auroc":
                    "base_auroc",
                "auprc":
                    "base_auprc",
                "ece_15":
                    "base_ece_15",
            }
        )
    )

    repaired = (
        raw[
            raw[
                "method"
            ]
            == (
                "Tuned XGBoost "
                "+ Structured Scan"
            )
        ][
            keys
            + [
                "logloss",
                "brier",
                "auroc",
                "auprc",
                "ece_15",
                "repair_seconds",
                "accepted_repair_terms",
                "intervened",
            ]
        ]
        .rename(
            columns={
                "logloss":
                    "repair_logloss",
                "brier":
                    "repair_brier",
                "auroc":
                    "repair_auroc",
                "auprc":
                    "repair_auprc",
                "ece_15":
                    "repair_ece_15",
            }
        )
    )

    out = base.merge(
        repaired,
        on=keys,
        how="inner",
    )

    out[
        "logloss_advantage"
    ] = (
        out[
            "base_logloss"
        ]
        - out[
            "repair_logloss"
        ]
    )

    out[
        "brier_advantage"
    ] = (
        out[
            "base_brier"
        ]
        - out[
            "repair_brier"
        ]
    )

    out[
        "auroc_change"
    ] = (
        out[
            "repair_auroc"
        ]
        - out[
            "base_auroc"
        ]
    )

    out[
        "auprc_change"
    ] = (
        out[
            "repair_auprc"
        ]
        - out[
            "base_auprc"
        ]
    )

    out[
        "ece_reduction"
    ] = (
        out[
            "base_ece_15"
        ]
        - out[
            "repair_ece_15"
        ]
    )

    return out


def bootstrap_mean_ci(
    x: Sequence[float],
    *,
    n_boot: int = 4000,
    seed: int = 1,
) -> Tuple[
    float,
    float,
]:
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(
        x
    ) < 2:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    index = rng.integers(
        0,
        len(
            x
        ),
        size=(
            n_boot,
            len(
                x
            ),
        ),
    )

    means = x[
        index
    ].mean(
        axis=1
    )

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def summarise_realdata(
    matched: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dataset, g in matched.groupby(
        "dataset"
    ):
        dll = g[
            "logloss_advantage"
        ].to_numpy(
            dtype=float
        )

        ll_lo, ll_hi = bootstrap_mean_ci(
            dll,
            seed=abs(
                hash(
                    (
                        dataset,
                        "ll",
                    )
                )
            )
            % 2_000_000_000,
        )

        rows.append(
            {
                "dataset": dataset,
                "runs": int(
                    len(
                        g
                    )
                ),
                "mean_base_logloss": float(
                    g[
                        "base_logloss"
                    ].mean()
                ),
                "mean_repair_logloss": float(
                    g[
                        "repair_logloss"
                    ].mean()
                ),
                "mean_logloss_advantage": float(
                    np.mean(
                        dll
                    )
                ),
                "logloss_ci_low": ll_lo,
                "logloss_ci_high": ll_hi,
                "mean_brier_advantage": float(
                    g[
                        "brier_advantage"
                    ].mean()
                ),
                "mean_auroc_change": float(
                    g[
                        "auroc_change"
                    ].mean()
                ),
                "mean_auprc_change": float(
                    g[
                        "auprc_change"
                    ].mean()
                ),
                "mean_ece_reduction": float(
                    g[
                        "ece_reduction"
                    ].mean()
                ),
                "intervention_rate": float(
                    g[
                        "intervened"
                    ].mean()
                ),
                "mean_repair_terms": float(
                    g[
                        "accepted_repair_terms"
                    ].mean()
                ),
                "mean_repair_seconds": float(
                    g[
                        "repair_seconds"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Dataset selection table
# =============================================================================

def dataset_methodological_roles(
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "Adult Income",
                "methodological_question": (
                    "Can structured residual repair add value in a standard "
                    "mixed-type benchmark where the base learner is already "
                    "strong and subgroup rules remain semantically interpretable?"
                ),
                "low_level_properties": (
                    "Mixed numeric/categorical predictors; standard held-out "
                    "test; moderate dimensionality; interpretable covariates; "
                    "well-established tree benchmarks."
                ),
            },
            {
                "dataset": "APS Failure at Scania Trucks",
                "methodological_question": (
                    "Can a model with very strong discrimination still retain "
                    "stable probability residuals in a rare-event, high-dimensional "
                    "operational setting?"
                ),
                "low_level_properties": (
                    "Rare positives; 170 anonymized numeric predictors; missing "
                    "values; official train/test split; operational cost-sensitive "
                    "context; strong published tree discrimination."
                ),
            },
            {
                "dataset": "Diabetes 130-US Hospitals",
                "methodological_question": (
                    "Does structured residual repair remain useful for clinical "
                    "risk probabilities when repeated patients, missingness, and "
                    "mixed/high-cardinality predictors require leakage-safe design?"
                ),
                "low_level_properties": (
                    "Early-readmission target; repeated patient encounters; "
                    "patient-level grouped splitting; mixed clinical variables; "
                    "missingness; coarse clinically interpretable ICD-9 chapters; "
                    "probability quality is substantively meaningful."
                ),
            },
        ]
    )
