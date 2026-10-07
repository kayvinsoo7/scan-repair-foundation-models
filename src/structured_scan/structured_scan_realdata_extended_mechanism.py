
"""
Extended real-data benchmarks + mechanistic audit for Structured Scan Repair
============================================================================

Adds three precommitted real-data benchmarks to the existing three-dataset
pipeline:

4. Bank Marketing (UCI bank-additional-full)
   - prospective/temporal stress test
   - duration is removed because it is unavailable before the call outcome
   - chronological 50/20/15/15 train/discovery/selection/test split

5. Default of Credit Card Clients
   - probability-of-default task
   - longitudinal repayment / bill / payment-history blocks
   - fixed stratified 15% TEST; remaining development data split repeatedly

6. NIJ Recidivism Forecasting Challenge
   - justice / legal-risk benchmark
   - one-year recidivism target
   - official challenge train/test membership reconstructed from IDs
   - only Year-1 baseline variables available in Test Dataset 1 are used
   - race and gender are retained only as audit metadata, not model/scan inputs

The file also adds a mechanistic audit for accepted Structured Scan corrections.

For each accepted TEST-set correction term it asks:

WHERE?
    Did the independently selected subgroup retain a coherent TEST residual?

HOW?
    What log-odds coefficient was applied, and in what direction?

WHY DID IT HELP?
    1. Did absolute subgroup residual shrink on TEST?
    2. Did local log-loss improve?
    3. Does removing the term from the full repaired model worsen TEST log-loss?
    4. Does the true discovered subgroup outperform score-matched placebo
       subgroups receiving the same coefficient?

The placebo control preserves the number of selected TEST observations within
base-model risk deciles. Thus it asks whether the gain is due to selecting the
right observations rather than merely shifting a similarly sized group with a
similar baseline-risk distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import importlib.util
import json
import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import structured_scan_realdata_benchmarks as core


# =============================================================================
# Public sources
# =============================================================================

BANK_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/222/"
    "bank+marketing.zip"
)

CREDIT_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)

NIJ_FULL_URL = (
    "https://nij.ojp.gov/sites/nij/files/media/document/"
    "nij-challenge2021_full_dataset.csv"
)

NIJ_TRAIN_URL = (
    "https://nij.ojp.gov/sites/nij/files/media/document/"
    "nij-challenge2021_training_dataset.csv"
)

NIJ_TEST1_URL = (
    "https://nij.ojp.gov/sites/nij/files/media/document/"
    "nij-challenge2021_test_dataset_1.csv"
)


# =============================================================================
# New dataset loaders
# =============================================================================

def load_bank_marketing(
    data_dir: Path,
    *,
    allow_download: bool = True,
) -> core.DatasetBundle:
    """
    UCI bank-additional-full, preserving chronological row order.

    'duration' is excluded because UCI explicitly notes that it is not known
    before a call is completed and should be removed for realistic prediction.
    """
    root = Path(data_dir) / "bank_marketing"
    root.mkdir(parents=True, exist_ok=True)

    candidates = list(
        root.rglob("bank-additional-full.csv")
    )

    if not candidates and allow_download:
        core._download_and_extract_zip(
            BANK_ZIP_URL,
            root,
            "bank_marketing.zip",
        )
        candidates = list(
            root.rglob("bank-additional-full.csv")
        )

    if not candidates:
        raise FileNotFoundError(
            "bank-additional-full.csv not found under "
            f"{root}. Download the UCI Bank Marketing archive."
        )

    df = pd.read_csv(
        candidates[0],
        sep=";",
        low_memory=False,
    )

    if "y" not in df.columns:
        raise ValueError("Expected target column 'y' in Bank Marketing.")

    y = (
        df["y"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(int)
        .to_numpy()
    )

    feature_cols = [
        c
        for c in df.columns
        if c not in {
            "y",
            "duration",
        }
    ]

    X = df[feature_cols].copy()

    categorical = [
        c
        for c in feature_cols
        if (
            X[c].dtype == object
            or str(X[c].dtype).startswith("string")
        )
    ]

    numeric = [
        c
        for c in feature_cols
        if c not in categorical
    ]

    for c in numeric:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce",
        )

    return core.DatasetBundle(
        name="Bank Marketing",
        methodological_role=(
            "Prospective temporal prediction with mixed categorical/contextual "
            "variables and moderate imbalance; tests whether interaction-rich "
            "residual structure persists under temporal generalization."
        ),
        X=X,
        y=y,
        numeric_columns=numeric,
        categorical_columns=categorical,
        official_train_mask=None,
        official_test_mask=None,
        groups=None,
        source_metadata={
            "source": "UCI Bank Marketing: bank-additional-full",
            "split_strategy": "chronological_50_20_15_15",
            "duration_removed": True,
            "n": int(len(df)),
            "event_rate": float(np.mean(y)),
        },
    )


def _read_credit_xls(
    xls_path: Path,
) -> pd.DataFrame:
    """
    Read the UCI legacy .xls file.

    xlrd is required by pandas for legacy .xls. A clear error is raised rather
    than silently changing the source.
    """
    if importlib.util.find_spec("xlrd") is None:
        raise ImportError(
            "Reading the original UCI credit-default .xls requires `xlrd`. "
            "Install once with: pip install xlrd"
        )

    # The original file commonly has a title row followed by the true header.
    df = pd.read_excel(
        xls_path,
        header=1,
        engine="xlrd",
    )

    target_candidates = [
        c
        for c in df.columns
        if (
            str(c).strip().lower()
            in {
                "default payment next month",
                "default.payment.next.month",
                "default",
                "y",
            }
        )
    ]

    if target_candidates:
        return df

    # Fallback for differently packaged mirrors.
    return pd.read_excel(
        xls_path,
        header=0,
        engine="xlrd",
    )


def load_credit_default(
    data_dir: Path,
    *,
    allow_download: bool = True,
    fixed_test_seed: int = 52001,
) -> core.DatasetBundle:
    """
    UCI Default of Credit Card Clients.

    A fixed 15% stratified TEST set is created once. Across experimental seeds,
    only the remaining development pool is repartitioned.
    """
    root = Path(data_dir) / "credit_default"
    root.mkdir(parents=True, exist_ok=True)

    candidates = list(
        root.rglob("default of credit card clients.xls")
    )

    if not candidates and allow_download:
        core._download_and_extract_zip(
            CREDIT_ZIP_URL,
            root,
            "credit_default.zip",
        )
        candidates = list(
            root.rglob("default of credit card clients.xls")
        )

    if not candidates:
        raise FileNotFoundError(
            "The original UCI credit-default .xls file was not found under "
            f"{root}."
        )

    df = _read_credit_xls(
        candidates[0]
    )

    # Normalize column labels.
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    target_candidates = [
        c
        for c in df.columns
        if (
            c.lower()
            in {
                "default payment next month",
                "default.payment.next.month",
                "default",
                "y",
            }
        )
    ]

    if not target_candidates:
        raise ValueError(
            "Could not identify the credit-default target column. "
            f"Columns were: {list(df.columns)}"
        )

    target = target_candidates[0]

    y = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    valid = y.notna()

    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).to_numpy()

    id_candidates = [
        c
        for c in df.columns
        if c.strip().lower() == "id"
    ]

    drop = {
        target,
        *id_candidates,
    }

    feature_cols = [
        c
        for c in df.columns
        if c not in drop
    ]

    X = df[feature_cols].copy()

    # These are nominal in the UCI documentation.
    categorical = [
        c
        for c in [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
        ]
        if c in X.columns
    ]

    numeric = [
        c
        for c in feature_cols
        if c not in categorical
    ]

    for c in numeric:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce",
        )

    for c in categorical:
        X[c] = (
            X[c]
            .astype("object")
        )

    indices = np.arange(len(X))

    dev_idx, test_idx = train_test_split(
        indices,
        test_size=0.15,
        random_state=fixed_test_seed,
        stratify=y,
    )

    train_mask = np.zeros(
        len(X),
        dtype=bool,
    )
    test_mask = np.zeros(
        len(X),
        dtype=bool,
    )

    train_mask[dev_idx] = True
    test_mask[test_idx] = True

    return core.DatasetBundle(
        name="Default of Credit Card Clients",
        methodological_role=(
            "Probability-of-default prediction with longitudinal repayment, "
            "billing and payment-history blocks; tests structured residuals "
            "under correlated temporal financial features."
        ),
        X=X,
        y=y,
        numeric_columns=numeric,
        categorical_columns=categorical,
        official_train_mask=train_mask,
        official_test_mask=test_mask,
        groups=None,
        source_metadata={
            "source": "UCI Default of Credit Card Clients",
            "split_strategy": "fixed_stratified_test_15pct",
            "fixed_test_seed": int(fixed_test_seed),
            "n": int(len(X)),
            "event_rate": float(np.mean(y)),
        },
    )


def _coerce_binary(
    series: pd.Series,
) -> np.ndarray:
    """
    Robust conversion of NIJ outcome columns to {0,1}.
    """
    if pd.api.types.is_numeric_dtype(
        series
    ):
        x = pd.to_numeric(
            series,
            errors="coerce",
        )
        return (
            x.fillna(0)
            .astype(int)
            .clip(0, 1)
            .to_numpy()
        )

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    positives = {
        "1",
        "yes",
        "y",
        "true",
        "recidivated",
    }

    return cleaned.isin(
        positives
    ).astype(int).to_numpy()


def _infer_nij_feature_types(
    X: pd.DataFrame,
) -> Tuple[
    List[str],
    List[str],
]:
    """
    Conservative typing for the NIJ baseline variables.

    Continuous/count-like fields remain numeric. Low-cardinality encoded
    variables are categorical unless they are the explicit supervision risk
    score, which is ordinal 1--10 and treated numerically.
    """
    force_numeric = {
        "Supervision_Risk_Score_First",
    }

    force_categorical = {
        "Residence_PUMA",
        "Age_at_Release",
        "Supervision_Level_First",
        "Education_Level",
        "Dependents",
        "Prison_Offense",
        "Prison_Years",
    }

    numeric = []
    categorical = []

    for c in X.columns:
        if c in force_numeric:
            numeric.append(c)
            continue

        if c in force_categorical:
            categorical.append(c)
            continue

        converted = pd.to_numeric(
            X[c],
            errors="coerce",
        )

        nonmissing = X[c].notna().sum()
        numeric_fraction = (
            float(converted.notna().sum())
            / max(1, int(nonmissing))
        )

        if numeric_fraction >= 0.95:
            unique = converted.nunique(
                dropna=True
            )

            if unique <= 5:
                categorical.append(c)
            else:
                numeric.append(c)
        else:
            categorical.append(c)

    return numeric, categorical


def load_nij_recidivism(
    data_dir: Path,
    *,
    allow_download: bool = True,
) -> core.DatasetBundle:
    """
    NIJ Recidivism Forecasting Challenge, one-year prediction.

    The official FULL dataset supplies final outcome labels; training/test-1 IDs
    reconstruct the official challenge split.

    Only variables present in the original Year-1 Test Dataset 1 are eligible
    predictors. This excludes later supervision-activity variables that accrue
    after release.

    Gender and Race are intentionally excluded from model and scan inputs in
    this benchmark. They may be retained separately for fairness auditing, but
    the correction algorithm is not allowed to use them as repair dimensions.
    """
    root = Path(data_dir) / "nij_recidivism"
    root.mkdir(parents=True, exist_ok=True)

    full_path = root / "nij_challenge2021_full_dataset.csv"
    train_path = root / "nij_challenge2021_training_dataset.csv"
    test1_path = root / "nij_challenge2021_test_dataset_1.csv"

    if allow_download:
        core._download(
            NIJ_FULL_URL,
            full_path,
        )
        core._download(
            NIJ_TRAIN_URL,
            train_path,
        )
        core._download(
            NIJ_TEST1_URL,
            test1_path,
        )

    for p in [
        full_path,
        train_path,
        test1_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing NIJ source file: {p}"
            )

    full = pd.read_csv(
        full_path,
        low_memory=False,
    )
    train = pd.read_csv(
        train_path,
        low_memory=False,
    )
    test1 = pd.read_csv(
        test1_path,
        low_memory=False,
    )

    full.columns = [
        str(c).strip()
        for c in full.columns
    ]
    train.columns = [
        str(c).strip()
        for c in train.columns
    ]
    test1.columns = [
        str(c).strip()
        for c in test1.columns
    ]

    id_col = next(
        (
            c
            for c in full.columns
            if c.lower() == "id"
        ),
        None,
    )

    if id_col is None:
        raise ValueError(
            "Could not find NIJ ID column."
        )

    target = next(
        (
            c
            for c in full.columns
            if c.lower()
            == "recidivism_arrest_year1"
        ),
        None,
    )

    if target is None:
        raise ValueError(
            "Could not find Recidivism_Arrest_Year1 in NIJ full dataset."
        )

    # Year-1 available baseline features are exactly those appearing in Test 1.
    test1_features = [
        c
        for c in test1.columns
        if c != id_col
    ]

    protected_audit_only = [
        c
        for c in [
            "Gender",
            "Race",
        ]
        if c in test1_features
    ]

    feature_cols = [
        c
        for c in test1_features
        if (
            c in full.columns
            and c not in protected_audit_only
        )
    ]

    target_like = {
        "Recidivism_Within_3years",
        "Recidivism_Arrest_Year1",
        "Recidivism_Arrest_Year2",
        "Recidivism_Arrest_Year3",
    }

    feature_cols = [
        c
        for c in feature_cols
        if c not in target_like
    ]

    y = _coerce_binary(
        full[target]
    )

    X = full[feature_cols].copy()

    numeric, categorical = (
        _infer_nij_feature_types(
            X
        )
    )

    for c in numeric:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce",
        )

    for c in categorical:
        X[c] = X[c].astype(
            "object"
        )

    train_ids = set(
        train[
            next(
                c
                for c in train.columns
                if c.lower() == "id"
            )
        ].astype(str)
    )

    test_ids = set(
        test1[
            next(
                c
                for c in test1.columns
                if c.lower() == "id"
            )
        ].astype(str)
    )

    full_ids = full[id_col].astype(str)

    official_train_mask = (
        full_ids.isin(
            train_ids
        ).to_numpy()
    )

    official_test_mask = (
        full_ids.isin(
            test_ids
        ).to_numpy()
    )

    if np.any(
        official_train_mask
        & official_test_mask
    ):
        raise ValueError(
            "NIJ train/test ID overlap detected."
        )

    covered = (
        official_train_mask
        | official_test_mask
    )

    if not np.all(
        covered
    ):
        # Keep only the official train/test-1 population.
        keep = np.flatnonzero(
            covered
        )

        X = X.iloc[
            keep
        ].reset_index(
            drop=True
        )
        y = y[
            keep
        ]
        official_train_mask = official_train_mask[
            keep
        ]
        official_test_mask = official_test_mask[
            keep
        ]

        full = full.iloc[
            keep
        ].reset_index(
            drop=True
        )

    audit_metadata = pd.DataFrame(
        {
            c: full[c].to_numpy()
            for c in protected_audit_only
        }
    )

    return core.DatasetBundle(
        name="NIJ Recidivism Forecasting Challenge",
        methodological_role=(
            "High-stakes justice risk forecasting with an official challenge "
            "split and explicit probability predictions; tests whether sparse "
            "residual repair can improve one-year recidivism probabilities "
            "without using race or gender as predictive repair dimensions."
        ),
        X=X,
        y=y,
        numeric_columns=numeric,
        categorical_columns=categorical,
        official_train_mask=official_train_mask,
        official_test_mask=official_test_mask,
        groups=None,
        source_metadata={
            "source": "National Institute of Justice Recidivism Forecasting Challenge",
            "target": target,
            "split_strategy": "official_challenge_train_test1",
            "protected_audit_only": protected_audit_only,
            "audit_metadata": audit_metadata,
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
            "event_rate": float(
                np.mean(
                    y
                )
            ),
        },
    )


# =============================================================================
# Split protocols
# =============================================================================

def chronological_four_way_split(
    n: int,
) -> core.FourWaySplit:
    """
    Fixed chronological 50/20/15/15 split.
    """
    n_train = int(
        round(
            0.50 * n
        )
    )
    n_disc = int(
        round(
            0.20 * n
        )
    )
    n_sel = int(
        round(
            0.15 * n
        )
    )

    i1 = n_train
    i2 = i1 + n_disc
    i3 = i2 + n_sel

    return core.FourWaySplit(
        train=np.arange(
            0,
            i1,
            dtype=int,
        ),
        discovery=np.arange(
            i1,
            i2,
            dtype=int,
        ),
        selection=np.arange(
            i2,
            i3,
            dtype=int,
        ),
        test=np.arange(
            i3,
            n,
            dtype=int,
        ),
    )


def _split_fixed_15pct_test(
    bundle: core.DatasetBundle,
    *,
    seed: int,
) -> core.FourWaySplit:
    """
    For the credit dataset: fixed 15% TEST, then split the 85% development pool
    so total proportions are approximately 50/20/15/15.
    """
    dev = np.flatnonzero(
        bundle.official_train_mask
    )
    test = np.flatnonzero(
        bundle.official_test_mask
    )

    y_dev = bundle.y[
        dev
    ]

    train_rel = (
        0.50
        / 0.85
    )

    train_idx, rest = train_test_split(
        dev,
        train_size=train_rel,
        random_state=seed,
        stratify=y_dev,
    )

    # 20 / (20 + 15)
    disc_rel = (
        20.0
        / 35.0
    )

    disc_idx, sel_idx = train_test_split(
        rest,
        train_size=disc_rel,
        random_state=seed + 1,
        stratify=bundle.y[
            rest
        ],
    )

    return core.FourWaySplit(
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
            test,
            dtype=int,
        ),
    )


def make_extended_split(
    bundle: core.DatasetBundle,
    *,
    seed: int,
) -> core.FourWaySplit:
    strategy = (
        bundle.source_metadata or {}
    ).get(
        "split_strategy",
        "",
    )

    if strategy == "chronological_50_20_15_15":
        return chronological_four_way_split(
            len(
                bundle.y
            )
        )

    if strategy == "fixed_stratified_test_15pct":
        return _split_fixed_15pct_test(
            bundle,
            seed=seed,
        )

    return core.make_dataset_split(
        bundle,
        seed=seed,
    )


# =============================================================================
# XGBoost tuning
# =============================================================================

def _select_xgb_temporal(
    bundle: core.DatasetBundle,
    split: core.FourWaySplit,
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
    Chronological tuning inside the Bank TRAIN segment:
    first 80% inner-train, last 20% inner-validation.
    """
    grid = list(
        grid
        if grid is not None
        else core.DEFAULT_XGB_GRID
    )

    train_idx = np.sort(
        np.asarray(
            split.train,
            dtype=int,
        )
    )

    cut = int(
        round(
            0.80
            * len(
                train_idx
            )
        )
    )

    inner_train = train_idx[
        :cut
    ]
    inner_valid = train_idx[
        cut:
    ]

    prep = core.make_base_preprocessor(
        bundle.numeric_columns,
        bundle.categorical_columns,
    )

    Z_train = prep.fit_transform(
        bundle.X.iloc[
            inner_train
        ]
    )

    Z_valid = prep.transform(
        bundle.X.iloc[
            inner_valid
        ]
    )

    y_train = bundle.y[
        inner_train
    ]
    y_valid = bundle.y[
        inner_valid
    ]

    rows = []
    best = None
    best_ll = np.inf

    for i, params in enumerate(
        grid
    ):
        model = core._xgb_classifier(
            params,
            seed=seed + i,
        )

        t0 = time.perf_counter()

        model.fit(
            Z_train,
            y_train,
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        p = core.clip_prob(
            model.predict_proba(
                Z_valid
            )[
                :,
                1,
            ]
        )

        ll = float(
            log_loss(
                y_valid,
                p,
                labels=[
                    0,
                    1,
                ],
            )
        )

        rows.append(
            {
                "candidate_index": i,
                "validation_logloss": ll,
                "fit_seconds": elapsed,
                "params_json": json.dumps(
                    dict(
                        params
                    ),
                    sort_keys=True,
                ),
            }
        )

        if ll < best_ll:
            best_ll = ll
            best = dict(
                params
            )

    if best is None:
        raise RuntimeError(
            "No temporal XGBoost tuning candidate completed."
        )

    return (
        best,
        pd.DataFrame(
            rows
        ).sort_values(
            "validation_logloss"
        ),
    )


def select_xgb_params_extended(
    bundle: core.DatasetBundle,
    split: core.FourWaySplit,
    *,
    seed: int,
) -> Tuple[
    Dict[
        str,
        Any,
    ],
    pd.DataFrame,
]:
    strategy = (
        bundle.source_metadata or {}
    ).get(
        "split_strategy",
        "",
    )

    if strategy == "chronological_50_20_15_15":
        return _select_xgb_temporal(
            bundle,
            split,
            seed=seed,
        )

    return core.select_xgb_params(
        bundle,
        split,
        seed=seed,
    )


# =============================================================================
# Mechanistic audit
# =============================================================================

def _risk_deciles(
    p_base: np.ndarray,
) -> np.ndarray:
    """
    Quantile groups with robust duplicate handling.
    """
    p_base = np.asarray(
        p_base,
        dtype=float,
    )

    try:
        bins = pd.qcut(
            p_base,
            q=10,
            labels=False,
            duplicates="drop",
        )

        return np.asarray(
            bins,
            dtype=int,
        )
    except Exception:
        order = np.argsort(
            p_base
        )

        groups = np.empty(
            len(
                p_base
            ),
            dtype=int,
        )

        chunks = np.array_split(
            order,
            10,
        )

        for j, idx in enumerate(
            chunks
        ):
            groups[
                idx
            ] = j

        return groups


def score_matched_placebo_masks(
    actual_mask: np.ndarray,
    p_base: np.ndarray,
    *,
    n_placebo: int,
    seed: int,
) -> Iterable[
    np.ndarray
]:
    """
    Generate same-size placebo groups matched exactly on counts within base-risk
    quantile bins.
    """
    actual_mask = np.asarray(
        actual_mask,
        dtype=bool,
    )

    strata = _risk_deciles(
        p_base
    )

    rng = np.random.default_rng(
        seed
    )

    unique_strata = np.unique(
        strata
    )

    requested = {
        s: int(
            np.sum(
                actual_mask
                & (
                    strata
                    == s
                )
            )
        )
        for s in unique_strata
    }

    candidates = {
        s: np.flatnonzero(
            strata
            == s
        )
        for s in unique_strata
    }

    for _ in range(
        n_placebo
    ):
        placebo = np.zeros(
            len(
                actual_mask
            ),
            dtype=bool,
        )

        for s in unique_strata:
            k = requested[
                s
            ]

            if k == 0:
                continue

            pool = candidates[
                s
            ]

            if k > len(
                pool
            ):
                raise RuntimeError(
                    "Requested placebo stratum count exceeds available pool."
                )

            chosen = rng.choice(
                pool,
                size=k,
                replace=False,
            )

            placebo[
                chosen
            ] = True

        yield placebo


def mechanistic_term_audit(
    learner: core.StabilitySelectedScanRepairLearner,
    scan_prep: core.CompactScanPreprocessor,
    X_test_scan: np.ndarray,
    y_test: np.ndarray,
    z_base: np.ndarray,
    *,
    dataset: str,
    seed: int,
    fixed_test: bool,
    n_placebo: int = 500,
) -> pd.DataFrame:
    """
    TEST-only mechanistic evaluation of already-selected correction terms.

    TEST labels are never used to choose rules or coefficients.
    """
    y_test = np.asarray(
        y_test,
        dtype=int,
    )
    z_base = np.asarray(
        z_base,
        dtype=float,
    )

    z_full = learner.repair_logits(
        X_test_scan,
        z_base,
    )

    p_base = core.sigmoid(
        z_base
    )
    p_full = core.sigmoid(
        z_full
    )

    full_ll = float(
        log_loss(
            y_test,
            core.clip_prob(
                p_full
            ),
            labels=[
                0,
                1,
            ],
        )
    )

    strata = _risk_deciles(
        p_base
    )

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
            continue

        z_without = (
            z_full
            - float(
                coef
            )
            * mask.astype(
                float
            )
        )

        p_without = core.sigmoid(
            z_without
        )

        ll_without = float(
            log_loss(
                y_test,
                core.clip_prob(
                    p_without
                ),
                labels=[
                    0,
                    1,
                ],
            )
        )

        ablation_gain = (
            ll_without
            - full_ll
        )

        y_s = y_test[
            mask
        ]

        p_base_s = p_base[
            mask
        ]
        p_without_s = p_without[
            mask
        ]
        p_full_s = p_full[
            mask
        ]

        event_rate = float(
            np.mean(
                y_s
            )
        )

        base_mean = float(
            np.mean(
                p_base_s
            )
        )

        context_mean = float(
            np.mean(
                p_without_s
            )
        )

        repair_mean = float(
            np.mean(
                p_full_s
            )
        )

        base_residual = (
            event_rate
            - base_mean
        )

        context_residual = (
            event_rate
            - context_mean
        )

        repair_residual = (
            event_rate
            - repair_mean
        )

        local_base_ll = float(
            log_loss(
                y_s,
                core.clip_prob(
                    p_base_s
                ),
                labels=[
                    0,
                    1,
                ],
            )
        )

        local_context_ll = float(
            log_loss(
                y_s,
                core.clip_prob(
                    p_without_s
                ),
                labels=[
                    0,
                    1,
                ],
            )
        )

        local_repair_ll = float(
            log_loss(
                y_s,
                core.clip_prob(
                    p_full_s
                ),
                labels=[
                    0,
                    1,
                ],
            )
        )

        placebo_gains = []

        for b, placebo in enumerate(
            score_matched_placebo_masks(
                mask,
                p_base,
                n_placebo=n_placebo,
                seed=(
                    int(
                        seed
                    )
                    * 10_000
                    + t * 100
                    + 17
                ),
            )
        ):
            z_placebo = (
                z_without
                + float(
                    coef
                )
                * placebo.astype(
                    float
                )
            )

            p_placebo = core.sigmoid(
                z_placebo
            )

            ll_placebo = float(
                log_loss(
                    y_test,
                    core.clip_prob(
                        p_placebo
                    ),
                    labels=[
                        0,
                        1,
                    ],
                )
            )

            placebo_gains.append(
                ll_without
                - ll_placebo
            )

        placebo_gains = np.asarray(
            placebo_gains,
            dtype=float,
        )

        placebo_p = float(
            (
                1
                + np.sum(
                    placebo_gains
                    >= ablation_gain
                )
            )
            / (
                len(
                    placebo_gains
                )
                + 1
            )
        )

        placebo_percentile = float(
            np.mean(
                ablation_gain
                > placebo_gains
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

        features = [
            scan_prep.feature_names[
                j
            ]
            for j in feature_indices
        ]

        direction_agreement = int(
            (
                np.sign(
                    float(
                        coef
                    )
                )
                == np.sign(
                    context_residual
                )
            )
            and (
                np.sign(
                    context_residual
                )
                != 0
            )
        )

        packed_hex = ""

        if fixed_test:
            packed_hex = (
                np.packbits(
                    mask.astype(
                        np.uint8
                    )
                )
                .tobytes()
                .hex()
            )

        rows.append(
            {
                "dataset": dataset,
                "seed": int(
                    seed
                ),
                "term": int(
                    t
                ),
                "rule": core.describe_basis(
                    basis,
                    scan_prep,
                ),
                "features": ";".join(
                    features
                ),
                "rule_depth": int(
                    len(
                        features
                    )
                ),
                "coefficient": float(
                    coef
                ),
                "odds_multiplier": float(
                    np.exp(
                        float(
                            coef
                        )
                    )
                ),
                "test_n": n,
                "test_support": float(
                    np.mean(
                        mask
                    )
                ),
                "test_event_rate": event_rate,
                "base_mean_probability": base_mean,
                "context_mean_probability_before_term": context_mean,
                "repair_mean_probability": repair_mean,
                "base_residual": base_residual,
                "context_residual_before_term": context_residual,
                "repair_residual": repair_residual,
                "base_absolute_residual_reduction": (
                    abs(
                        base_residual
                    )
                    - abs(
                        repair_residual
                    )
                ),
                "context_absolute_residual_reduction": (
                    abs(
                        context_residual
                    )
                    - abs(
                        repair_residual
                    )
                ),
                "direction_agreement": direction_agreement,
                "local_logloss_advantage_base_to_full": (
                    local_base_ll
                    - local_repair_ll
                ),
                "local_logloss_advantage_term_context": (
                    local_context_ll
                    - local_repair_ll
                ),
                "global_term_ablation_gain": ablation_gain,
                "placebo_gain_mean": float(
                    np.mean(
                        placebo_gains
                    )
                ),
                "placebo_gain_median": float(
                    np.median(
                        placebo_gains
                    )
                ),
                "placebo_gain_q95": float(
                    np.quantile(
                        placebo_gains,
                        0.95,
                    )
                ),
                "placebo_percentile": placebo_percentile,
                "placebo_empirical_p": placebo_p,
                "placebo_replicates": int(
                    len(
                        placebo_gains
                    )
                ),
                "fixed_test_mask_available": bool(
                    fixed_test
                ),
                "mask_packed_hex": packed_hex,
                "mask_length": int(
                    len(
                        mask
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# One run with mechanism output
# =============================================================================

def run_seed_with_mechanism(
    bundle: core.DatasetBundle,
    selected_params: Mapping[str, Any],
    *,
    seed: int,
    n_placebo: int = 500,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    split = make_extended_split(
        bundle,
        seed=seed,
    )

    base = core.fit_xgb_base(
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

    p_base = core.sigmoid(
        zt
    )

    scan_prep = core.CompactScanPreprocessor(
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

    learner = core.make_realdata_scanner(
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

    p_repair = core.sigmoid(
        z_repair
    )

    base_metrics = core.binary_metrics(
        yt,
        p_base,
    )

    repair_metrics = core.binary_metrics(
        yt,
        p_repair,
    )

    accepted = len(
        learner.bases_
    )

    result_rows = []

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
        result_rows.append(
            {
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
                **metrics,
            }
        )

    rules = core.rule_diagnostics(
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

    fixed_test = (
        (
            bundle.official_test_mask
            is not None
        )
        or (
            (
                bundle.source_metadata
                or {}
            ).get(
                "split_strategy",
                ""
            )
            == "chronological_50_20_15_15"
        )
    )

    mechanism = mechanistic_term_audit(
        learner,
        scan_prep,
        Xt_scan,
        yt,
        zt,
        dataset=bundle.name,
        seed=seed,
        fixed_test=fixed_test,
        n_placebo=n_placebo,
    )

    return (
        pd.DataFrame(
            result_rows
        ),
        rules,
        mechanism,
    )


# =============================================================================
# Mechanism summaries
# =============================================================================

def summarise_mechanism(
    mechanism: pd.DataFrame,
) -> pd.DataFrame:
    if mechanism.empty:
        return pd.DataFrame()

    rows = []

    for dataset, g in mechanism.groupby(
        "dataset"
    ):
        rows.append(
            {
                "dataset": dataset,
                "accepted_terms_audited": int(
                    len(
                        g
                    )
                ),
                "accepted_seeds_audited": int(
                    g[
                        "seed"
                    ].nunique()
                ),
                "mean_test_support": float(
                    g[
                        "test_support"
                    ].mean()
                ),
                "direction_agreement_rate": float(
                    g[
                        "direction_agreement"
                    ].mean()
                ),
                "mean_abs_residual_reduction_base_to_full": float(
                    g[
                        "base_absolute_residual_reduction"
                    ].mean()
                ),
                "positive_residual_reduction_rate": float(
                    np.mean(
                        g[
                            "base_absolute_residual_reduction"
                        ]
                        > 0
                    )
                ),
                "mean_local_logloss_advantage": float(
                    g[
                        "local_logloss_advantage_base_to_full"
                    ].mean()
                ),
                "positive_local_logloss_rate": float(
                    np.mean(
                        g[
                            "local_logloss_advantage_base_to_full"
                        ]
                        > 0
                    )
                ),
                "mean_global_term_ablation_gain": float(
                    g[
                        "global_term_ablation_gain"
                    ].mean()
                ),
                "positive_ablation_gain_rate": float(
                    np.mean(
                        g[
                            "global_term_ablation_gain"
                        ]
                        > 0
                    )
                ),
                "mean_placebo_percentile": float(
                    g[
                        "placebo_percentile"
                    ].mean()
                ),
                "placebo_p_lt_0_05_rate": float(
                    np.mean(
                        g[
                            "placebo_empirical_p"
                        ]
                        < 0.05
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def feature_recurrence(
    mechanism: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for r in mechanism.itertuples():
        features = (
            str(
                r.features
            ).split(
                ";"
            )
            if pd.notna(
                r.features
            )
            and str(
                r.features
            ).strip()
            else []
        )

        for feature in features:
            rows.append(
                {
                    "dataset": r.dataset,
                    "seed": int(
                        r.seed
                    ),
                    "term": int(
                        r.term
                    ),
                    "feature": feature,
                    "coefficient_sign": int(
                        np.sign(
                            r.coefficient
                        )
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    d = pd.DataFrame(
        rows
    )

    return (
        d.groupby(
            [
                "dataset",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            selected_terms=(
                "term",
                "size",
            ),
            selected_seeds=(
                "seed",
                "nunique",
            ),
            mean_coefficient_sign=(
                "coefficient_sign",
                "mean",
            ),
        )
        .sort_values(
            [
                "dataset",
                "selected_seeds",
                "selected_terms",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )


def _decode_mask(
    hex_string: str,
    length: int,
) -> np.ndarray:
    raw = bytes.fromhex(
        hex_string
    )

    bits = np.unpackbits(
        np.frombuffer(
            raw,
            dtype=np.uint8,
        )
    )

    return bits[
        :int(
            length
        )
    ].astype(
        bool
    )


def fixed_test_rule_stability(
    mechanism: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pairwise best-mask Jaccard across seeds for datasets sharing the same TEST
    observations across runs.

    For each seed pair we compare every accepted term from seed A with every
    accepted term from seed B and record the maximum Jaccard overlap.
    """
    eligible = mechanism[
        mechanism[
            "fixed_test_mask_available"
        ].astype(
            bool
        )
        & mechanism[
            "mask_packed_hex"
        ].astype(
            str
        ).str.len().gt(
            0
        )
    ].copy()

    rows = []

    for dataset, g in eligible.groupby(
        "dataset"
    ):
        seed_groups = {
            int(
                seed
            ): sub
            for seed, sub
            in g.groupby(
                "seed"
            )
        }

        seeds = sorted(
            seed_groups
        )

        for i in range(
            len(
                seeds
            )
        ):
            for j in range(
                i + 1,
                len(
                    seeds
                )
            ):
                a = seed_groups[
                    seeds[
                        i
                    ]
                ]
                b = seed_groups[
                    seeds[
                        j
                    ]
                ]

                best = 0.0

                for ra in a.itertuples():
                    ma = _decode_mask(
                        ra.mask_packed_hex,
                        ra.mask_length,
                    )

                    for rb in b.itertuples():
                        if (
                            int(
                                ra.mask_length
                            )
                            != int(
                                rb.mask_length
                            )
                        ):
                            continue

                        mb = _decode_mask(
                            rb.mask_packed_hex,
                            rb.mask_length,
                        )

                        union = np.sum(
                            ma
                            | mb
                        )

                        if union == 0:
                            continue

                        jac = float(
                            np.sum(
                                ma
                                & mb
                            )
                            / union
                        )

                        best = max(
                            best,
                            jac,
                        )

                rows.append(
                    {
                        "dataset": dataset,
                        "seed_a": seeds[
                            i
                        ],
                        "seed_b": seeds[
                            j
                        ],
                        "best_rule_mask_jaccard": best,
                    }
                )

    return pd.DataFrame(
        rows
    )


def summarise_fixed_test_stability(
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()

    return (
        pairwise.groupby(
            "dataset",
            as_index=False,
        )
        .agg(
            seed_pairs=(
                "best_rule_mask_jaccard",
                "size",
            ),
            mean_best_rule_mask_jaccard=(
                "best_rule_mask_jaccard",
                "mean",
            ),
            median_best_rule_mask_jaccard=(
                "best_rule_mask_jaccard",
                "median",
            ),
            max_best_rule_mask_jaccard=(
                "best_rule_mask_jaccard",
                "max",
            ),
        )
    )


# =============================================================================
# Methodological-role table
# =============================================================================

def added_dataset_roles(
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "Bank Marketing",
                "methodological_question": (
                    "Does repair generalize prospectively when the covariate "
                    "distribution evolves over time and residual interactions "
                    "involve client, campaign and macroeconomic context?"
                ),
                "design_choice": (
                    "bank-additional-full; duration removed; chronological "
                    "50/20/15/15 split."
                ),
            },
            {
                "dataset": "Default of Credit Card Clients",
                "methodological_question": (
                    "Can scan repair identify localized probability errors "
                    "within correlated longitudinal repayment and credit-history "
                    "blocks?"
                ),
                "design_choice": (
                    "Fixed stratified 15% TEST; SEX/EDUCATION/MARRIAGE treated "
                    "as categorical; repayment histories retained as ordered "
                    "numeric variables."
                ),
            },
            {
                "dataset": "NIJ Recidivism Forecasting Challenge",
                "methodological_question": (
                    "Can sparse structured correction improve an official "
                    "probability-forecasting task in a high-stakes justice "
                    "setting without relying on post-release future information?"
                ),
                "design_choice": (
                    "One-year recidivism; official challenge train/test-1 split; "
                    "predictors restricted to Year-1 baseline variables; Race "
                    "and Gender excluded from predictive repair dimensions."
                ),
            },
        ]
    )
