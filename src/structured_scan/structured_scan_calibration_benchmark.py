
"""
Non-MCGrad calibration benchmark for Structured Scan Repair.

This benchmark compares:
    Base XGBoost
    Continued-tree repair (historical paired result)
    Platt scaling
    Isotonic regression
    Beta calibration
    Structured Scan Repair (final historical paired result)

The new calibrators are fitted on the combined DISCOVERY + SELECTION pool,
which gives them the full labelled correction budget available after base
training. TEST remains untouched.

Conditions:
    localized primary: support=.08, OR=8, seeds 20001..20050
    broad control:     support=.25, OR=8, seeds 20001..20020

The base generator/split/model exactly match the final Block B configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.stats import ttest_1samp, wilcoxon

from mdss_treescan_stage1 import (
    SyntheticConfig,
    generate_structured_residual_data,
    sigmoid,
)
from benchmark_treescan import fit_fixed_budget_xgb
from crossfit_repair_benchmark_strict import four_way_split
from block_b_correction_benchmarks import (
    evaluate_predictions,
    clip_prob,
    logit,
)

warnings.filterwarnings("ignore")


def _fit_logistic_no_penalty(
    X: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    try:
        model = LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(X, y)
        return model
    except Exception:
        model = LogisticRegression(
            C=1e9,
            solver="lbfgs",
            max_iter=2000,
        )
        model.fit(X, y)
        return model


def fit_platt(
    p: np.ndarray,
    y: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    z = logit(
        clip_prob(p)
    ).reshape(-1, 1)

    model = _fit_logistic_no_penalty(
        z,
        y,
    )

    def predict(p_new: np.ndarray) -> np.ndarray:
        z_new = logit(
            clip_prob(p_new)
        ).reshape(-1, 1)

        return clip_prob(
            model.predict_proba(
                z_new
            )[:, 1]
        )

    return predict


def fit_beta_calibration(
    p: np.ndarray,
    y: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    p = clip_prob(p)

    X = np.column_stack(
        [
            np.log(p),
            np.log1p(-p),
        ]
    )

    model = _fit_logistic_no_penalty(
        X,
        y,
    )

    def predict(p_new: np.ndarray) -> np.ndarray:
        q = clip_prob(
            p_new
        )

        X_new = np.column_stack(
            [
                np.log(q),
                np.log1p(-q),
            ]
        )

        return clip_prob(
            model.predict_proba(
                X_new
            )[:, 1]
        )

    return predict


def run_condition(
    support: float,
    odds_ratio: float,
    seed: int,
) -> pd.DataFrame:
    cfg = SyntheticConfig(
        n=12_000,
        p=30,
        interaction_order=5,
        subgroup_prevalence=support,
        subgroup_log_odds=float(
            np.log(
                odds_ratio
            )
        ),
        event_prevalence=0.10,
        seed=seed,
    )

    data = generate_structured_residual_data(
        cfg
    )

    split = four_way_split(
        12_000,
        train_fraction=0.50,
        discovery_fraction=0.20,
        selection_fraction=0.15,
        seed=seed + 1000,
    )

    Xtr = data.X[
        split.train
    ]
    ytr = data.y[
        split.train
    ]

    Xd = data.X[
        split.discovery
    ]
    yd = data.y[
        split.discovery
    ]

    Xs = data.X[
        split.selection
    ]
    ys = data.y[
        split.selection
    ]

    Xt = data.X[
        split.test
    ]
    yt = data.y[
        split.test
    ]

    st = data.subgroup_mask[
        split.test
    ]

    base = fit_fixed_budget_xgb(
        Xtr,
        ytr,
        max_depth=2,
        rounds=70,
        learning_rate=0.08,
        seed=seed,
    )

    pd0 = sigmoid(
        base.logits(
            Xd
        )
    )

    ps0 = sigmoid(
        base.logits(
            Xs
        )
    )

    pt0 = sigmoid(
        base.logits(
            Xt
        )
    )

    p_cal = np.concatenate(
        [
            pd0,
            ps0,
        ]
    )

    y_cal = np.concatenate(
        [
            yd,
            ys,
        ]
    )

    predictions: Dict[
        str,
        np.ndarray,
    ] = {
        "Base XGBoost": pt0,
    }

    platt = fit_platt(
        p_cal,
        y_cal,
    )

    predictions[
        "Platt scaling"
    ] = platt(
        pt0
    )

    iso = IsotonicRegression(
        out_of_bounds="clip"
    )

    iso.fit(
        p_cal,
        y_cal,
    )

    predictions[
        "Isotonic regression"
    ] = clip_prob(
        iso.predict(
            pt0
        )
    )

    beta = fit_beta_calibration(
        p_cal,
        y_cal,
    )

    predictions[
        "Beta calibration"
    ] = beta(
        pt0
    )

    rows = []

    for method, pred in predictions.items():
        row = evaluate_predictions(
            method,
            yt,
            pred,
            st,
            runtime_seconds=np.nan,
            correction_seconds=np.nan,
            updates=0,
            scan_terms=0,
            tree_terms=0,
        )

        row.update(
            {
                "support": support,
                "odds_ratio": odds_ratio,
                "seed": seed,
                "benchmark_role": (
                    "primary"
                    if (
                        np.isclose(
                            support,
                            0.08,
                        )
                        and np.isclose(
                            odds_ratio,
                            8.0,
                        )
                    )
                    else "control"
                ),
            }
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_plan() -> List[
    Tuple[
        float,
        float,
        int,
    ]
]:
    return (
        [
            (
                0.08,
                8.0,
                seed,
            )
            for seed in range(
                20001,
                20051,
            )
        ]
        + [
            (
                0.25,
                8.0,
                seed,
            )
            for seed in range(
                20001,
                20021,
            )
        ]
    )
