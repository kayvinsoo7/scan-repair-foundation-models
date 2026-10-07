
"""
Final frozen Structured Scan benchmark update
=============================================

Purpose
-------
Update the existing Block B correction-method benchmark with the frozen
Structured Scan v2.3 WITHOUT rerunning unchanged comparators such as MCGrad,
Multiaccuracy-style Boost, Newton Rule Boosting, or continued-tree repair.

Why this is fair
----------------
The original Block B is fully paired:
    - same synthetic replicate
    - same train/discovery/selection/test split
    - same frozen XGBoost base predictions

For every historical Block B condition, this module regenerates the exact
condition and base model, verifies compatibility against the saved benchmark,
and runs only the frozen final scanner.

Frozen Structured Scan
----------------------
Search:
    - adaptive numeric threshold proposals
    - ALTSS-style categorical proposal machinery
    - diversity-aware beam
    - coordinate refinement
    - protected v1 fixed-five-bin candidate

Selection:
    - two-half stability ranking on SELECTION
    - same correction direction required across halves
    - rank by minimum per-observation half gain
    - full-SELECTION nominal likelihood-ratio p < 0.05
    - no multiplicity adjustment

The final benchmark should therefore answer:
    - how v2.3 compares with the unchanged correction methods;
    - not whether it can be tuned until it beats any specific comparator.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

from mdss_treescan_stage1 import (
    SyntheticConfig,
    generate_structured_residual_data,
    sigmoid,
)
from benchmark_treescan import (
    fit_fixed_budget_xgb,
)
from crossfit_repair_benchmark_strict import (
    four_way_split,
)

from block_b_correction_benchmarks import (
    BlockBCondition,
    evaluate_predictions,
)

from adaptive_structured_scanner_v2_2 import (
    AdaptiveStructuredScannerV22,
)
from adaptive_structured_scanner_v2_3 import (
    StabilitySelectedScanRepairLearner,
)


FINAL_METHOD_NAME = "Structured Scan Repair (final)"
DEVELOPMENT_V1_NAME = "Structured Scan Repair v1 (development)"

FINAL_BEAM_WIDTH = 24
FINAL_ADAPTIVE_CANDIDATES = 4
FINAL_ALPHA = 0.05


# ============================================================================
# Frozen final scanner factory
# ============================================================================

def make_final_scanner(
    cond: BlockBCondition,
    *,
    discovery_n: int,
    selection_n: int,
) -> StabilitySelectedScanRepairLearner:
    """
    Exact frozen v2.3 settings carried forward from scanner development.
    """
    scanner = AdaptiveStructuredScannerV22(
        feature_types=[
            "numeric"
        ] * cond.p,
        max_rule_features=cond.k,
        beam_width=FINAL_BEAM_WIDTH,
        numeric_shortlist_per_direction=8,
        numeric_exact_pool=6,
        per_feature_exact_candidates=2,
        refine_top_n=3,
        refinement_passes=1,
        min_size=max(
            35,
            int(
                round(
                    0.015
                    * discovery_n
                )
            ),
        ),
        min_events=3,
        min_nonevents=3,
        candidate_pool_size=4,
        candidate_mask_jaccard_max=0.90,
    )

    learner = StabilitySelectedScanRepairLearner(
        scanner,
        repair_rounds=cond.repair_rounds,
        adaptive_candidates=FINAL_ADAPTIVE_CANDIDATES,
        include_legacy_candidate=True,
        legacy_beam_width=cond.beam_width,
        legacy_n_bins=5,
        coefficient_l2=1e-3,
        alpha=FINAL_ALPHA,
        validation_split_seed=cond.seed + 17001,
        selection_min_size=max(
            20,
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
            10,
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

    return learner


# ============================================================================
# One final-scanner condition
# ============================================================================

def run_final_scanner_condition(
    cond: BlockBCondition,
    *,
    benchmark_role: str,
) -> pd.DataFrame:
    """
    Regenerate ONE historical Block B condition and run only:
        Base XGBoost
        Structured Scan Repair (final)

    Base row is returned for identity checking.
    """
    cfg = SyntheticConfig(
        n=cond.n,
        p=cond.p,
        interaction_order=cond.k,
        subgroup_prevalence=cond.support,
        subgroup_log_odds=float(
            np.log(
                cond.odds_ratio
            )
        ),
        event_prevalence=cond.event_prevalence,
        seed=cond.seed,
    )

    data = generate_structured_residual_data(
        cfg
    )

    split = four_way_split(
        cond.n,
        train_fraction=0.50,
        discovery_fraction=0.20,
        selection_fraction=0.15,
        seed=cond.seed + 1000,
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

    # -------------------------------------------------------------
    # Same historical frozen XGBoost specification.
    # -------------------------------------------------------------
    t0 = time.perf_counter()

    base = fit_fixed_budget_xgb(
        Xtr,
        ytr,
        max_depth=cond.tree_depth,
        rounds=cond.base_tree_rounds,
        learning_rate=0.08,
        seed=cond.seed,
    )

    base_seconds = (
        time.perf_counter()
        - t0
    )

    zd = base.logits(
        Xd
    )
    zs = base.logits(
        Xs
    )
    zt = base.logits(
        Xt
    )

    pt = sigmoid(
        zt
    )

    base_row = evaluate_predictions(
        "Base XGBoost",
        yt,
        pt,
        st,
        runtime_seconds=base_seconds,
        correction_seconds=0.0,
        updates=0,
        scan_terms=0,
        tree_terms=0,
    )

    # -------------------------------------------------------------
    # Frozen Structured Scan v2.3.
    # -------------------------------------------------------------
    learner = make_final_scanner(
        cond,
        discovery_n=len(
            Xd
        ),
        selection_n=len(
            Xs
        ),
    )

    t0 = time.perf_counter()

    learner.fit(
        Xd,
        yd,
        zd,
        Xs,
        ys,
        zs,
    )

    correction_seconds = (
        time.perf_counter()
        - t0
    )

    z_final = learner.repair_logits(
        Xt,
        zt,
    )

    accepted_rounds = [
        h
        for h in learner.history_
        if h.selected_source
        is not None
    ]

    first = (
        learner.history_[0]
        if learner.history_
        else None
    )

    final_row = evaluate_predictions(
        FINAL_METHOD_NAME,
        yt,
        sigmoid(
            z_final
        ),
        st,
        runtime_seconds=(
            base_seconds
            + correction_seconds
        ),
        correction_seconds=(
            correction_seconds
        ),
        updates=len(
            accepted_rounds
        ),
        scan_terms=len(
            accepted_rounds
        ),
        tree_terms=0,
        extra={
            "v23_first_round_proposed_candidates": (
                first.proposed_candidates
                if first is not None
                else 0
            ),
            "v23_first_round_stable_candidates": (
                first.stable_candidates
                if first is not None
                else 0
            ),
            "v23_first_round_selected_source": (
                first.selected_source
                if first is not None
                else None
            ),
            "v23_first_round_stability_score": (
                first.selected_stability_score
                if first is not None
                else np.nan
            ),
            "v23_first_round_half_p_a": (
                first.selected_half_p_a
                if first is not None
                else np.nan
            ),
            "v23_first_round_half_p_b": (
                first.selected_half_p_b
                if first is not None
                else np.nan
            ),
            "v23_first_round_full_selection_gain": (
                first.full_selection_gain
                if first is not None
                else np.nan
            ),
            "v23_first_round_full_selection_p": (
                first.full_selection_p
                if first is not None
                else np.nan
            ),
            "v23_first_round_delta_consistency": (
                first.selected_delta_consistency_ratio
                if first is not None
                else np.nan
            ),
            "v23_first_round_support_consistency": (
                first.selected_support_consistency_ratio
                if first is not None
                else np.nan
            ),
            "v23_accepted_legacy_terms": int(
                sum(
                    h.selected_source
                    == "legacy_fixed5"
                    for h in accepted_rounds
                )
            ),
            "v23_accepted_adaptive_terms": int(
                sum(
                    h.selected_source
                    == "adaptive"
                    for h in accepted_rounds
                )
            ),
        },
    )

    out = pd.DataFrame(
        [
            base_row,
            final_row,
        ]
    )

    for key, value in asdict(
        cond
    ).items():
        out[
            key
        ] = value

    out[
        "delta"
    ] = float(
        np.log(
            cond.odds_ratio
        )
    )

    out[
        "realized_event_prevalence"
    ] = float(
        data.y.mean()
    )

    out[
        "realized_subgroup_prevalence"
    ] = float(
        data.subgroup_mask.mean()
    )

    out[
        "base_fit_seconds"
    ] = float(
        base_seconds
    )

    out[
        "benchmark_role"
    ] = benchmark_role

    return out


# ============================================================================
# Historical base identity check
# ============================================================================

BASE_IDENTITY_METRICS = (
    "logloss",
    "auroc",
    "auprc",
    "brier",
    "ece_15",
    "true_subgroup_mean_residual",
)


def historical_base_row(
    prior_matched: pd.DataFrame,
    *,
    support: float,
    odds_ratio: float,
    seed: int,
) -> pd.Series:
    d = prior_matched[
        (
            np.isclose(
                prior_matched[
                    "support"
                ],
                support,
            )
        )
        & (
            np.isclose(
                prior_matched[
                    "odds_ratio"
                ],
                odds_ratio,
            )
        )
        & (
            prior_matched[
                "seed"
            ].astype(int)
            == int(seed)
        )
        & (
            prior_matched[
                "method"
            ]
            == "Base XGBoost"
        )
    ]

    if len(d) != 1:
        raise ValueError(
            "Expected exactly one historical Base XGBoost row for "
            f"s={support}, OR={odds_ratio}, seed={seed}; found {len(d)}."
        )

    return d.iloc[0]


def check_base_identity(
    regenerated: pd.DataFrame,
    prior_matched: pd.DataFrame,
    *,
    atol: float = 1e-10,
) -> pd.DataFrame:
    """
    Verify that regenerated base metrics match the saved benchmark.

    Tiny floating-point differences are tolerated; substantive differences
    should stop the merge because the paired benchmark would no longer share
    an identical starting point.
    """
    base = regenerated[
        regenerated[
            "method"
        ]
        == "Base XGBoost"
    ]

    rows = []

    for _, row in base.iterrows():
        historical = historical_base_row(
            prior_matched,
            support=float(
                row[
                    "support"
                ]
            ),
            odds_ratio=float(
                row[
                    "odds_ratio"
                ]
            ),
            seed=int(
                row[
                    "seed"
                ]
            ),
        )

        record = {
            "support": float(
                row[
                    "support"
                ]
            ),
            "odds_ratio": float(
                row[
                    "odds_ratio"
                ]
            ),
            "seed": int(
                row[
                    "seed"
                ]
            ),
        }

        max_abs_diff = 0.0

        for metric in BASE_IDENTITY_METRICS:
            new = float(
                row[
                    metric
                ]
            )
            old = float(
                historical[
                    metric
                ]
            )

            diff = abs(
                new - old
            )

            record[
                f"{metric}_abs_diff"
            ] = diff

            max_abs_diff = max(
                max_abs_diff,
                diff,
            )

        record[
            "max_abs_diff"
        ] = max_abs_diff

        record[
            "identity_pass"
        ] = bool(
            max_abs_diff
            <= atol
        )

        rows.append(
            record
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# Final benchmark merge
# ============================================================================

DERIVED_OLD_COLUMNS = {
    "base_logloss",
    "base_auroc",
    "base_auprc",
    "base_brier",
    "base_ece_15",
    "base_true_subgroup_mean_residual",
    "base_runtime_seconds",
    "logloss_advantage_vs_base",
    "auroc_advantage_vs_base",
    "auprc_advantage_vs_base",
    "brier_advantage_vs_base",
    "ece_reduction_vs_base",
    "subgroup_bias_reduction_vs_base",
    "beats_base_logloss",
    "structured_scan_logloss",
    "structured_scan_advantage_vs_method",
}


def _raw_like_prior(
    prior_matched: pd.DataFrame,
) -> pd.DataFrame:
    keep = [
        c
        for c in prior_matched.columns
        if c
        not in DERIVED_OLD_COLUMNS
    ]

    return prior_matched[
        keep
    ].copy()


def build_final_matched(
    prior_matched: pd.DataFrame,
    final_v23_raw: pd.DataFrame,
    *,
    retain_development_v1: bool = False,
) -> pd.DataFrame:
    """
    Merge unchanged historical comparators with frozen v2.3.

    Historical v1 Structured Scan is excluded from the main benchmark unless
    explicitly retained for development comparison.
    """
    historical = _raw_like_prior(
        prior_matched
    )

    old_scan_mask = (
        historical[
            "method"
        ]
        == "Structured Scan Repair"
    )

    historical.loc[
        historical["method"] == "Competitive Tree-Scan",
        "method",
    ] = "Competitive Tree-Scan (v1 scan)"

    if retain_development_v1:
        historical.loc[
            old_scan_mask,
            "method",
        ] = DEVELOPMENT_V1_NAME
    else:
        historical = historical.loc[
            ~old_scan_mask
        ].copy()

    final_only = final_v23_raw[
        final_v23_raw[
            "method"
        ]
        == FINAL_METHOD_NAME
    ].copy()

    # Align union of columns.
    raw = pd.concat(
        [
            historical,
            final_only,
        ],
        ignore_index=True,
        sort=False,
    )

    keys = [
        "benchmark_role",
        "support",
        "odds_ratio",
        "seed",
    ]

    base_cols = [
        "logloss",
        "auroc",
        "auprc",
        "brier",
        "ece_15",
        "true_subgroup_mean_residual",
        "runtime_seconds",
    ]

    base = (
        raw.loc[
            raw[
                "method"
            ]
            == "Base XGBoost",
            keys
            + base_cols,
        ]
        .rename(
            columns={
                c: f"base_{c}"
                for c in base_cols
            }
        )
    )

    out = raw.merge(
        base,
        on=keys,
        how="left",
    )

    out[
        "logloss_advantage_vs_base"
    ] = (
        out[
            "base_logloss"
        ]
        - out[
            "logloss"
        ]
    )

    out[
        "auroc_advantage_vs_base"
    ] = (
        out[
            "auroc"
        ]
        - out[
            "base_auroc"
        ]
    )

    out[
        "auprc_advantage_vs_base"
    ] = (
        out[
            "auprc"
        ]
        - out[
            "base_auprc"
        ]
    )

    out[
        "brier_advantage_vs_base"
    ] = (
        out[
            "base_brier"
        ]
        - out[
            "brier"
        ]
    )

    out[
        "ece_reduction_vs_base"
    ] = (
        out[
            "base_ece_15"
        ]
        - out[
            "ece_15"
        ]
    )

    out[
        "subgroup_bias_reduction_vs_base"
    ] = (
        np.abs(
            out[
                "base_true_subgroup_mean_residual"
            ]
        )
        - np.abs(
            out[
                "true_subgroup_mean_residual"
            ]
        )
    )

    out[
        "beats_base_logloss"
    ] = (
        out[
            "logloss_advantage_vs_base"
        ]
        > 0
    ).astype(
        int
    )

    final_scan = (
        raw.loc[
            raw[
                "method"
            ]
            == FINAL_METHOD_NAME,
            keys
            + [
                "logloss"
            ],
        ]
        .rename(
            columns={
                "logloss":
                    "final_scan_logloss"
            }
        )
    )

    out = out.merge(
        final_scan,
        on=keys,
        how="left",
    )

    out[
        "final_scan_advantage_vs_method"
    ] = (
        out[
            "logloss"
        ]
        - out[
            "final_scan_logloss"
        ]
    )

    # Positive => final Structured Scan has lower TEST log-loss.
    return out


# ============================================================================
# Summary helpers
# ============================================================================

def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 3000,
    seed: int = 123,
) -> Tuple[
    float,
    float,
]:
    x = np.asarray(
        values,
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

    idx = rng.integers(
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
        idx
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


def safe_wilcoxon(
    values: Sequence[float],
) -> float:
    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if (
        len(
            x
        )
        == 0
        or np.allclose(
            x,
            0.0,
        )
    ):
        return np.nan

    try:
        return float(
            wilcoxon(
                x,
                zero_method="wilcox",
            ).pvalue
        )
    except ValueError:
        return np.nan


def summarise_final_benchmark(
    matched: pd.DataFrame,
    *,
    n_boot: int = 3000,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "benchmark_role",
        "support",
        "odds_ratio",
        "method",
    ]

    for (
        role,
        support,
        odds_ratio,
        method,
    ), g in matched.groupby(
        group_cols
    ):
        dll = g[
            "logloss_advantage_vs_base"
        ].to_numpy(
            dtype=float
        )

        dbias = g[
            "subgroup_bias_reduction_vs_base"
        ].to_numpy(
            dtype=float
        )

        final_adv = g[
            "final_scan_advantage_vs_method"
        ].to_numpy(
            dtype=float
        )

        ll_lo, ll_hi = bootstrap_mean_ci(
            dll,
            n_boot=n_boot,
            seed=int(
                100000
                * float(
                    support
                )
                + 1000
                * float(
                    odds_ratio
                )
                + len(
                    str(
                        method
                    )
                )
            ),
        )

        bias_lo, bias_hi = bootstrap_mean_ci(
            dbias,
            n_boot=n_boot,
            seed=int(
                200000
                * float(
                    support
                )
                + 2000
                * float(
                    odds_ratio
                )
                + len(
                    str(
                        method
                    )
                )
            ),
        )

        final_lo, final_hi = bootstrap_mean_ci(
            final_adv,
            n_boot=n_boot,
            seed=int(
                300000
                * float(
                    support
                )
                + 3000
                * float(
                    odds_ratio
                )
                + len(
                    str(
                        method
                    )
                )
            ),
        )

        rows.append({
            "benchmark_role": role,
            "support": support,
            "odds_ratio": odds_ratio,
            "method": method,
            "runs": len(
                g
            ),
            "mean_logloss": float(
                g[
                    "logloss"
                ].mean()
            ),
            "mean_logloss_advantage_vs_base": float(
                np.nanmean(
                    dll
                )
            ),
            "logloss_advantage_ci_low": ll_lo,
            "logloss_advantage_ci_high": ll_hi,
            "logloss_t_p": (
                float(
                    ttest_1samp(
                        dll,
                        0.0,
                        nan_policy="omit",
                    ).pvalue
                )
                if np.any(
                    np.abs(
                        dll
                    )
                    > 0
                )
                else np.nan
            ),
            "logloss_wilcoxon_p": safe_wilcoxon(
                dll
            ),
            "win_rate_vs_base": float(
                np.mean(
                    dll
                    > 0
                )
            ),
            "mean_subgroup_bias_reduction": float(
                np.nanmean(
                    dbias
                )
            ),
            "subgroup_bias_ci_low": bias_lo,
            "subgroup_bias_ci_high": bias_hi,
            "mean_auroc_advantage": float(
                g[
                    "auroc_advantage_vs_base"
                ].mean()
            ),
            "mean_auprc_advantage": float(
                g[
                    "auprc_advantage_vs_base"
                ].mean()
            ),
            "mean_brier_advantage": float(
                g[
                    "brier_advantage_vs_base"
                ].mean()
            ),
            "mean_ece_reduction": float(
                g[
                    "ece_reduction_vs_base"
                ].mean()
            ),
            "mean_updates": float(
                g[
                    "updates"
                ].mean()
            ),
            "mean_correction_seconds": float(
                g[
                    "correction_seconds"
                ].mean()
            ),
            "mean_final_scan_advantage_vs_method": float(
                np.nanmean(
                    final_adv
                )
            ),
            "final_scan_vs_method_ci_low": final_lo,
            "final_scan_vs_method_ci_high": final_hi,
            "final_scan_vs_method_t_p": (
                float(
                    ttest_1samp(
                        final_adv,
                        0.0,
                        nan_policy="omit",
                    ).pvalue
                )
                if np.any(
                    np.abs(
                        final_adv
                    )
                    > 0
                )
                else np.nan
            ),
            "final_scan_vs_method_wilcoxon_p": safe_wilcoxon(
                final_adv
            ),
            "final_scan_vs_method_win_rate": float(
                np.mean(
                    final_adv
                    > 0
                )
            ),
        })

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "benchmark_role",
                "support",
                "odds_ratio",
                "method",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def primary_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    d = summary[
        (
            summary[
                "benchmark_role"
            ]
            == "primary"
        )
        & np.isclose(
            summary[
                "support"
            ],
            0.08,
        )
        & np.isclose(
            summary[
                "odds_ratio"
            ],
            8.0,
        )
    ].copy()

    preferred = [
        "Base XGBoost",
        "Continued-tree repair",
        "Multiaccuracy-style Boost",
        "Official MCGrad",
        "Newton Rule Boosting",
        FINAL_METHOD_NAME,
        "Competitive Tree-Scan (v1 scan)",
    ]

    d[
        "method"
    ] = pd.Categorical(
        d[
            "method"
        ],
        categories=preferred,
        ordered=True,
    )

    d = d.sort_values(
        "method"
    )

    return d[
        [
            "method",
            "runs",
            "mean_logloss",
            "mean_logloss_advantage_vs_base",
            "logloss_advantage_ci_low",
            "logloss_advantage_ci_high",
            "logloss_t_p",
            "logloss_wilcoxon_p",
            "win_rate_vs_base",
            "mean_subgroup_bias_reduction",
            "mean_auroc_advantage",
            "mean_auprc_advantage",
            "mean_ece_reduction",
            "mean_final_scan_advantage_vs_method",
            "final_scan_vs_method_ci_low",
            "final_scan_vs_method_ci_high",
            "final_scan_vs_method_t_p",
            "final_scan_vs_method_wilcoxon_p",
            "final_scan_vs_method_win_rate",
            "mean_updates",
            "mean_correction_seconds",
        ]
    ]


def development_v1_vs_final(
    prior_matched: pd.DataFrame,
    final_v23_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Separate diagnostic table: historical v1 scan vs frozen final scan.
    Not part of the main final correction benchmark.
    """
    keys = [
        "benchmark_role",
        "support",
        "odds_ratio",
        "seed",
    ]

    v1 = (
        prior_matched.loc[
            prior_matched[
                "method"
            ]
            == "Structured Scan Repair",
            keys
            + [
                "logloss",
                "true_subgroup_mean_residual",
            ],
        ]
        .rename(
            columns={
                "logloss":
                    "v1_logloss",
                "true_subgroup_mean_residual":
                    "v1_subgroup_residual",
            }
        )
    )

    final = (
        final_v23_raw.loc[
            final_v23_raw[
                "method"
            ]
            == FINAL_METHOD_NAME,
            keys
            + [
                "logloss",
                "true_subgroup_mean_residual",
            ],
        ]
        .rename(
            columns={
                "logloss":
                    "final_logloss",
                "true_subgroup_mean_residual":
                    "final_subgroup_residual",
            }
        )
    )

    out = v1.merge(
        final,
        on=keys,
        how="inner",
    )

    out[
        "final_logloss_advantage_vs_v1"
    ] = (
        out[
            "v1_logloss"
        ]
        - out[
            "final_logloss"
        ]
    )

    out[
        "final_abs_subgroup_residual_reduction_vs_v1"
    ] = (
        np.abs(
            out[
                "v1_subgroup_residual"
            ]
        )
        - np.abs(
            out[
                "final_subgroup_residual"
            ]
        )
    )

    return out
