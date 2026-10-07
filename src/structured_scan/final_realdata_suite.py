
"""
Final six-dataset real-data suite for Structured Scan Repair.

This is the only real-data runner used by the final paper code.

Datasets
--------
Adult Income
APS Failure at Scania Trucks
Diabetes 130-US Hospitals
Bank Marketing
Default of Credit Card Clients
NIJ Recidivism Forecasting Challenge

The runner:
1. downloads/loads each public dataset;
2. tunes one XGBoost base using TRAIN only;
3. applies the frozen Structured Scan Repair;
4. evaluates 10 precommitted seeds per dataset;
5. audits every accepted correction on untouched TEST data;
6. computes ablation and score-matched placebo diagnostics;
7. saves paper-ready summaries.

No scanner tuning occurs in this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple
import json
import time

import numpy as np
import pandas as pd

import structured_scan_realdata_benchmarks as core
import structured_scan_realdata_extended_mechanism as ext


DATASET_CONFIG = {
    "adult": {
        "name": "Adult Income",
        "seeds": tuple(range(41001, 41011)),
        "tuning_seed": 41001,
    },
    "aps": {
        "name": "APS Failure at Scania Trucks",
        "seeds": tuple(range(41001, 41011)),
        "tuning_seed": 41001,
    },
    "diabetes130": {
        "name": "Diabetes 130-US Hospitals",
        "seeds": tuple(range(41001, 41011)),
        "tuning_seed": 41001,
    },
    "bank": {
        "name": "Bank Marketing",
        "seeds": tuple(range(42001, 42011)),
        "tuning_seed": 42001,
    },
    "credit": {
        "name": "Default of Credit Card Clients",
        "seeds": tuple(range(42001, 42011)),
        "tuning_seed": 42001,
    },
    "nij": {
        "name": "NIJ Recidivism Forecasting Challenge",
        "seeds": tuple(range(42001, 42011)),
        "tuning_seed": 42001,
    },
}


def load_all_datasets(
    data_dir: Path,
    *,
    allow_download: bool = True,
) -> Dict[str, core.DatasetBundle]:
    data_dir = Path(data_dir)

    return {
        "adult": core.load_adult(
            data_dir,
            allow_download=allow_download,
        ),
        "aps": core.load_aps(
            data_dir,
            allow_download=allow_download,
        ),
        "diabetes130": core.load_diabetes130(
            data_dir,
            allow_download=allow_download,
        ),
        "bank": ext.load_bank_marketing(
            data_dir,
            allow_download=allow_download,
        ),
        "credit": ext.load_credit_default(
            data_dir,
            allow_download=allow_download,
        ),
        "nij": ext.load_nij_recidivism(
            data_dir,
            allow_download=allow_download,
        ),
    }


def _split_for(
    key: str,
    bundle: core.DatasetBundle,
    *,
    seed: int,
) -> core.FourWaySplit:
    if key in {
        "bank",
        "credit",
        "nij",
    }:
        return ext.make_extended_split(
            bundle,
            seed=seed,
        )

    return core.make_dataset_split(
        bundle,
        seed=seed,
    )


def tune_all_bases(
    datasets: Mapping[str, core.DatasetBundle],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Dict[str, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    params_path = (
        output_dir
        / "realdata_selected_xgb_params.json"
    )

    if params_path.exists() and not overwrite:
        return json.loads(
            params_path.read_text()
        )

    selected = {}

    for key, cfg in DATASET_CONFIG.items():
        bundle = datasets[
            key
        ]

        tuning_seed = int(
            cfg[
                "tuning_seed"
            ]
        )

        print(
            "\n"
            + "=" * 80
        )
        print(
            "Tuning:",
            bundle.name,
        )

        split = _split_for(
            key,
            bundle,
            seed=tuning_seed,
        )

        if key in {
            "bank",
            "credit",
            "nij",
        }:
            params, table = ext.select_xgb_params_extended(
                bundle,
                split,
                seed=tuning_seed,
            )
        else:
            params, table = core.select_xgb_params(
                bundle,
                split,
                seed=tuning_seed,
            )

        selected[
            key
        ] = params

        table.to_csv(
            output_dir
            / f"{key}_xgb_tuning.csv",
            index=False,
        )

        print(
            "Selected:",
            params,
        )

    params_path.write_text(
        json.dumps(
            selected,
            indent=2,
            sort_keys=True,
        )
    )

    return selected


def _append_checkpoint(
    df: pd.DataFrame,
    path: Path,
    *,
    subset,
) -> None:
    if df is None or df.empty:
        return

    if path.exists():
        old = pd.read_csv(
            path
        )

        df = pd.concat(
            [
                old,
                df,
            ],
            ignore_index=True,
        )

    df = df.drop_duplicates(
        subset=subset,
        keep="last",
    )

    df.to_csv(
        path,
        index=False,
    )


def classify_mechanism(
    mechanism: pd.DataFrame,
    *,
    near_global_support: float = 0.95,
) -> pd.DataFrame:
    d = mechanism.copy()

    d[
        "near_global_correction"
    ] = (
        d[
            "test_support"
        ]
        >= near_global_support
    )

    d[
        "residual_shrunk"
    ] = (
        d[
            "base_absolute_residual_reduction"
        ]
        > 0
    )

    d[
        "term_helped_global_test_ll"
    ] = (
        d[
            "global_term_ablation_gain"
        ]
        > 0
    )

    if "placebo_gain_q95" in d.columns:
        d[
            "beats_placebo_q95"
        ] = (
            d[
                "global_term_ablation_gain"
            ]
            > d[
                "placebo_gain_q95"
            ]
        )
    else:
        d[
            "beats_placebo_q95"
        ] = (
            d[
                "placebo_empirical_p"
            ]
            < 0.05
        )

    d[
        "localized_mechanistic_success"
    ] = (
        ~d[
            "near_global_correction"
        ]
        & d[
            "residual_shrunk"
        ]
        & d[
            "term_helped_global_test_ll"
        ]
        & d[
            "beats_placebo_q95"
        ]
    )

    d[
        "global_shift_mechanistic_success"
    ] = (
        d[
            "near_global_correction"
        ]
        & d[
            "residual_shrunk"
        ]
        & d[
            "term_helped_global_test_ll"
        ]
    )

    d[
        "mechanism_category"
    ] = np.select(
        [
            d[
                "localized_mechanistic_success"
            ],
            d[
                "global_shift_mechanistic_success"
            ],
            (
                d[
                    "residual_shrunk"
                ]
                & d[
                    "term_helped_global_test_ll"
                ]
            ),
            d[
                "residual_shrunk"
            ],
        ],
        [
            "localized correction supported",
            "near-global recalibration supported",
            (
                "predictive correction supported; "
                "localization not above placebo q95"
            ),
            (
                "residual reduced but no positive "
                "term-level TEST likelihood contribution"
            ),
        ],
        default=(
            "accepted correction not supported "
            "on TEST"
        ),
    )

    return d


def summarise_mechanism(
    mechanism: pd.DataFrame,
) -> pd.DataFrame:
    d = classify_mechanism(
        mechanism
    )

    rows = []

    for dataset, g in d.groupby(
        "dataset"
    ):
        localized = g[
            ~g[
                "near_global_correction"
            ]
        ]

        rows.append(
            {
                "dataset": dataset,
                "accepted_terms_audited": len(
                    g
                ),
                "accepted_seeds_audited": g[
                    "seed"
                ].nunique(),
                "mean_test_support": g[
                    "test_support"
                ].mean(),
                "near_global_term_rate": g[
                    "near_global_correction"
                ].mean(),
                "residual_shrink_rate": g[
                    "residual_shrunk"
                ].mean(),
                "positive_term_ablation_rate": g[
                    "term_helped_global_test_ll"
                ].mean(),
                "localized_terms": len(
                    localized
                ),
                "localized_placebo_q95_win_rate": (
                    localized[
                        "beats_placebo_q95"
                    ].mean()
                    if len(
                        localized
                    )
                    else np.nan
                ),
                "localized_mechanistic_success_rate": (
                    localized[
                        "localized_mechanistic_success"
                    ].mean()
                    if len(
                        localized
                    )
                    else np.nan
                ),
                "supported_localized_terms": g[
                    "localized_mechanistic_success"
                ].sum(),
                "supported_global_shift_terms": g[
                    "global_shift_mechanistic_success"
                ].sum(),
                "mean_abs_residual_reduction": g[
                    "base_absolute_residual_reduction"
                ].mean(),
                "mean_term_ablation_gain": g[
                    "global_term_ablation_gain"
                ].mean(),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "dataset"
        )
        .reset_index(
            drop=True
        )
    )


def run_all_realdata(
    datasets: Mapping[str, core.DatasetBundle],
    selected_params: Mapping[str, dict],
    output_dir: Path,
    *,
    overwrite: bool = False,
    n_placebo: int = 500,
) -> Dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "realdata_raw.csv"
    )

    rules_path = (
        output_dir
        / "realdata_rules.csv"
    )

    mechanism_path = (
        output_dir
        / "realdata_mechanism_terms_raw.csv"
    )

    if overwrite:
        for p in [
            raw_path,
            rules_path,
            mechanism_path,
        ]:
            if p.exists():
                p.unlink()

    completed = set()

    if raw_path.exists():
        old = pd.read_csv(
            raw_path
        )

        repaired = old[
            old[
                "method"
            ].eq(
                "Tuned XGBoost + Structured Scan"
            )
        ]

        completed = {
            (
                str(
                    r.dataset
                ),
                int(
                    r.seed
                ),
            )
            for r in repaired.itertuples()
        }

    plan = [
        (
            key,
            int(
                seed
            ),
        )
        for key, cfg
        in DATASET_CONFIG.items()
        for seed in cfg[
            "seeds"
        ]
    ]

    todo = [
        (
            key,
            seed,
        )
        for key, seed
        in plan
        if (
            datasets[
                key
            ].name,
            seed,
        )
        not in completed
    ]

    print(
        f"Total conditions={len(plan)} | "
        f"complete={len(completed)} | "
        f"remaining={len(todo)}"
    )

    for i, (
        key,
        seed,
    ) in enumerate(
        todo,
        start=1,
    ):
        bundle = datasets[
            key
        ]

        print(
            f"\n[{i:02d}/{len(todo):02d}] "
            f"{bundle.name} | seed={seed}"
        )

        t0 = time.perf_counter()

        raw, rules, mechanism = (
            ext.run_seed_with_mechanism(
                bundle,
                selected_params[
                    key
                ],
                seed=seed,
                n_placebo=n_placebo,
            )
        )

        raw[
            "condition_wall_seconds"
        ] = (
            time.perf_counter()
            - t0
        )

        _append_checkpoint(
            raw,
            raw_path,
            subset=[
                "dataset",
                "seed",
                "method",
            ],
        )

        if rules is not None and not rules.empty:
            _append_checkpoint(
                rules,
                rules_path,
                subset=[
                    "dataset",
                    "seed",
                    "term",
                ],
            )

        if (
            mechanism is not None
            and not mechanism.empty
        ):
            _append_checkpoint(
                mechanism,
                mechanism_path,
                subset=[
                    "dataset",
                    "seed",
                    "term",
                ],
            )

    raw = pd.read_csv(
        raw_path
    )

    matched = core.build_realdata_matched(
        raw
    )

    summary = core.summarise_realdata(
        matched
    )

    matched.to_csv(
        output_dir
        / "realdata_matched.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "realdata_summary.csv",
        index=False,
    )

    if mechanism_path.exists():
        mechanism_raw = pd.read_csv(
            mechanism_path
        )

        mechanism = classify_mechanism(
            mechanism_raw
        )

        mechanism_summary = summarise_mechanism(
            mechanism_raw
        )

        recurrence = ext.feature_recurrence(
            mechanism
        )

        pairwise_stability = (
            ext.fixed_test_rule_stability(
                mechanism
            )
        )

        stability_summary = (
            ext.summarise_fixed_test_stability(
                pairwise_stability
            )
        )

        mechanism.to_csv(
            output_dir
            / "realdata_mechanism_terms.csv",
            index=False,
        )

        mechanism_summary.to_csv(
            output_dir
            / "realdata_mechanism_summary.csv",
            index=False,
        )

        recurrence.to_csv(
            output_dir
            / "realdata_feature_recurrence.csv",
            index=False,
        )

        pairwise_stability.to_csv(
            output_dir
            / "realdata_rule_mask_jaccard_pairs.csv",
            index=False,
        )

        stability_summary.to_csv(
            output_dir
            / "realdata_rule_stability_summary.csv",
            index=False,
        )

    else:
        mechanism = pd.DataFrame()
        mechanism_summary = pd.DataFrame()
        recurrence = pd.DataFrame()
        stability_summary = pd.DataFrame()

    return {
        "raw": raw,
        "matched": matched,
        "summary": summary,
        "mechanism": mechanism,
        "mechanism_summary": mechanism_summary,
        "feature_recurrence": recurrence,
        "rule_stability_summary": stability_summary,
    }
