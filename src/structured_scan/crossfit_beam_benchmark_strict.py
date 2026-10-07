
"""
Cross-fitted benchmark using StructuredBeamMDSS as the scan oracle.

This is the same discovery -> independent selection/refit -> test protocol as
crossfit_repair_benchmark.py, but replaces the initial coordinate-ascent scan
with a structured beam search over threshold literals.

The goal is to test the hard regime:
    a strong shallow-tree base model has already removed much of the low-order
    signal, leaving a concentrated higher-order residual.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import time

import numpy as np
import pandas as pd

from mdss_treescan_stage1 import SyntheticConfig, generate_structured_residual_data, sigmoid
from benchmark_treescan import fit_fixed_budget_xgb, subgroup_diagnostics
from treescan_prototype import binary_metrics
from crossfit_repair_benchmark_strict import (
    CrossfitResidualLearner,
    four_way_split,
)
from structured_mdss import StructuredBeamMDSS


@dataclass(frozen=True)
class BeamCondition:
    n: int = 12_000
    p: int = 30
    k: int = 5
    s: float = 0.08
    delta: float = float(np.log(4.0))
    pi: float = 0.10
    tree_depth: int = 2
    base_tree_rounds: int = 70
    repair_rounds: int = 4
    beam_width: int = 25
    seed: int = 123


def evaluate(method, y, p, subgroup, runtime, **extra):
    row = {
        "method": method,
        **binary_metrics(y, p),
        **subgroup_diagnostics(y, p, subgroup),
        "runtime_seconds": float(runtime),
    }
    row.update(extra)
    return row


def make_learner(
    *,
    mode: str,
    cond: BeamCondition,
    discovery_n: int,
    selection_n: int,
    verbose: bool = False,
) -> CrossfitResidualLearner:
    learner = CrossfitResidualLearner(
        mode=mode,
        repair_rounds=cond.repair_rounds,
        tree_max_depth=cond.tree_depth,
        coefficient_l2=1e-3,
        selection_gain_tolerance=1.9207,
        # Placeholder MDSS settings; scanner is replaced below.
        mdss_n_bins=5,
        mdss_n_restarts=1,
        mdss_max_passes=1,
        mdss_min_size=max(35, int(round(0.015 * discovery_n))),
        mdss_min_events=3,
        mdss_min_nonevents=3,
        mdss_max_rule_features=max(2, min(cond.k, 6)),
        selection_scan_min_size=max(20, int(round(0.01 * selection_n))),
        selection_scan_min_events=2,
        selection_scan_min_nonevents=2,
        seed=cond.seed,
        verbose=verbose,
    )

    learner.scanner = StructuredBeamMDSS(
        n_bins=5,
        max_rule_features=max(2, min(cond.k, 6)),
        beam_width=cond.beam_width,
        literal_mode="tails",
        min_size=max(35, int(round(0.015 * discovery_n))),
        min_events=3,
        min_nonevents=3,
    )
    return learner


def run_beam_condition(
    cond: BeamCondition,
    *,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, CrossfitResidualLearner]]:
    cfg = SyntheticConfig(
        n=cond.n,
        p=cond.p,
        interaction_order=cond.k,
        subgroup_prevalence=cond.s,
        subgroup_log_odds=cond.delta,
        event_prevalence=cond.pi,
        seed=cond.seed,
    )
    data = generate_structured_residual_data(cfg)

    split = four_way_split(
        cond.n,
        train_fraction=0.50,
        discovery_fraction=0.20,
        selection_fraction=0.15,
        seed=cond.seed + 1000,
    )

    Xtr, ytr = data.X[split.train], data.y[split.train]
    Xd, yd = data.X[split.discovery], data.y[split.discovery]
    Xs, ys = data.X[split.selection], data.y[split.selection]
    Xt, yt = data.X[split.test], data.y[split.test]
    st = data.subgroup_mask[split.test]

    t0 = time.perf_counter()
    base = fit_fixed_budget_xgb(
        Xtr,
        ytr,
        max_depth=cond.tree_depth,
        rounds=cond.base_tree_rounds,
        learning_rate=0.08,
        seed=cond.seed,
    )
    base_time = time.perf_counter() - t0

    zd, zs, zt = base.logits(Xd), base.logits(Xs), base.logits(Xt)

    rows = [
        evaluate(
            "Base XGBoost",
            yt,
            sigmoid(zt),
            st,
            base_time,
            repair_terms=0,
            scan_terms=0,
            tree_repair_terms=0,
        )
    ]

    learners = {}

    for label, mode in [
        ("Crossfit extra trees", "tree"),
        ("Crossfit structured scan", "scan"),
        ("Crossfit competitive Tree-Scan", "competitive"),
    ]:
        learner = make_learner(
            mode=mode,
            cond=cond,
            discovery_n=len(Xd),
            selection_n=len(Xs),
            verbose=verbose,
        )

        t0 = time.perf_counter()
        learner.fit(Xd, yd, zd, Xs, ys, zs)
        repair_time = time.perf_counter() - t0

        z = learner.repair_logits(Xt, zt)
        n_scan = sum(r.chosen == "scan" for r in learner.history_)
        n_tree = sum(r.chosen == "tree" for r in learner.history_)

        rows.append(
            evaluate(
                label,
                yt,
                sigmoid(z),
                st,
                base_time + repair_time,
                repair_terms=len(learner.history_),
                scan_terms=n_scan,
                tree_repair_terms=n_tree,
            )
        )
        learners[label] = learner

    df = pd.DataFrame(rows)
    for k, v in asdict(cond).items():
        df[k] = v
    df["k_minus_d"] = cond.k - cond.tree_depth
    df["realized_event_prevalence"] = float(data.y.mean())
    df["realized_subgroup_prevalence"] = float(data.subgroup_mask.mean())

    return df, learners


def smoke_beam_experiment():
    frames = []

    for k in (2, 5):
        for seed in (1101, 2202):
            print(f"beam smoke: k={k}, depth=2, seed={seed}")
            cond = BeamCondition(
                n=12_000,
                p=30,
                k=k,
                s=0.08,
                delta=float(np.log(4.0)),
                pi=0.10,
                tree_depth=2,
                base_tree_rounds=70,
                repair_rounds=3,
                beam_width=20,
                seed=seed,
            )
            df, _ = run_beam_condition(cond, verbose=False)
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    keys = ["k", "tree_depth", "seed"]
    base = (
        out[out.method == "Base XGBoost"][keys + ["logloss"]]
        .rename(columns={"logloss": "base_logloss"})
    )
    tmp = out.merge(base, on=keys, how="left")
    tmp["logloss_advantage_vs_base"] = tmp["base_logloss"] - tmp["logloss"]

    summary = (
        tmp.groupby(["method", "k_minus_d"], as_index=False)
        .agg(
            mean_logloss=("logloss", "mean"),
            mean_logloss_advantage_vs_base=("logloss_advantage_vs_base", "mean"),
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_true_subgroup_residual=("true_subgroup_mean_residual", "mean"),
            mean_scan_terms=("scan_terms", "mean"),
            mean_tree_repair_terms=("tree_repair_terms", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            runs=("logloss", "size"),
        )
    )

    out.to_csv("/mnt/data/beam_crossfit_smoke_results.csv", index=False)
    summary.to_csv("/mnt/data/beam_crossfit_smoke_summary.csv", index=False)

    return out, summary


if __name__ == "__main__":
    results, summary = smoke_beam_experiment()
    print(summary.to_string(index=False))
