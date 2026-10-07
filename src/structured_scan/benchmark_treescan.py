
"""
Benchmark harness for the Tree–Scan project.

Purpose
-------
Test the central mechanism under controlled synthetic conditions:

    Does the proposed competitive Tree–Scan learner become more useful as the
    planted interaction order k exceeds the allowed tree depth d?

Methods compared
----------------
1. XGBoost (fixed additive-update budget)
2. XGBoost + one-shot MDSS correction
3. XGBoost + iterative MDSS corrections
4. Competitive fully-corrective Tree–Scan

This is the FIRST controlled benchmark harness. It deliberately keeps the
protocol transparent. A later paper-grade version should add repeated nested
cross-fitting and external benchmark algorithms (MCGrad, Optimal Rule Boosting).

The primary metric is test log-loss because the theory is expressed in exact
logistic-loss reduction. AUROC, AUPRC and Brier score are reported as secondary
metrics.

Important scientific distinction
---------------------------------
"Budget" here means the maximum number of additive update rounds available to
the tree learner. It is not yet equal wall-clock time. Runtime is recorded so
we can later build a true compute-normalised experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from mdss_treescan_stage1 import (
    BinnedMDSS,
    SyntheticConfig,
    exact_subgroup_logit_correction,
    generate_structured_residual_data,
    logistic_loss_from_logits,
    sigmoid,
)
from treescan_prototype import (
    CompetitiveTreeScan,
    ScanBasis,
    binary_metrics,
)


# ---------------------------------------------------------------------
# Reproducible data splitting
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


def three_way_split(
    n: int,
    *,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    seed: int = 123,
) -> Split:
    """
    Train:
        fit the base learner / Tree–Scan.
    Calibration:
        estimate MDSS post-processing corrections for XGBoost baselines.
    Test:
        untouched final comparison.
    """
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("Need non-zero test fraction.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(round(train_fraction * n))
    n_cal = int(round(calibration_fraction * n))

    return Split(
        train=idx[:n_train],
        calibration=idx[n_train:n_train + n_cal],
        test=idx[n_train + n_cal:],
    )


# ---------------------------------------------------------------------
# Fixed-budget XGBoost
# ---------------------------------------------------------------------

@dataclass
class XGBModel:
    booster: xgb.Booster

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(
            xgb.DMatrix(np.asarray(X, dtype=float)),
            output_margin=True,
        ).astype(float)

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(X))


def fit_fixed_budget_xgb(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int,
    rounds: int,
    learning_rate: float = 0.10,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    seed: int = 123,
) -> XGBModel:
    dtrain = xgb.DMatrix(np.asarray(X, dtype=float), label=np.asarray(y, dtype=int))

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(max_depth),
        "eta": float(learning_rate),
        "lambda": float(reg_lambda),
        "min_child_weight": float(min_child_weight),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "seed": int(seed),
        "verbosity": 0,
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(rounds),
    )
    return XGBModel(booster)


# ---------------------------------------------------------------------
# Scan-rule application helpers
# ---------------------------------------------------------------------

@dataclass
class FittedScanRule:
    basis: ScanBasis
    delta: float
    calibration_gain: float
    calibration_size: int

    def apply(self, X: np.ndarray, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=float) + self.delta * self.basis.transform(X)


def fit_scan_rule_on_calibration(
    scanner: BinnedMDSS,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    logits_cal: np.ndarray,
) -> Tuple[FittedScanRule, object]:
    """
    Discover and calibrate one rule on an independent calibration split.
    """
    result = scanner.scan(X_cal, y_cal, logits_cal)

    basis = ScanBasis(
        restrictions=dict(result.restrictions),
        binner=scanner.binner_,
    )

    rule = FittedScanRule(
        basis=basis,
        delta=float(result.delta),
        calibration_gain=float(result.gain),
        calibration_size=int(result.score.size),
    )
    return rule, result


def fit_iterative_scan_corrections(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    base_logits_cal: np.ndarray,
    *,
    max_rules: int = 6,
    gain_tolerance: float = 0.5,
    n_bins: int = 5,
    n_restarts: int = 6,
    max_passes: int = 6,
    min_size: int = 40,
    min_events: int = 3,
    min_nonevents: int = 3,
    max_rule_features: int = 6,
    seed: int = 123,
) -> Tuple[List[FittedScanRule], np.ndarray]:
    """
    Sequential exact MDSS corrections fitted on the calibration set.

    The binner is fitted once and then held fixed across all rounds.
    """
    scanner = BinnedMDSS(
        n_bins=n_bins,
        n_restarts=n_restarts,
        max_passes=max_passes,
        min_size=min_size,
        min_events=min_events,
        min_nonevents=min_nonevents,
        max_rule_features=max_rule_features,
        seed=seed,
    )
    scanner.fit_binner(X_cal)

    logits = np.asarray(base_logits_cal, dtype=float).copy()
    rules: List[FittedScanRule] = []

    for t in range(max_rules):
        result = scanner.scan(X_cal, y_cal, logits)

        if result.gain < gain_tolerance:
            break

        basis = ScanBasis(
            restrictions=dict(result.restrictions),
            binner=scanner.binner_,
        )
        rule = FittedScanRule(
            basis=basis,
            delta=float(result.delta),
            calibration_gain=float(result.gain),
            calibration_size=int(result.score.size),
        )

        logits = rule.apply(X_cal, logits)
        rules.append(rule)

    return rules, logits


def apply_scan_rules(
    rules: Sequence[FittedScanRule],
    X: np.ndarray,
    base_logits: np.ndarray,
) -> np.ndarray:
    logits = np.asarray(base_logits, dtype=float).copy()
    for rule in rules:
        logits = rule.apply(X, logits)
    return logits


# ---------------------------------------------------------------------
# Audit metrics specific to the planted subgroup
# ---------------------------------------------------------------------

def subgroup_diagnostics(
    y: np.ndarray,
    p: np.ndarray,
    true_subgroup: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    s = np.asarray(true_subgroup, dtype=bool)

    if s.sum() == 0:
        return {
            "true_subgroup_n": 0,
            "true_subgroup_mean_residual": np.nan,
            "true_subgroup_logloss": np.nan,
        }

    z = np.log(np.clip(p, 1e-12, 1 - 1e-12)) - np.log1p(-np.clip(p, 1e-12, 1 - 1e-12))

    return {
        "true_subgroup_n": int(s.sum()),
        "true_subgroup_mean_residual": float(np.mean(y[s] - p[s])),
        "true_subgroup_logloss": float(
            logistic_loss_from_logits(y[s], z[s]) / s.sum()
        ),
    }


def evaluate_method(
    name: str,
    y_test: np.ndarray,
    p_test: np.ndarray,
    subgroup_test: np.ndarray,
    *,
    runtime_seconds: float,
    extra: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": name,
        **binary_metrics(y_test, p_test),
        **subgroup_diagnostics(y_test, p_test, subgroup_test),
        "runtime_seconds": float(runtime_seconds),
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------
# One complete benchmark condition
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkCondition:
    n: int = 12_000
    p: int = 30
    k: int = 5
    s: float = 0.08
    delta: float = float(np.log(4.0))
    pi: float = 0.10
    tree_depth: int = 2
    update_budget: int = 20
    seed: int = 123


def run_condition(
    condition: BenchmarkCondition,
    *,
    mdss_n_bins: int = 5,
    mdss_n_restarts: int = 4,
    mdss_max_passes: int = 5,
    mdss_max_rule_features: Optional[int] = None,
    iterative_scan_rules: int = 5,
    verbose_treescan: bool = False,
) -> pd.DataFrame:
    """
    Run all four methods on EXACTLY the same generated data and test split.
    """
    if mdss_max_rule_features is None:
        # Let the scan family be rich enough to represent the planted rule,
        # while still preventing arbitrary 30-dimensional overfitting.
        mdss_max_rule_features = max(2, min(condition.k + 1, 8))

    cfg = SyntheticConfig(
        n=condition.n,
        p=condition.p,
        interaction_order=condition.k,
        subgroup_prevalence=condition.s,
        subgroup_log_odds=condition.delta,
        event_prevalence=condition.pi,
        seed=condition.seed,
    )
    data = generate_structured_residual_data(cfg)
    split = three_way_split(
        condition.n,
        train_fraction=0.60,
        calibration_fraction=0.20,
        seed=condition.seed + 1000,
    )

    Xtr, ytr = data.X[split.train], data.y[split.train]
    Xcal, ycal = data.X[split.calibration], data.y[split.calibration]
    Xte, yte = data.X[split.test], data.y[split.test]
    ste = data.subgroup_mask[split.test]

    results: List[Dict[str, object]] = []

    # -------------------------------------------------------------
    # 1. Fixed-budget XGBoost
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    xgb_model = fit_fixed_budget_xgb(
        Xtr,
        ytr,
        max_depth=condition.tree_depth,
        rounds=condition.update_budget,
        learning_rate=0.10,
        seed=condition.seed,
    )
    xgb_runtime = time.perf_counter() - t0

    logits_cal = xgb_model.logits(Xcal)
    logits_test = xgb_model.logits(Xte)
    p_test = sigmoid(logits_test)

    results.append(
        evaluate_method(
            "XGBoost",
            yte,
            p_test,
            ste,
            runtime_seconds=xgb_runtime,
            extra={
                "n_scan_rules": 0,
                "n_tree_updates": condition.update_budget,
            },
        )
    )

    # -------------------------------------------------------------
    # 2. XGBoost + one-shot MDSS
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    scanner = BinnedMDSS(
        n_bins=mdss_n_bins,
        n_restarts=mdss_n_restarts,
        max_passes=mdss_max_passes,
        min_size=max(30, int(round(0.01 * len(Xcal)))),
        min_events=3,
        min_nonevents=3,
        max_rule_features=mdss_max_rule_features,
        seed=condition.seed,
    )
    scanner.fit_binner(Xcal)

    one_rule, one_result = fit_scan_rule_on_calibration(
        scanner,
        Xcal,
        ycal,
        logits_cal,
    )
    one_logits_test = one_rule.apply(Xte, logits_test)
    one_runtime = xgb_runtime + (time.perf_counter() - t0)

    results.append(
        evaluate_method(
            "XGBoost + one-shot MDSS",
            yte,
            sigmoid(one_logits_test),
            ste,
            runtime_seconds=one_runtime,
            extra={
                "n_scan_rules": 1,
                "n_tree_updates": condition.update_budget,
                "last_scan_gain": one_rule.calibration_gain,
                "last_scan_delta": one_rule.delta,
            },
        )
    )

    # -------------------------------------------------------------
    # 3. XGBoost + iterative MDSS
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    rules, _ = fit_iterative_scan_corrections(
        Xcal,
        ycal,
        logits_cal,
        max_rules=iterative_scan_rules,
        gain_tolerance=0.5,
        n_bins=mdss_n_bins,
        n_restarts=mdss_n_restarts,
        max_passes=mdss_max_passes,
        min_size=max(30, int(round(0.01 * len(Xcal)))),
        min_events=3,
        min_nonevents=3,
        max_rule_features=mdss_max_rule_features,
        seed=condition.seed,
    )
    iter_logits_test = apply_scan_rules(rules, Xte, logits_test)
    iter_runtime = xgb_runtime + (time.perf_counter() - t0)

    results.append(
        evaluate_method(
            "XGBoost + iterative MDSS",
            yte,
            sigmoid(iter_logits_test),
            ste,
            runtime_seconds=iter_runtime,
            extra={
                "n_scan_rules": len(rules),
                "n_tree_updates": condition.update_budget,
                "last_scan_gain": (
                    rules[-1].calibration_gain if rules else np.nan
                ),
                "last_scan_delta": (
                    rules[-1].delta if rules else np.nan
                ),
            },
        )
    )

    # -------------------------------------------------------------
    # 4. Competitive Tree–Scan
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    ts = CompetitiveTreeScan(
        max_rounds=condition.update_budget,
        tree_max_depth=condition.tree_depth,
        coefficient_l2=1e-4,
        gain_tolerance=0.5,
        mdss_n_bins=mdss_n_bins,
        mdss_n_restarts=mdss_n_restarts,
        mdss_max_passes=mdss_max_passes,
        mdss_min_size=max(30, int(round(0.01 * len(Xtr)))),
        mdss_min_events=3,
        mdss_min_nonevents=3,
        mdss_max_rule_features=mdss_max_rule_features,
        # No validation early stopping in the equal-update-budget experiment.
        # This keeps the allowed number of additive update opportunities explicit.
        validation_patience=condition.update_budget + 1,
        seed=condition.seed,
        verbose=verbose_treescan,
    )
    ts.fit(Xtr, ytr)
    ts_runtime = time.perf_counter() - t0

    p_ts = ts.predict_proba(Xte)
    n_scan = sum(h.chosen == "scan" for h in ts.history_)
    n_tree = sum(h.chosen == "tree" for h in ts.history_)

    results.append(
        evaluate_method(
            "Competitive Tree-Scan",
            yte,
            p_ts,
            ste,
            runtime_seconds=ts_runtime,
            extra={
                "n_scan_rules": n_scan,
                "n_tree_updates": n_tree,
                "actual_updates": len(ts.history_),
            },
        )
    )

    df = pd.DataFrame(results)

    # Attach the condition to every row for easy aggregation.
    for key, value in asdict(condition).items():
        df[key] = value
    df["k_minus_d"] = condition.k - condition.tree_depth
    df["realized_event_prevalence"] = float(data.y.mean())
    df["realized_subgroup_prevalence"] = float(data.subgroup_mask.mean())

    return df


# ---------------------------------------------------------------------
# k-d mechanism experiment
# ---------------------------------------------------------------------

def run_k_minus_d_experiment(
    *,
    k_values: Sequence[int] = (1, 2, 3, 4, 5, 6),
    depth_values: Sequence[int] = (1, 2, 3),
    seeds: Sequence[int] = (101, 202, 303),
    n: int = 12_000,
    p: int = 30,
    s: float = 0.08,
    delta: float = float(np.log(4.0)),
    pi: float = 0.10,
    update_budget: int = 20,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    total = len(k_values) * len(depth_values) * len(seeds)
    counter = 0

    for k in k_values:
        for depth in depth_values:
            for seed in seeds:
                counter += 1
                print(
                    f"[{counter:03d}/{total:03d}] "
                    f"k={k}, depth={depth}, k-d={k-depth}, seed={seed}"
                )

                condition = BenchmarkCondition(
                    n=n,
                    p=p,
                    k=k,
                    s=s,
                    delta=delta,
                    pi=pi,
                    tree_depth=depth,
                    update_budget=update_budget,
                    seed=seed,
                )
                frames.append(run_condition(condition))

    out = pd.concat(frames, ignore_index=True)

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)

    return out


def summarise_k_minus_d(results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise performance by k-d and method.

    Positive logloss_advantage_vs_xgb means the method has LOWER test log-loss
    than the matched XGBoost run.
    """
    keys = ["k", "tree_depth", "seed"]

    xgb = (
        results.loc[results["method"] == "XGBoost", keys + ["logloss"]]
        .rename(columns={"logloss": "xgb_logloss"})
    )

    merged = results.merge(xgb, on=keys, how="left")
    merged["logloss_advantage_vs_xgb"] = (
        merged["xgb_logloss"] - merged["logloss"]
    )

    summary = (
        merged
        .groupby(["method", "k_minus_d"], as_index=False)
        .agg(
            mean_logloss=("logloss", "mean"),
            sd_logloss=("logloss", "std"),
            mean_logloss_advantage_vs_xgb=(
                "logloss_advantage_vs_xgb", "mean"
            ),
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_true_subgroup_residual=(
                "true_subgroup_mean_residual", "mean"
            ),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            runs=("logloss", "size"),
        )
    )
    return summary


# ---------------------------------------------------------------------
# Small smoke experiment for correctness, not publication
# ---------------------------------------------------------------------

def smoke_experiment() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tiny grid chosen to complete quickly.

    This is a software/mechanism smoke test, NOT an inferential result.
    """
    results = run_k_minus_d_experiment(
        k_values=(2, 5),
        depth_values=(2,),
        seeds=(1101, 2202),
        n=7_500,
        p=30,
        s=0.08,
        delta=float(np.log(4.0)),
        pi=0.10,
        update_budget=10,
        output_csv="/mnt/data/treescan_smoke_results.csv",
    )
    summary = summarise_k_minus_d(results)
    summary.to_csv("/mnt/data/treescan_smoke_summary.csv", index=False)
    return results, summary


if __name__ == "__main__":
    results, summary = smoke_experiment()
    print("\n=== Smoke-test summary ===")
    print(summary.to_string(index=False))
