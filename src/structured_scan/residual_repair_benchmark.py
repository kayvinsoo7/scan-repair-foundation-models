
"""
Residual-repair benchmark for the Tree–Scan project.

This experiment isolates the mechanism more cleanly than the first benchmark:

Stage 1:
    Train a reasonably strong shallow-tree XGBoost base model on TRAIN data.

Stage 2:
    Freeze that base model.

Stage 3:
    On an independent CALIBRATION/REPAIR split, compare four ways to spend an
    additional repair budget:
        A. no repair
        B. extra shallow XGBoost residual trees
        C. iterative MDSS corrections
        D. competitive residual Tree–Scan:
              at each repair round, propose one shallow residual tree and one
              MDSS subgroup; score both by exact logistic-loss reduction; add
              the better one; fully refit all residual coefficients while the
              base model remains a fixed offset.

Stage 4:
    Evaluate once on untouched TEST data.

This directly tests the locked narrative:
    trees learn broad structure first; scans should help when concentrated,
    multidimensional residual structure remains.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

from mdss_treescan_stage1 import (
    BinnedMDSS,
    SyntheticConfig,
    generate_structured_residual_data,
    logistic_loss_from_logits,
    sigmoid,
)
from treescan_prototype import (
    DirectionScore,
    ScanBasis,
    TreeBasis,
    binary_metrics,
    exact_direction_gain,
    propose_xgb_tree,
)
from benchmark_treescan import (
    BenchmarkCondition,
    fit_fixed_budget_xgb,
    subgroup_diagnostics,
    three_way_split,
)


# ---------------------------------------------------------------------
# Penalised logistic refit with a FIXED base-model offset
# ---------------------------------------------------------------------

@dataclass
class OffsetCoefficientFit:
    coefficients: np.ndarray
    loss: float
    success: bool


def fit_residual_dictionary_with_offset(
    y: np.ndarray,
    base_logits: np.ndarray,
    Phi: np.ndarray,
    *,
    l2: float = 1e-4,
    init: Optional[np.ndarray] = None,
) -> OffsetCoefficientFit:
    """
    Minimise:
        sum_i logistic_loss(y_i, base_logit_i + Phi_i @ beta)
        + l2/2 ||beta||^2

    The base predictor is frozen. Only repair-basis coefficients are refitted.
    """
    y = np.asarray(y, dtype=float)
    offset = np.asarray(base_logits, dtype=float)
    Phi = np.asarray(Phi, dtype=float)

    q = Phi.shape[1]
    if q == 0:
        return OffsetCoefficientFit(
            coefficients=np.empty(0),
            loss=logistic_loss_from_logits(y, offset),
            success=True,
        )

    if init is None or len(init) != q:
        x0 = np.zeros(q, dtype=float)
    else:
        x0 = np.asarray(init, dtype=float).copy()

    def objective(beta: np.ndarray):
        logits = offset + Phi @ beta
        p = sigmoid(logits)

        loss = (
            logistic_loss_from_logits(y, logits)
            + 0.5 * l2 * float(beta @ beta)
        )

        residual = p - y
        grad = Phi.T @ residual + l2 * beta
        return loss, grad

    res = minimize(
        lambda b: objective(b),
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )

    return OffsetCoefficientFit(
        coefficients=np.asarray(res.x, dtype=float),
        loss=float(res.fun),
        success=bool(res.success),
    )


# ---------------------------------------------------------------------
# Competitive residual repair learner
# ---------------------------------------------------------------------

Basis = Union[TreeBasis, ScanBasis]


@dataclass
class RepairRound:
    round_index: int
    tree_gain: float
    scan_gain: float
    chosen: str
    scan_delta: float
    tree_alpha: float
    active_repair_terms: int
    repair_loss_after_refit: float


class CompetitiveResidualTreeScan:
    """
    Frozen base predictor + competitive fully-corrective residual repairs.
    """

    def __init__(
        self,
        *,
        repair_rounds: int = 6,
        tree_max_depth: int = 2,
        coefficient_l2: float = 1e-4,
        gain_tolerance: float = 0.5,
        mdss_n_bins: int = 5,
        mdss_n_restarts: int = 6,
        mdss_max_passes: int = 6,
        mdss_min_size: int = 40,
        mdss_min_events: int = 3,
        mdss_min_nonevents: int = 3,
        mdss_max_rule_features: int = 6,
        seed: int = 123,
        verbose: bool = False,
    ) -> None:
        self.repair_rounds = int(repair_rounds)
        self.tree_max_depth = int(tree_max_depth)
        self.coefficient_l2 = float(coefficient_l2)
        self.gain_tolerance = float(gain_tolerance)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.scanner = BinnedMDSS(
            n_bins=mdss_n_bins,
            n_restarts=mdss_n_restarts,
            max_passes=mdss_max_passes,
            min_size=mdss_min_size,
            min_events=mdss_min_events,
            min_nonevents=mdss_min_nonevents,
            max_rule_features=mdss_max_rule_features,
            seed=seed,
        )

        self.bases_: List[Basis] = []
        self.coefficients_: np.ndarray = np.empty(0)
        self.history_: List[RepairRound] = []

    def _Phi(self, X: np.ndarray) -> np.ndarray:
        if not self.bases_:
            return np.empty((len(X), 0), dtype=float)
        return np.column_stack([b.transform(X) for b in self.bases_])

    def repair_logits(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
    ) -> np.ndarray:
        base_logits = np.asarray(base_logits, dtype=float)
        if not self.bases_:
            return base_logits.copy()
        return base_logits + self._Phi(X) @ self.coefficients_

    def fit(
        self,
        X_repair: np.ndarray,
        y_repair: np.ndarray,
        base_logits_repair: np.ndarray,
    ) -> "CompetitiveResidualTreeScan":
        X = np.asarray(X_repair, dtype=float)
        y = np.asarray(y_repair, dtype=int)
        offset = np.asarray(base_logits_repair, dtype=float)

        self.scanner.fit_binner(X)
        self.bases_ = []
        self.coefficients_ = np.empty(0)
        self.history_ = []

        for t in range(self.repair_rounds):
            current_logits = self.repair_logits(X, offset)

            # Candidate 1: one shallow residual tree.
            tree = propose_xgb_tree(
                X,
                y,
                current_logits,
                max_depth=self.tree_max_depth,
                seed=self.seed + t,
            )
            tree_direction = tree.transform(X)
            tree_score = exact_direction_gain(
                y,
                current_logits,
                tree_direction,
            )

            # Candidate 2: one exact-likelihood MDSS rule.
            scan = self.scanner.scan(X, y, current_logits)
            scan_basis = ScanBasis(
                restrictions=dict(scan.restrictions),
                binner=self.scanner.binner_,
            )

            if tree_score.gain > scan.gain:
                chosen = "tree"
                chosen_basis: Basis = tree
                candidate_coef = float(tree_score.alpha)
                best_gain = float(tree_score.gain)
            else:
                chosen = "scan"
                chosen_basis = scan_basis
                candidate_coef = float(scan.delta)
                best_gain = float(scan.gain)

            if self.verbose:
                print(
                    f"repair {t+1:02d} | "
                    f"tree={tree_score.gain:8.3f} | "
                    f"scan={scan.gain:8.3f} | choose={chosen}"
                )

            if best_gain < self.gain_tolerance:
                break

            self.bases_.append(chosen_basis)

            Phi = self._Phi(X)
            init = np.concatenate(
                [self.coefficients_, np.array([candidate_coef])]
            )

            fit = fit_residual_dictionary_with_offset(
                y,
                offset,
                Phi,
                l2=self.coefficient_l2,
                init=init,
            )
            self.coefficients_ = fit.coefficients

            self.history_.append(
                RepairRound(
                    round_index=t + 1,
                    tree_gain=float(tree_score.gain),
                    scan_gain=float(scan.gain),
                    chosen=chosen,
                    scan_delta=float(scan.delta),
                    tree_alpha=float(tree_score.alpha),
                    active_repair_terms=len(self.bases_),
                    repair_loss_after_refit=float(fit.loss),
                )
            )

        return self


# ---------------------------------------------------------------------
# Extra-tree-only repair baseline
# ---------------------------------------------------------------------

class ExtraTreeRepair:
    """
    Repair a frozen base predictor with shallow XGBoost trees only.

    Each repair tree is trained sequentially on the repair split using the
    current repaired logits as XGBoost base_margin. Its exact line coefficient
    is then fitted. Coefficients are not fully corrected here; this is a
    deliberately simple "spend all repair rounds on trees" comparator.
    """

    def __init__(
        self,
        *,
        repair_rounds: int = 6,
        tree_max_depth: int = 2,
        gain_tolerance: float = 0.5,
        seed: int = 123,
    ) -> None:
        self.repair_rounds = repair_rounds
        self.tree_max_depth = tree_max_depth
        self.gain_tolerance = gain_tolerance
        self.seed = seed
        self.trees_: List[TreeBasis] = []
        self.alphas_: List[float] = []
        self.gains_: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_logits: np.ndarray,
    ) -> "ExtraTreeRepair":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        logits = np.asarray(base_logits, dtype=float).copy()

        self.trees_ = []
        self.alphas_ = []
        self.gains_ = []

        for t in range(self.repair_rounds):
            tree = propose_xgb_tree(
                X,
                y,
                logits,
                max_depth=self.tree_max_depth,
                seed=self.seed + t,
            )
            direction = tree.transform(X)
            sc = exact_direction_gain(y, logits, direction)

            if sc.gain < self.gain_tolerance:
                break

            logits = logits + sc.alpha * direction
            self.trees_.append(tree)
            self.alphas_.append(float(sc.alpha))
            self.gains_.append(float(sc.gain))

        return self

    def repair_logits(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
    ) -> np.ndarray:
        logits = np.asarray(base_logits, dtype=float).copy()
        for tree, alpha in zip(self.trees_, self.alphas_):
            logits += alpha * tree.transform(X)
        return logits


# ---------------------------------------------------------------------
# Iterative scan-only repair
# ---------------------------------------------------------------------

class ScanOnlyRepair:
    def __init__(
        self,
        *,
        repair_rounds: int = 6,
        gain_tolerance: float = 0.5,
        n_bins: int = 5,
        n_restarts: int = 6,
        max_passes: int = 6,
        min_size: int = 40,
        min_events: int = 3,
        min_nonevents: int = 3,
        max_rule_features: int = 6,
        seed: int = 123,
    ) -> None:
        self.repair_rounds = repair_rounds
        self.gain_tolerance = gain_tolerance
        self.scanner = BinnedMDSS(
            n_bins=n_bins,
            n_restarts=n_restarts,
            max_passes=max_passes,
            min_size=min_size,
            min_events=min_events,
            min_nonevents=min_nonevents,
            max_rule_features=max_rule_features,
            seed=seed,
        )
        self.rules_: List[ScanBasis] = []
        self.deltas_: List[float] = []
        self.gains_: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_logits: np.ndarray,
    ) -> "ScanOnlyRepair":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        logits = np.asarray(base_logits, dtype=float).copy()

        self.scanner.fit_binner(X)
        self.rules_, self.deltas_, self.gains_ = [], [], []

        for _ in range(self.repair_rounds):
            scan = self.scanner.scan(X, y, logits)
            if scan.gain < self.gain_tolerance:
                break

            basis = ScanBasis(
                restrictions=dict(scan.restrictions),
                binner=self.scanner.binner_,
            )
            logits += scan.delta * basis.transform(X)

            self.rules_.append(basis)
            self.deltas_.append(float(scan.delta))
            self.gains_.append(float(scan.gain))

        return self

    def repair_logits(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
    ) -> np.ndarray:
        logits = np.asarray(base_logits, dtype=float).copy()
        for basis, delta in zip(self.rules_, self.deltas_):
            logits += delta * basis.transform(X)
        return logits


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(
    method: str,
    y: np.ndarray,
    p: np.ndarray,
    subgroup: np.ndarray,
    runtime: float,
    **extra,
) -> Dict[str, object]:
    row = {
        "method": method,
        **binary_metrics(y, p),
        **subgroup_diagnostics(y, p, subgroup),
        "runtime_seconds": float(runtime),
    }
    row.update(extra)
    return row


@dataclass(frozen=True)
class RepairCondition:
    n: int = 12_000
    p: int = 30
    k: int = 5
    s: float = 0.08
    delta: float = float(np.log(4.0))
    pi: float = 0.10
    tree_depth: int = 2
    base_tree_rounds: int = 80
    repair_rounds: int = 5
    seed: int = 123


def run_repair_condition(
    cond: RepairCondition,
    *,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, CompetitiveResidualTreeScan]:
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

    split = three_way_split(
        cond.n,
        train_fraction=0.60,
        calibration_fraction=0.20,
        seed=cond.seed + 1000,
    )

    Xtr, ytr = data.X[split.train], data.y[split.train]
    Xrep, yrep = data.X[split.calibration], data.y[split.calibration]
    Xte, yte = data.X[split.test], data.y[split.test]
    ste = data.subgroup_mask[split.test]

    # -----------------------------------------------------------------
    # Strong shallow-tree base model
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    base = fit_fixed_budget_xgb(
        Xtr,
        ytr,
        max_depth=cond.tree_depth,
        rounds=cond.base_tree_rounds,
        learning_rate=0.08,
        seed=cond.seed,
    )
    base_fit_time = time.perf_counter() - t0

    zrep = base.logits(Xrep)
    zte = base.logits(Xte)

    rows: List[Dict[str, object]] = []

    rows.append(
        evaluate(
            "Base XGBoost",
            yte,
            sigmoid(zte),
            ste,
            base_fit_time,
            repair_terms=0,
            scan_terms=0,
            tree_repair_terms=0,
        )
    )

    # -----------------------------------------------------------------
    # Spend all repair budget on extra shallow trees
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    tree_repair = ExtraTreeRepair(
        repair_rounds=cond.repair_rounds,
        tree_max_depth=cond.tree_depth,
        seed=cond.seed,
    ).fit(Xrep, yrep, zrep)
    elapsed = time.perf_counter() - t0

    z = tree_repair.repair_logits(Xte, zte)
    rows.append(
        evaluate(
            "Base + extra trees",
            yte,
            sigmoid(z),
            ste,
            base_fit_time + elapsed,
            repair_terms=len(tree_repair.trees_),
            scan_terms=0,
            tree_repair_terms=len(tree_repair.trees_),
        )
    )

    # -----------------------------------------------------------------
    # Spend all repair budget on scan rules
    # -----------------------------------------------------------------
    max_rule_features = max(2, min(cond.k + 1, 8))

    t0 = time.perf_counter()
    scan_repair = ScanOnlyRepair(
        repair_rounds=cond.repair_rounds,
        max_rule_features=max_rule_features,
        min_size=max(35, int(round(0.015 * len(Xrep)))),
        seed=cond.seed,
    ).fit(Xrep, yrep, zrep)
    elapsed = time.perf_counter() - t0

    z = scan_repair.repair_logits(Xte, zte)
    rows.append(
        evaluate(
            "Base + scan repair",
            yte,
            sigmoid(z),
            ste,
            base_fit_time + elapsed,
            repair_terms=len(scan_repair.rules_),
            scan_terms=len(scan_repair.rules_),
            tree_repair_terms=0,
        )
    )

    # -----------------------------------------------------------------
    # Competitive residual Tree–Scan
    # -----------------------------------------------------------------
    t0 = time.perf_counter()
    hybrid = CompetitiveResidualTreeScan(
        repair_rounds=cond.repair_rounds,
        tree_max_depth=cond.tree_depth,
        mdss_max_rule_features=max_rule_features,
        mdss_min_size=max(35, int(round(0.015 * len(Xrep)))),
        seed=cond.seed,
        verbose=verbose,
    ).fit(Xrep, yrep, zrep)
    elapsed = time.perf_counter() - t0

    z = hybrid.repair_logits(Xte, zte)
    n_scan = sum(r.chosen == "scan" for r in hybrid.history_)
    n_tree = sum(r.chosen == "tree" for r in hybrid.history_)

    rows.append(
        evaluate(
            "Competitive residual Tree-Scan",
            yte,
            sigmoid(z),
            ste,
            base_fit_time + elapsed,
            repair_terms=len(hybrid.history_),
            scan_terms=n_scan,
            tree_repair_terms=n_tree,
        )
    )

    df = pd.DataFrame(rows)
    for k, v in asdict(cond).items():
        df[k] = v
    df["k_minus_d"] = cond.k - cond.tree_depth
    df["realized_event_prevalence"] = float(data.y.mean())
    df["realized_subgroup_prevalence"] = float(data.subgroup_mask.mean())

    return df, hybrid


def smoke_repair_experiment() -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []

    for k in (2, 5):
        for seed in (1101, 2202):
            print(f"repair smoke: k={k}, depth=2, seed={seed}")
            cond = RepairCondition(
                n=9_000,
                p=30,
                k=k,
                s=0.08,
                delta=float(np.log(4.0)),
                pi=0.10,
                tree_depth=2,
                base_tree_rounds=60,
                repair_rounds=4,
                seed=seed,
            )
            df, _ = run_repair_condition(cond, verbose=False)
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Matched advantage relative to frozen base XGBoost.
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

    out.to_csv("/mnt/data/residual_repair_smoke_results.csv", index=False)
    summary.to_csv("/mnt/data/residual_repair_smoke_summary.csv", index=False)
    return out, summary


if __name__ == "__main__":
    results, summary = smoke_repair_experiment()
    print("\n=== Residual-repair smoke summary ===")
    print(summary.to_string(index=False))
