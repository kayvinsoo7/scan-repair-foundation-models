
"""
Cross-fitted residual Tree–Scan benchmark.

Why this version exists
-----------------------
The first repair benchmark showed an important failure mode:
candidate discovery and correction fitting on the same calibration sample can
reduce a local residual while worsening out-of-sample global log-loss.

This implementation separates roles:

    TRAIN       -> fit the frozen base XGBoost model
    DISCOVERY   -> generate candidate trees / MDSS subgroups
    SELECTION   -> independently score candidates and refit repair coefficients
    TEST        -> untouched final evaluation

At every repair round:
1. Generate one residual-tree candidate and one MDSS subgroup on DISCOVERY.
2. Evaluate each candidate's exact logistic-loss gain on SELECTION.
3. Add the candidate with larger *selection* gain.
4. Fully refit all active repair coefficients on SELECTION.
5. Use those coefficients to form the next current model on both splits.

This is the cleaner algorithmic test of whether a discovered structured
correction generalises.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import time

import numpy as np
import pandas as pd

from mdss_treescan_stage1 import (
    BinnedMDSS,
    SyntheticConfig,
    exact_subgroup_logit_correction,
    generate_structured_residual_data,
    sigmoid,
)
from treescan_prototype import (
    ScanBasis,
    TreeBasis,
    binary_metrics,
    exact_direction_gain,
    propose_xgb_tree,
)
from benchmark_treescan import (
    fit_fixed_budget_xgb,
    subgroup_diagnostics,
)
from residual_repair_benchmark import (
    fit_residual_dictionary_with_offset,
)


# ---------------------------------------------------------------------
# Four-way split
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FourWaySplit:
    train: np.ndarray
    discovery: np.ndarray
    selection: np.ndarray
    test: np.ndarray


def four_way_split(
    n: int,
    *,
    train_fraction: float = 0.50,
    discovery_fraction: float = 0.20,
    selection_fraction: float = 0.15,
    seed: int = 123,
) -> FourWaySplit:
    total = train_fraction + discovery_fraction + selection_fraction
    if total >= 1.0:
        raise ValueError("Fractions must leave a non-zero test split.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(round(train_fraction * n))
    n_disc = int(round(discovery_fraction * n))
    n_sel = int(round(selection_fraction * n))

    a = n_train
    b = a + n_disc
    c = b + n_sel

    return FourWaySplit(
        train=idx[:a],
        discovery=idx[a:b],
        selection=idx[b:c],
        test=idx[c:],
    )


# ---------------------------------------------------------------------
# Cross-fitted candidate learner
# ---------------------------------------------------------------------

Basis = Union[TreeBasis, ScanBasis]


@dataclass
class CrossfitRound:
    round_index: int
    tree_discovery_gain: float
    scan_discovery_gain: float
    tree_selection_gain: float
    scan_selection_gain: float
    chosen: str
    selected_gain: float
    active_terms: int


class CrossfitResidualLearner:
    """
    mode:
        "tree"        -> only residual trees are eligible
        "scan"        -> only scan rules are eligible
        "competitive" -> tree and scan candidates compete on selection gain
    """

    def __init__(
        self,
        *,
        mode: str = "competitive",
        repair_rounds: int = 5,
        tree_max_depth: int = 2,
        coefficient_l2: float = 1e-3,
        selection_gain_tolerance: float = 1.9207,
        mdss_n_bins: int = 5,
        mdss_n_restarts: int = 6,
        mdss_max_passes: int = 6,
        mdss_min_size: int = 40,
        mdss_min_events: int = 3,
        mdss_min_nonevents: int = 3,
        mdss_max_rule_features: int = 6,
        selection_scan_min_size: int = 25,
        selection_scan_min_events: int = 2,
        selection_scan_min_nonevents: int = 2,
        seed: int = 123,
        verbose: bool = False,
    ) -> None:
        if mode not in {"tree", "scan", "competitive"}:
            raise ValueError("mode must be tree, scan, or competitive")

        self.mode = mode
        self.repair_rounds = repair_rounds
        self.tree_max_depth = tree_max_depth
        self.coefficient_l2 = coefficient_l2
        self.selection_gain_tolerance = selection_gain_tolerance
        self.seed = seed
        self.verbose = verbose

        self.selection_scan_min_size = selection_scan_min_size
        self.selection_scan_min_events = selection_scan_min_events
        self.selection_scan_min_nonevents = selection_scan_min_nonevents

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
        self.history_: List[CrossfitRound] = []

    def _Phi(self, X: np.ndarray) -> np.ndarray:
        if not self.bases_:
            return np.empty((len(X), 0), dtype=float)
        return np.column_stack([b.transform(X) for b in self.bases_])

    def repair_logits(self, X: np.ndarray, base_logits: np.ndarray) -> np.ndarray:
        z = np.asarray(base_logits, dtype=float)
        if not self.bases_:
            return z.copy()
        return z + self._Phi(X) @ self.coefficients_

    def _score_scan_on_selection(
        self,
        scan_basis: ScanBasis,
        X_sel: np.ndarray,
        y_sel: np.ndarray,
        current_logits_sel: np.ndarray,
    ):
        mask = scan_basis.transform(X_sel).astype(bool)

        sc = exact_subgroup_logit_correction(
            y_sel,
            current_logits_sel,
            mask,
            min_size=self.selection_scan_min_size,
            min_events=self.selection_scan_min_events,
            min_nonevents=self.selection_scan_min_nonevents,
        )

        if sc is None:
            return 0.0, 0.0

        return float(sc.gain), float(sc.delta)

    def fit(
        self,
        X_disc: np.ndarray,
        y_disc: np.ndarray,
        base_logits_disc: np.ndarray,
        X_sel: np.ndarray,
        y_sel: np.ndarray,
        base_logits_sel: np.ndarray,
    ) -> "CrossfitResidualLearner":
        X_disc = np.asarray(X_disc, dtype=float)
        y_disc = np.asarray(y_disc, dtype=int)
        z0_disc = np.asarray(base_logits_disc, dtype=float)

        X_sel = np.asarray(X_sel, dtype=float)
        y_sel = np.asarray(y_sel, dtype=int)
        z0_sel = np.asarray(base_logits_sel, dtype=float)

        self.scanner.fit_binner(X_disc)

        self.bases_ = []
        self.coefficients_ = np.empty(0)
        self.history_ = []

        for t in range(self.repair_rounds):
            z_disc = self.repair_logits(X_disc, z0_disc)
            z_sel = self.repair_logits(X_sel, z0_sel)

            # ---------------------------------------------------------
            # Tree candidate generated on discovery
            # ---------------------------------------------------------
            tree = None
            tree_disc_gain = 0.0
            tree_sel_gain = -np.inf
            tree_sel_alpha = 0.0

            if self.mode in {"tree", "competitive"}:
                tree = propose_xgb_tree(
                    X_disc,
                    y_disc,
                    z_disc,
                    max_depth=self.tree_max_depth,
                    seed=self.seed + t,
                )
                dir_disc = tree.transform(X_disc)
                dir_sel = tree.transform(X_sel)

                tree_disc = exact_direction_gain(y_disc, z_disc, dir_disc)
                tree_sel = exact_direction_gain(y_sel, z_sel, dir_sel)

                tree_disc_gain = float(tree_disc.gain)
                tree_sel_gain = float(tree_sel.gain)
                tree_sel_alpha = float(tree_sel.alpha)

            # ---------------------------------------------------------
            # Scan candidate generated on discovery
            # ---------------------------------------------------------
            scan_basis = None
            scan_disc_gain = 0.0
            scan_sel_gain = -np.inf
            scan_sel_delta = 0.0

            if self.mode in {"scan", "competitive"}:
                scan = self.scanner.scan(X_disc, y_disc, z_disc)
                scan_basis = ScanBasis(
                    restrictions=dict(scan.restrictions),
                    binner=self.scanner.binner_,
                )

                scan_disc_gain = float(scan.gain)
                scan_sel_gain, scan_sel_delta = self._score_scan_on_selection(
                    scan_basis,
                    X_sel,
                    y_sel,
                    z_sel,
                )

            # ---------------------------------------------------------
            # Candidate choice is based ONLY on independent selection gain
            # ---------------------------------------------------------
            if self.mode == "tree":
                chosen = "tree"
                chosen_basis = tree
                chosen_coef = tree_sel_alpha
                best_sel_gain = tree_sel_gain
            elif self.mode == "scan":
                chosen = "scan"
                chosen_basis = scan_basis
                chosen_coef = scan_sel_delta
                best_sel_gain = scan_sel_gain
            else:
                if tree_sel_gain >= scan_sel_gain:
                    chosen = "tree"
                    chosen_basis = tree
                    chosen_coef = tree_sel_alpha
                    best_sel_gain = tree_sel_gain
                else:
                    chosen = "scan"
                    chosen_basis = scan_basis
                    chosen_coef = scan_sel_delta
                    best_sel_gain = scan_sel_gain

            if self.verbose:
                print(
                    f"round {t+1:02d} | "
                    f"disc(tree={tree_disc_gain:7.3f}, scan={scan_disc_gain:7.3f}) | "
                    f"sel(tree={tree_sel_gain:7.3f}, scan={scan_sel_gain:7.3f}) | "
                    f"choose={chosen}"
                )

            if (
                chosen_basis is None
                or not np.isfinite(best_sel_gain)
                or best_sel_gain < self.selection_gain_tolerance
            ):
                if self.verbose:
                    print("Stopping: no independently supported repair candidate.")
                break

            self.bases_.append(chosen_basis)

            # ---------------------------------------------------------
            # Fully refit ALL repair coefficients on selection only
            # ---------------------------------------------------------
            Phi_sel = self._Phi(X_sel)
            init = np.concatenate([
                self.coefficients_,
                np.array([chosen_coef], dtype=float),
            ])

            fit = fit_residual_dictionary_with_offset(
                y_sel,
                z0_sel,
                Phi_sel,
                l2=self.coefficient_l2,
                init=init,
            )
            self.coefficients_ = fit.coefficients

            self.history_.append(
                CrossfitRound(
                    round_index=t + 1,
                    tree_discovery_gain=float(tree_disc_gain),
                    scan_discovery_gain=float(scan_disc_gain),
                    tree_selection_gain=float(tree_sel_gain),
                    scan_selection_gain=float(scan_sel_gain),
                    chosen=chosen,
                    selected_gain=float(best_sel_gain),
                    active_terms=len(self.bases_),
                )
            )

        return self


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CrossfitCondition:
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


def evaluate(method, y, p, subgroup, runtime, **extra):
    row = {
        "method": method,
        **binary_metrics(y, p),
        **subgroup_diagnostics(y, p, subgroup),
        "runtime_seconds": float(runtime),
    }
    row.update(extra)
    return row


def run_crossfit_condition(
    cond: CrossfitCondition,
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

    # Strong frozen base learner.
    t0 = time.perf_counter()
    base = fit_fixed_budget_xgb(
        Xtr,
        ytr,
        max_depth=cond.tree_depth,
        rounds=cond.base_tree_rounds,
        learning_rate=0.08,
        seed=cond.seed,
    )
    base_runtime = time.perf_counter() - t0

    zd = base.logits(Xd)
    zs = base.logits(Xs)
    zt = base.logits(Xt)

    rows = [
        evaluate(
            "Base XGBoost",
            yt,
            sigmoid(zt),
            st,
            base_runtime,
            repair_terms=0,
            scan_terms=0,
            tree_repair_terms=0,
        )
    ]

    max_rule_features = max(2, min(cond.k + 1, 8))
    learners: Dict[str, CrossfitResidualLearner] = {}

    specs = [
        ("Crossfit extra trees", "tree"),
        ("Crossfit scan repair", "scan"),
        ("Crossfit competitive Tree-Scan", "competitive"),
    ]

    for label, mode in specs:
        t0 = time.perf_counter()

        learner = CrossfitResidualLearner(
            mode=mode,
            repair_rounds=cond.repair_rounds,
            tree_max_depth=cond.tree_depth,
            coefficient_l2=1e-3,
            selection_gain_tolerance=1.9207,
            mdss_max_rule_features=max_rule_features,
            mdss_min_size=max(35, int(round(0.015 * len(Xd)))),
            selection_scan_min_size=max(20, int(round(0.01 * len(Xs)))),
            seed=cond.seed,
            verbose=verbose,
        ).fit(
            Xd, yd, zd,
            Xs, ys, zs,
        )

        repair_runtime = time.perf_counter() - t0
        z = learner.repair_logits(Xt, zt)

        n_scan = sum(r.chosen == "scan" for r in learner.history_)
        n_tree = sum(r.chosen == "tree" for r in learner.history_)

        rows.append(
            evaluate(
                label,
                yt,
                sigmoid(z),
                st,
                base_runtime + repair_runtime,
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


def smoke_crossfit_experiment():
    frames = []

    for k in (2, 5):
        for seed in (1101, 2202):
            print(f"crossfit smoke: k={k}, depth=2, seed={seed}")
            cond = CrossfitCondition(
                n=12_000,
                p=30,
                k=k,
                s=0.08,
                delta=float(np.log(4.0)),
                pi=0.10,
                tree_depth=2,
                base_tree_rounds=70,
                repair_rounds=4,
                seed=seed,
            )
            df, _ = run_crossfit_condition(cond, verbose=False)
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

    out.to_csv("/mnt/data/crossfit_repair_smoke_results.csv", index=False)
    summary.to_csv("/mnt/data/crossfit_repair_smoke_summary.csv", index=False)
    return out, summary


if __name__ == "__main__":
    results, summary = smoke_crossfit_experiment()
    print(summary.to_string(index=False))
