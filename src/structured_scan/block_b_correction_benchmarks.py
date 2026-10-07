
"""
Block B: correction-method benchmark for Structured Scan Repair.

The scientific comparison is deliberately paired.

For every synthetic replicate:
    TRAIN       -> fit ONE frozen XGBoost base model
    DISCOVERY   -> correction discovery/auditing
    SELECTION   -> independent correction validation where method-appropriate
    TEST        -> common untouched final evaluation

All post-processing methods start from the SAME frozen XGBoost predictions.

Methods
-------
1. Base XGBoost
2. Continued-tree repair
3. Multiaccuracy-style residual boosting
4. Official MCGrad (when the external `mcgrad` package is installed)
5. Newton Rule Boosting (same rule language/search budget; second-order objective)
6. Structured Scan Repair (exact Bernoulli/logistic likelihood)
7. Competitive Tree-Scan

Important naming
----------------
- "Multiaccuracy-style" is a transparent tabular implementation inspired by
  Kim, Ghorbani & Zou. It is NOT claimed to be their original TensorFlow code.
- "Newton Rule Boosting" is an ORB-style second-order rule comparator using
  our same beam-search rule class. It is NOT claimed to reproduce ORB's
  branch-and-bound optimal search.

This lets us distinguish:
    correction paradigm,
    search class,
    and exact-likelihood vs second-order rule scoring
without falsely relabeling approximations as external published software.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib.util
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from mdss_treescan_stage1 import (
    SyntheticConfig,
    generate_structured_residual_data,
    fit_quantile_binner,
    sigmoid,
)
from benchmark_treescan import (
    fit_fixed_budget_xgb,
    subgroup_diagnostics,
)
from treescan_prototype import (
    ScanBasis,
    binary_metrics,
)
from crossfit_repair_benchmark_strict import four_way_split
from crossfit_beam_benchmark_strict import make_learner
from structured_mdss import candidate_bin_subsets


EPS = 1e-7
STRICT_SELECTION_GAIN = 1.9207


# =====================================================================
# Common helpers
# =====================================================================

def clip_prob(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = clip_prob(p, eps=eps)
    return np.log(p / (1.0 - p))


def logistic_loss_sum(y: np.ndarray, logits: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    z = np.asarray(logits, dtype=float)
    # Stable logistic negative log-likelihood.
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Simple equal-width ECE used identically for every method."""
    y = np.asarray(y, dtype=float)
    p = clip_prob(p)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    n = len(y)

    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        ece += (
            float(mask.mean())
            * abs(float(y[mask].mean()) - float(p[mask].mean()))
        )

    return float(ece)


def evaluate_predictions(
    method: str,
    y: np.ndarray,
    p: np.ndarray,
    subgroup: np.ndarray,
    *,
    runtime_seconds: float,
    correction_seconds: float,
    updates: float = np.nan,
    scan_terms: float = np.nan,
    tree_terms: float = np.nan,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "method": method,
        **binary_metrics(y, p),
        **subgroup_diagnostics(y, p, subgroup),
        "ece_15": expected_calibration_error(y, p, n_bins=15),
        "runtime_seconds": float(runtime_seconds),
        "correction_seconds": float(correction_seconds),
        "updates": updates,
        "scan_terms": scan_terms,
        "tree_terms": tree_terms,
    }

    if extra:
        row.update(extra)

    return row


# =====================================================================
# Multiaccuracy-style residual boosting
# =====================================================================

@dataclass
class MultiaccuracyRound:
    round_index: int
    partition: str
    selection_correlation: float


class MultiaccuracyStyleBoost:
    """
    Transparent tabular implementation of the Multiaccuracy Boost idea.

    Auditor:
        decision-tree regression (depth 5) of current probability residual
        f(x) - y, matching the tree-auditor choice used in the original paper.

    Fixed initial partitions:
        all,
        base prediction <= 0.5,
        base prediction > 0.5.

    Update:
        logit(f_{t+1}) = logit(f_t) - eta * h_t(x)
    on the selected partition, which is the binary multiplicative-weights
    update written in log-odds form.

    We use DISCOVERY to fit auditors and independent SELECTION correlation to
    choose/stop. This is intentionally conservative and avoids reusing test data.
    """

    def __init__(
        self,
        *,
        max_rounds: int = 25,
        auditor_depth: int = 5,
        auditor_min_leaf: int = 25,
        eta: float = 1.0,
        alpha: float = 1e-3,
        seed: int = 123,
    ) -> None:
        self.max_rounds = int(max_rounds)
        self.auditor_depth = int(auditor_depth)
        self.auditor_min_leaf = int(auditor_min_leaf)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.seed = int(seed)

        self.steps_: List[Tuple[str, DecisionTreeRegressor]] = []
        self.history_: List[MultiaccuracyRound] = []

    @staticmethod
    def _partition_mask(
        partition: str,
        base_prob: np.ndarray,
    ) -> np.ndarray:
        if partition == "all":
            return np.ones(len(base_prob), dtype=bool)
        if partition == "base_negative":
            return np.asarray(base_prob) <= 0.5
        if partition == "base_positive":
            return np.asarray(base_prob) > 0.5
        raise ValueError(f"Unknown partition: {partition}")

    def _apply_steps(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
        base_prob_for_partition: np.ndarray,
    ) -> np.ndarray:
        z = np.asarray(base_logits, dtype=float).copy()

        for partition, auditor in self.steps_:
            mask = self._partition_mask(
                partition,
                base_prob_for_partition,
            )
            if not np.any(mask):
                continue

            h = np.clip(
                auditor.predict(np.asarray(X, dtype=float)),
                -1.0,
                1.0,
            )
            z[mask] -= self.eta * h[mask]

        return z

    def fit(
        self,
        X_disc: np.ndarray,
        y_disc: np.ndarray,
        base_logits_disc: np.ndarray,
        X_sel: np.ndarray,
        y_sel: np.ndarray,
        base_logits_sel: np.ndarray,
    ) -> "MultiaccuracyStyleBoost":
        X_disc = np.asarray(X_disc, dtype=float)
        y_disc = np.asarray(y_disc, dtype=float)
        z0_disc = np.asarray(base_logits_disc, dtype=float)

        X_sel = np.asarray(X_sel, dtype=float)
        y_sel = np.asarray(y_sel, dtype=float)
        z0_sel = np.asarray(base_logits_sel, dtype=float)

        base_p_disc = sigmoid(z0_disc)
        base_p_sel = sigmoid(z0_sel)

        self.steps_ = []
        self.history_ = []

        partitions = (
            "all",
            "base_negative",
            "base_positive",
        )

        for t in range(self.max_rounds):
            z_disc = self._apply_steps(
                X_disc,
                z0_disc,
                base_p_disc,
            )
            z_sel = self._apply_steps(
                X_sel,
                z0_sel,
                base_p_sel,
            )

            p_disc = sigmoid(z_disc)
            p_sel = sigmoid(z_sel)

            residual_disc = p_disc - y_disc
            residual_sel = p_sel - y_sel

            candidates = []

            for j, partition in enumerate(partitions):
                mask_disc = self._partition_mask(
                    partition,
                    base_p_disc,
                )
                mask_sel = self._partition_mask(
                    partition,
                    base_p_sel,
                )

                if (
                    mask_disc.sum() < max(50, self.auditor_min_leaf * 2)
                    or mask_sel.sum() < 30
                ):
                    continue

                auditor = DecisionTreeRegressor(
                    max_depth=self.auditor_depth,
                    min_samples_leaf=self.auditor_min_leaf,
                    random_state=self.seed + 100 * t + j,
                )
                auditor.fit(
                    X_disc[mask_disc],
                    residual_disc[mask_disc],
                )

                h_sel = np.clip(
                    auditor.predict(X_sel),
                    -1.0,
                    1.0,
                )

                # E[h_S(x) * (f(x)-y)] over the full selection population,
                # where h_S is zero outside the fixed partition S.
                correlation = float(
                    np.mean(
                        h_sel
                        * residual_sel
                        * mask_sel.astype(float)
                    )
                )

                candidates.append(
                    (correlation, partition, auditor)
                )

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            best_corr, best_partition, best_auditor = candidates[0]

            if not np.isfinite(best_corr) or best_corr <= self.alpha:
                break

            self.steps_.append(
                (best_partition, best_auditor)
            )
            self.history_.append(
                MultiaccuracyRound(
                    round_index=t + 1,
                    partition=best_partition,
                    selection_correlation=float(best_corr),
                )
            )

        return self

    def repair_logits(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
    ) -> np.ndarray:
        return self._apply_steps(
            X,
            base_logits,
            sigmoid(base_logits),
        )


# =====================================================================
# Newton / second-order rule boosting comparator
# =====================================================================

@dataclass
class NewtonRuleState:
    restrictions: Dict[int, Tuple[int, ...]]
    mask: np.ndarray
    gain: float
    delta: float


@dataclass
class NewtonRuleResult:
    gain: float
    delta: float
    restrictions: Dict[int, Tuple[int, ...]]
    depth: int
    evaluated_candidates: int


class NewtonBeamRuleScanner:
    """
    Same binner, literal family, conjunction depth and beam width as the exact
    StructuredBeamMDSS scanner, but candidates are scored by the conventional
    second-order logistic/Newton gain:

        gain ~= 0.5 * G^2 / (H + lambda)
        delta = -G / (H + lambda)

    where G = sum(p-y), H = sum p(1-p).

    This is an ORB-style objective comparator, not ORB's exact branch-and-bound
    search implementation.
    """

    def __init__(
        self,
        *,
        n_bins: int = 5,
        max_rule_features: int = 5,
        beam_width: int = 20,
        literal_mode: str = "tails",
        min_size: int = 40,
        min_events: int = 3,
        min_nonevents: int = 3,
        l2: float = 1e-3,
    ) -> None:
        self.n_bins = int(n_bins)
        self.max_rule_features = int(max_rule_features)
        self.beam_width = int(beam_width)
        self.literal_mode = literal_mode
        self.min_size = int(min_size)
        self.min_events = int(min_events)
        self.min_nonevents = int(min_nonevents)
        self.l2 = float(l2)

        self.binner_ = None

    def fit_binner(
        self,
        X: np.ndarray,
    ) -> "NewtonBeamRuleScanner":
        self.binner_ = fit_quantile_binner(
            np.asarray(X, dtype=float),
            self.n_bins,
        )
        return self

    @staticmethod
    def _rule_key(
        restrictions: Dict[int, Tuple[int, ...]]
    ) -> Tuple:
        return tuple(
            (int(j), tuple(int(v) for v in vals))
            for j, vals in sorted(restrictions.items())
        )

    def _score(
        self,
        y: np.ndarray,
        logits: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        y = np.asarray(y, dtype=float)
        z = np.asarray(logits, dtype=float)
        mask = np.asarray(mask, dtype=bool)

        n = int(mask.sum())
        if n < self.min_size:
            return None

        ys = y[mask]
        events = int(ys.sum())
        nonevents = n - events

        if (
            events < self.min_events
            or nonevents < self.min_nonevents
        ):
            return None

        p = sigmoid(z[mask])
        g = p - ys
        h = p * (1.0 - p)

        G = float(g.sum())
        H = float(h.sum())

        denom = H + self.l2
        if denom <= 0:
            return None

        delta = -G / denom
        gain = 0.5 * (G * G) / denom

        return float(gain), float(delta)

    def scan(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
    ) -> NewtonRuleResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        z = np.asarray(logits, dtype=float)

        if self.binner_ is None:
            self.fit_binner(X)

        Xb = self.binner_.transform(X)
        n, p = Xb.shape

        levels = [
            tuple(int(v) for v in np.unique(Xb[:, j]))
            for j in range(p)
        ]
        literals = [
            candidate_bin_subsets(
                levels[j],
                mode=self.literal_mode,
            )
            for j in range(p)
        ]

        beam_rules = [
            ({}, np.ones(n, dtype=bool))
        ]

        best = None
        best_depth = 0
        evaluated = 0

        for depth in range(1, self.max_rule_features + 1):
            child_states: Dict[Tuple, NewtonRuleState] = {}

            for restrictions, parent_mask in beam_rules:
                used = set(restrictions)

                for j in range(p):
                    if j in used:
                        continue

                    for allowed in literals[j]:
                        mask = parent_mask & np.isin(
                            Xb[:, j],
                            allowed,
                        )

                        if int(mask.sum()) < self.min_size:
                            continue

                        evaluated += 1
                        scored = self._score(y, z, mask)
                        if scored is None:
                            continue

                        gain, delta = scored

                        new_restrictions = dict(restrictions)
                        new_restrictions[j] = tuple(allowed)
                        key = self._rule_key(new_restrictions)

                        state = NewtonRuleState(
                            restrictions=new_restrictions,
                            mask=mask,
                            gain=gain,
                            delta=delta,
                        )

                        old = child_states.get(key)
                        if old is None or state.gain > old.gain:
                            child_states[key] = state

            if not child_states:
                break

            ranked = sorted(
                child_states.values(),
                key=lambda st: st.gain,
                reverse=True,
            )

            beam = ranked[: self.beam_width]

            if best is None or beam[0].gain > best.gain:
                best = beam[0]
                best_depth = depth

            beam_rules = [
                (st.restrictions, st.mask)
                for st in beam
            ]

        if best is None:
            raise RuntimeError("No valid Newton rule found.")

        return NewtonRuleResult(
            gain=float(best.gain),
            delta=float(best.delta),
            restrictions=dict(best.restrictions),
            depth=int(best_depth),
            evaluated_candidates=int(evaluated),
        )


@dataclass
class NewtonRuleRound:
    round_index: int
    discovery_gain: float
    selection_gain: float
    delta: float
    rule_depth: int


class NewtonRuleBoost:
    def __init__(
        self,
        *,
        max_rounds: int = 4,
        n_bins: int = 5,
        max_rule_features: int = 5,
        beam_width: int = 20,
        selection_gain_tolerance: float = STRICT_SELECTION_GAIN,
        min_size_discovery: int = 40,
        min_size_selection: int = 25,
        min_events_discovery: int = 3,
        min_events_selection: int = 2,
        min_nonevents_discovery: int = 3,
        min_nonevents_selection: int = 2,
        l2: float = 1e-3,
    ) -> None:
        self.max_rounds = int(max_rounds)
        self.selection_gain_tolerance = float(
            selection_gain_tolerance
        )

        self.scanner = NewtonBeamRuleScanner(
            n_bins=n_bins,
            max_rule_features=max_rule_features,
            beam_width=beam_width,
            literal_mode="tails",
            min_size=min_size_discovery,
            min_events=min_events_discovery,
            min_nonevents=min_nonevents_discovery,
            l2=l2,
        )

        self.selection_min_size = int(min_size_selection)
        self.selection_min_events = int(min_events_selection)
        self.selection_min_nonevents = int(min_nonevents_selection)
        self.l2 = float(l2)

        self.rules_: List[Tuple[ScanBasis, float]] = []
        self.history_: List[NewtonRuleRound] = []

    def _current_logits(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
    ) -> np.ndarray:
        z = np.asarray(base_logits, dtype=float).copy()
        for basis, delta in self.rules_:
            z += float(delta) * basis.transform(X)
        return z

    def _selection_score(
        self,
        basis: ScanBasis,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        mask = basis.transform(X).astype(bool)

        n = int(mask.sum())
        if n < self.selection_min_size:
            return None

        ys = np.asarray(y, dtype=float)[mask]
        events = int(ys.sum())
        nonevents = n - events

        if (
            events < self.selection_min_events
            or nonevents < self.selection_min_nonevents
        ):
            return None

        p = sigmoid(np.asarray(logits, dtype=float)[mask])
        g = p - ys
        h = p * (1.0 - p)

        G = float(g.sum())
        H = float(h.sum())
        denom = H + self.l2

        if denom <= 0:
            return None

        delta = -G / denom
        gain = 0.5 * G * G / denom
        return float(gain), float(delta)

    def fit(
        self,
        X_disc: np.ndarray,
        y_disc: np.ndarray,
        base_logits_disc: np.ndarray,
        X_sel: np.ndarray,
        y_sel: np.ndarray,
        base_logits_sel: np.ndarray,
    ) -> "NewtonRuleBoost":
        self.rules_ = []
        self.history_ = []
        self.scanner.fit_binner(X_disc)

        for t in range(self.max_rounds):
            z_disc = self._current_logits(
                X_disc,
                base_logits_disc,
            )
            z_sel = self._current_logits(
                X_sel,
                base_logits_sel,
            )

            result = self.scanner.scan(
                X_disc,
                y_disc,
                z_disc,
            )

            basis = ScanBasis(
                restrictions=dict(result.restrictions),
                binner=self.scanner.binner_,
            )

            sel = self._selection_score(
                basis,
                X_sel,
                y_sel,
                z_sel,
            )

            if sel is None:
                break

            selection_gain, selection_delta = sel

            if (
                not np.isfinite(selection_gain)
                or selection_gain < self.selection_gain_tolerance
            ):
                break

            self.rules_.append(
                (basis, float(selection_delta))
            )
            self.history_.append(
                NewtonRuleRound(
                    round_index=t + 1,
                    discovery_gain=float(result.gain),
                    selection_gain=float(selection_gain),
                    delta=float(selection_delta),
                    rule_depth=int(result.depth),
                )
            )

        return self

    def repair_logits(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
    ) -> np.ndarray:
        return self._current_logits(X, base_logits)


# =====================================================================
# Official MCGrad adapter
# =====================================================================

def mcgrad_is_available() -> bool:
    return importlib.util.find_spec("mcgrad") is not None


def _mcgrad_frame(
    X: np.ndarray,
    prediction: np.ndarray,
    y: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    X = np.asarray(X, dtype=float)
    p = clip_prob(prediction)

    feature_names = [
        f"x{j+1}" for j in range(X.shape[1])
    ]

    df = pd.DataFrame(
        X,
        columns=feature_names,
    )
    df["prediction"] = p

    if y is not None:
        df["label"] = np.asarray(y, dtype=int)

    return df, feature_names


def run_official_mcgrad(
    X_correction: np.ndarray,
    y_correction: np.ndarray,
    p_correction: np.ndarray,
    X_test: np.ndarray,
    p_test: np.ndarray,
    *,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Uses the external official `mcgrad` package with its published/default
    early-stopping configuration.

    The full DISCOVERY+SELECTION correction pool is supplied because MCGrad
    performs its own internal validation/cross-validation. Thus every method
    receives the same total labeled correction-data budget, while retaining
    method-native stopping.
    """
    if not mcgrad_is_available():
        raise ImportError(
            "The official MCGrad package is not installed. "
            "Install it with: python -m pip install mcgrad"
        )

    from mcgrad import methods

    df_corr, feature_names = _mcgrad_frame(
        X_correction,
        p_correction,
        y_correction,
    )
    df_test, _ = _mcgrad_frame(
        X_test,
        p_test,
        None,
    )

    calibrator = methods.MCGrad(
        random_state=int(seed),
    )

    t0 = time.perf_counter()

    calibrator.fit(
        df_train=df_corr,
        prediction_column_name="prediction",
        label_column_name="label",
        categorical_feature_column_names=[],
        numerical_feature_column_names=feature_names,
    )

    pred = calibrator.predict(
        df=df_test,
        prediction_column_name="prediction",
        categorical_feature_column_names=[],
        numerical_feature_column_names=feature_names,
    )

    runtime = time.perf_counter() - t0

    pred = clip_prob(np.asarray(pred, dtype=float).reshape(-1))

    info = {
        "mcgrad_runtime_seconds": float(runtime),
        "mcgrad_version": None,
    }

    try:
        import mcgrad
        info["mcgrad_version"] = getattr(
            mcgrad,
            "__version__",
            None,
        )
    except Exception:
        pass

    return pred, info


# =====================================================================
# Main Block B experiment
# =====================================================================

@dataclass(frozen=True)
class BlockBCondition:
    support: float
    odds_ratio: float
    seed: int
    n: int = 12_000
    p: int = 30
    k: int = 5
    event_prevalence: float = 0.10
    tree_depth: int = 2
    base_tree_rounds: int = 70
    repair_rounds: int = 4
    beam_width: int = 20

    # Multiaccuracy-style fixed settings
    ma_max_rounds: int = 10
    ma_auditor_depth: int = 5
    ma_auditor_min_leaf: int = 25
    ma_eta: float = 1.0
    ma_alpha: float = 1e-4


BLOCK_B_METHODS = (
    "Base XGBoost",
    "Continued-tree repair",
    "Multiaccuracy-style Boost",
    "Official MCGrad",
    "Newton Rule Boosting",
    "Structured Scan Repair",
    "Competitive Tree-Scan",
)


def run_block_b_condition(
    cond: BlockBCondition,
    *,
    include_mcgrad: bool = True,
    allow_missing_mcgrad: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fit ONE base model, then benchmark every correction method from that
    identical starting point.
    """
    cfg = SyntheticConfig(
        n=cond.n,
        p=cond.p,
        interaction_order=cond.k,
        subgroup_prevalence=cond.support,
        subgroup_log_odds=float(np.log(cond.odds_ratio)),
        event_prevalence=cond.event_prevalence,
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

    # -------------------------------------------------------------
    # ONE frozen XGBoost base, shared by every correction method.
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
    base_seconds = time.perf_counter() - t0

    zd = base.logits(Xd)
    zs = base.logits(Xs)
    zt = base.logits(Xt)

    pd_base = sigmoid(zd)
    ps_base = sigmoid(zs)
    pt_base = sigmoid(zt)

    rows = [
        evaluate_predictions(
            "Base XGBoost",
            yt,
            pt_base,
            st,
            runtime_seconds=base_seconds,
            correction_seconds=0.0,
            updates=0,
            scan_terms=0,
            tree_terms=0,
        )
    ]

    # -------------------------------------------------------------
    # Existing strict crossfit continued-tree repair
    # -------------------------------------------------------------
    tree_learner = make_learner(
        mode="tree",
        cond=type(
            "_Cond",
            (),
            {
                "repair_rounds": cond.repair_rounds,
                "tree_depth": cond.tree_depth,
                "k": cond.k,
                "beam_width": cond.beam_width,
                "seed": cond.seed,
            },
        )(),
        discovery_n=len(Xd),
        selection_n=len(Xs),
        verbose=verbose,
    )

    t0 = time.perf_counter()
    tree_learner.fit(
        Xd, yd, zd,
        Xs, ys, zs,
    )
    correction_seconds = time.perf_counter() - t0

    z_tree = tree_learner.repair_logits(Xt, zt)

    rows.append(
        evaluate_predictions(
            "Continued-tree repair",
            yt,
            sigmoid(z_tree),
            st,
            runtime_seconds=base_seconds + correction_seconds,
            correction_seconds=correction_seconds,
            updates=len(tree_learner.history_),
            scan_terms=0,
            tree_terms=len(tree_learner.history_),
        )
    )

    # -------------------------------------------------------------
    # Multiaccuracy-style Boost
    # -------------------------------------------------------------
    ma = MultiaccuracyStyleBoost(
        max_rounds=cond.ma_max_rounds,
        auditor_depth=cond.ma_auditor_depth,
        auditor_min_leaf=cond.ma_auditor_min_leaf,
        eta=cond.ma_eta,
        alpha=cond.ma_alpha,
        seed=cond.seed,
    )

    t0 = time.perf_counter()
    ma.fit(
        Xd, yd, zd,
        Xs, ys, zs,
    )
    correction_seconds = time.perf_counter() - t0

    z_ma = ma.repair_logits(Xt, zt)

    rows.append(
        evaluate_predictions(
            "Multiaccuracy-style Boost",
            yt,
            sigmoid(z_ma),
            st,
            runtime_seconds=base_seconds + correction_seconds,
            correction_seconds=correction_seconds,
            updates=len(ma.history_),
            extra={
                "ma_last_selection_correlation": (
                    ma.history_[-1].selection_correlation
                    if ma.history_ else np.nan
                ),
            },
        )
    )

    # -------------------------------------------------------------
    # Official MCGrad
    # -------------------------------------------------------------
    if include_mcgrad:
        if mcgrad_is_available():
            Xcorr = np.vstack([Xd, Xs])
            ycorr = np.concatenate([yd, ys])
            pcorr = np.concatenate([pd_base, ps_base])

            p_mcgrad, mc_info = run_official_mcgrad(
                Xcorr,
                ycorr,
                pcorr,
                Xt,
                pt_base,
                seed=cond.seed,
            )

            correction_seconds = float(
                mc_info["mcgrad_runtime_seconds"]
            )

            rows.append(
                evaluate_predictions(
                    "Official MCGrad",
                    yt,
                    p_mcgrad,
                    st,
                    runtime_seconds=base_seconds + correction_seconds,
                    correction_seconds=correction_seconds,
                    extra=mc_info,
                )
            )
        elif allow_missing_mcgrad:
            rows.append(
                {
                    "method": "Official MCGrad",
                    "mcgrad_missing": True,
                }
            )
        else:
            raise ImportError(
                "MCGrad is required for the full Block B benchmark. "
                "Install it with `python -m pip install mcgrad`, "
                "then restart the notebook kernel."
            )

    # -------------------------------------------------------------
    # Newton / second-order rule boosting
    # -------------------------------------------------------------
    newton = NewtonRuleBoost(
        max_rounds=cond.repair_rounds,
        n_bins=5,
        max_rule_features=max(2, min(cond.k, 6)),
        beam_width=cond.beam_width,
        selection_gain_tolerance=STRICT_SELECTION_GAIN,
        min_size_discovery=max(
            35,
            int(round(0.015 * len(Xd))),
        ),
        min_size_selection=max(
            20,
            int(round(0.01 * len(Xs))),
        ),
        min_events_discovery=3,
        min_events_selection=2,
        min_nonevents_discovery=3,
        min_nonevents_selection=2,
        l2=1e-3,
    )

    t0 = time.perf_counter()
    newton.fit(
        Xd, yd, zd,
        Xs, ys, zs,
    )
    correction_seconds = time.perf_counter() - t0

    z_newton = newton.repair_logits(Xt, zt)

    rows.append(
        evaluate_predictions(
            "Newton Rule Boosting",
            yt,
            sigmoid(z_newton),
            st,
            runtime_seconds=base_seconds + correction_seconds,
            correction_seconds=correction_seconds,
            updates=len(newton.history_),
            scan_terms=len(newton.history_),
            tree_terms=0,
            extra={
                "newton_mean_selection_gain": (
                    np.mean(
                        [h.selection_gain for h in newton.history_]
                    )
                    if newton.history_ else np.nan
                ),
            },
        )
    )

    # -------------------------------------------------------------
    # Exact Structured Scan Repair
    # -------------------------------------------------------------
    scan_learner = make_learner(
        mode="scan",
        cond=type(
            "_Cond",
            (),
            {
                "repair_rounds": cond.repair_rounds,
                "tree_depth": cond.tree_depth,
                "k": cond.k,
                "beam_width": cond.beam_width,
                "seed": cond.seed,
            },
        )(),
        discovery_n=len(Xd),
        selection_n=len(Xs),
        verbose=verbose,
    )

    t0 = time.perf_counter()
    scan_learner.fit(
        Xd, yd, zd,
        Xs, ys, zs,
    )
    correction_seconds = time.perf_counter() - t0

    z_scan = scan_learner.repair_logits(Xt, zt)

    rows.append(
        evaluate_predictions(
            "Structured Scan Repair",
            yt,
            sigmoid(z_scan),
            st,
            runtime_seconds=base_seconds + correction_seconds,
            correction_seconds=correction_seconds,
            updates=len(scan_learner.history_),
            scan_terms=len(scan_learner.history_),
            tree_terms=0,
        )
    )

    # -------------------------------------------------------------
    # Competitive Tree-Scan
    # -------------------------------------------------------------
    comp_learner = make_learner(
        mode="competitive",
        cond=type(
            "_Cond",
            (),
            {
                "repair_rounds": cond.repair_rounds,
                "tree_depth": cond.tree_depth,
                "k": cond.k,
                "beam_width": cond.beam_width,
                "seed": cond.seed,
            },
        )(),
        discovery_n=len(Xd),
        selection_n=len(Xs),
        verbose=verbose,
    )

    t0 = time.perf_counter()
    comp_learner.fit(
        Xd, yd, zd,
        Xs, ys, zs,
    )
    correction_seconds = time.perf_counter() - t0

    z_comp = comp_learner.repair_logits(Xt, zt)

    n_scan = sum(
        h.chosen == "scan"
        for h in comp_learner.history_
    )
    n_tree = sum(
        h.chosen == "tree"
        for h in comp_learner.history_
    )

    rows.append(
        evaluate_predictions(
            "Competitive Tree-Scan",
            yt,
            sigmoid(z_comp),
            st,
            runtime_seconds=base_seconds + correction_seconds,
            correction_seconds=correction_seconds,
            updates=len(comp_learner.history_),
            scan_terms=n_scan,
            tree_terms=n_tree,
        )
    )

    out = pd.DataFrame(rows)

    for key, value in asdict(cond).items():
        out[key] = value

    out["delta"] = float(np.log(cond.odds_ratio))
    out["realized_event_prevalence"] = float(
        data.y.mean()
    )
    out["realized_subgroup_prevalence"] = float(
        data.subgroup_mask.mean()
    )
    out["base_fit_seconds"] = float(base_seconds)

    return out


# =====================================================================
# Optional official MCGrad MCE metric
# =====================================================================

def official_mcgrad_mce(
    X: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
) -> Dict[str, float]:
    """
    Optional method-neutral evaluation using MCGrad's published
    MulticalibrationError metric.

    This can be computationally heavier, so it is intentionally separate from
    the main benchmark loop.
    """
    if not mcgrad_is_available():
        raise ImportError(
            "Install mcgrad before computing official MCE."
        )

    from mcgrad.metrics import MulticalibrationError

    df, feature_names = _mcgrad_frame(
        X,
        p,
        y,
    )

    mce = MulticalibrationError(
        df=df,
        label_column="label",
        score_column="prediction",
        categorical_segment_columns=[],
        numerical_segment_columns=feature_names,
    )

    return {
        "mce": float(mce.mce),
        "mce_sigma": float(mce.mce_sigma),
        "mce_pvalue": float(mce.mce_pvalue),
        "mde": float(mce.mde),
    }


if __name__ == "__main__":
    # Smoke test excludes MCGrad so the local module can be validated even
    # before the external package is installed.
    cond = BlockBCondition(
        support=0.08,
        odds_ratio=8.0,
        seed=20001,
        n=5_000,
        repair_rounds=2,
        beam_width=8,
        ma_max_rounds=10,
    )

    result = run_block_b_condition(
        cond,
        include_mcgrad=False,
    )

    print(
        result[
            [
                "method",
                "logloss",
                "auroc",
                "auprc",
                "brier",
                "ece_15",
                "true_subgroup_mean_residual",
                "updates",
                "runtime_seconds",
            ]
        ].to_string(index=False)
    )
