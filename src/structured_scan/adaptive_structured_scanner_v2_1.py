
"""
Adaptive Structured Scanner v2.1
==============================

A stronger search oracle for exact-likelihood Structured Scan Repair.

Core objective -- unchanged
---------------------------
For any subgroup S and logit correction delta,

    Delta(S, delta)
      = sum_{i in S} [
            y_i * delta
            - log(1 - p_i + exp(delta) p_i)
        ]

is the exact logistic-loss improvement.

Scanner v2 improves ONLY the search oracle:

1. Numeric features
   ----------------
   No fixed 5-bin representation.

   Within each current parent subgroup:
     * consider every feasible observed threshold;
     * compute the second-order / Newton rule gain cheaply by cumulative sums;
     * retain the strongest threshold candidates per direction;
     * exact-score those candidates with the Bernoulli likelihood objective.

   Thus Newton gain is a proposal/screening device, not the final score.

2. Categorical features
   --------------------
   Categories remain unordered.

   We use the Bernoulli predictive-bias / ALTSS structure:
     * for each category, compute the q interval in which that category makes a
       positive contribution to the likelihood-ratio score;
     * candidate category subsets change only at these q roots;
     * exact-score the resulting linear-size collection of subsets.

   This avoids arbitrary ordinal coding and exponential enumeration.

3. Conjunction search
   ------------------
   * diversity-aware beam retention;
   * multiple exact candidates per feature;
   * final coordinate refinement of the strongest beam states.

The global conjunction problem remains non-convex. Scanner v2 is NOT claimed
to be a globally exact combinatorial optimizer.

The module also provides a scan-only cross-fitted learner compatible with the
existing discovery -> selection -> test protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal as TypingLiteral
import math

import numpy as np

from mdss_treescan_stage1 import (
    SubgroupScore,
    exact_subgroup_logit_correction,
    sigmoid,
)
from residual_repair_benchmark import fit_residual_dictionary_with_offset


# ============================================================================
# Literal / basis representation
# ============================================================================

@dataclass(frozen=True)
class NumericLiteral:
    feature: int
    direction: str  # "le" or "gt"
    threshold: float

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X)[:, self.feature]
        if self.direction == "le":
            return x <= self.threshold
        if self.direction == "gt":
            return x > self.threshold
        raise ValueError(f"Unknown direction: {self.direction}")

    def key(self) -> Tuple:
        return (
            "num",
            int(self.feature),
            self.direction,
            float(self.threshold),
        )


@dataclass(frozen=True)
class CategoricalLiteral:
    feature: int
    allowed: Tuple[float, ...]

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X)[:, self.feature]
        return np.isin(x, np.asarray(self.allowed))

    def key(self) -> Tuple:
        return (
            "cat",
            int(self.feature),
            tuple(float(v) for v in self.allowed),
        )


RuleLiteral = NumericLiteral | CategoricalLiteral


@dataclass
class AdaptiveRuleBasis:
    literals: Dict[int, RuleLiteral]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        mask = np.ones(len(X), dtype=bool)

        for j in sorted(self.literals):
            mask &= self.literals[j].transform(X)
            if not np.any(mask):
                break

        return mask.astype(float)

    @property
    def n_features(self) -> int:
        return len(self.literals)

    def key(self) -> Tuple:
        return tuple(
            self.literals[j].key()
            for j in sorted(self.literals)
        )


# ============================================================================
# Search result
# ============================================================================

@dataclass
class AdaptiveBeamState:
    basis: AdaptiveRuleBasis
    mask: np.ndarray
    score: SubgroupScore
    last_feature: Optional[int] = None


@dataclass
class AdaptiveScanResult:
    gain: float
    delta: float
    basis: AdaptiveRuleBasis
    mask: np.ndarray
    score: SubgroupScore
    depth: int
    evaluated_exact_candidates: int
    screened_numeric_candidates: int
    refined: bool


# ============================================================================
# Exact score helpers
# ============================================================================

def _valid_score(
    y: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    min_size: int,
    min_events: int,
    min_nonevents: int,
    l2: float,
) -> Optional[SubgroupScore]:
    return exact_subgroup_logit_correction(
        y,
        logits,
        mask,
        l2=l2,
        min_size=min_size,
        min_events=min_events,
        min_nonevents=min_nonevents,
    )


def _bernoulli_contribution_at_q(
    observed_sum: float,
    expectations: np.ndarray,
    q: float,
) -> float:
    """
    Bernoulli predictive-bias likelihood contribution for one category.

    q = exp(delta)
    """
    p = np.clip(
        np.asarray(expectations, dtype=float),
        1e-12,
        1.0 - 1e-12,
    )
    q = float(q)

    if q <= 0:
        return -np.inf

    return float(
        observed_sum * np.log(q)
        - np.log(1.0 - p + q * p).sum()
    )


def _category_positive_q_interval(
    y_group: np.ndarray,
    p_group: np.ndarray,
    *,
    max_q: float = 1e6,
    min_q: float = 1e-8,
    root_tol: float = 1e-9,
    max_iter: int = 80,
) -> Optional[Tuple[str, float]]:
    """
    With zero category-level penalty, score(q=1)=0.

    If residual sum > 0:
        category contributes positively for q in (1, q_max_root)
        (possibly q_max_root = +inf).

    If residual sum < 0:
        category contributes positively for q in (q_min_root, 1)
        (possibly q_min_root = 0).

    Returns:
        ("positive", q_max_root)   for underprediction / q > 1
        ("negative", q_min_root)   for overprediction / q < 1
        None                       if derivative at q=1 is zero.
    """
    y_group = np.asarray(y_group, dtype=float)
    p_group = np.asarray(p_group, dtype=float)

    residual = float(np.sum(y_group - p_group))

    if abs(residual) <= 1e-12:
        return None

    observed_sum = float(y_group.sum())

    if residual > 0:
        # Find the non-unit root to the right of q=1.
        lo = 1.0
        hi = 2.0

        f_hi = _bernoulli_contribution_at_q(
            observed_sum,
            p_group,
            hi,
        )

        while f_hi > 0 and hi < max_q:
            hi *= 2.0
            f_hi = _bernoulli_contribution_at_q(
                observed_sum,
                p_group,
                hi,
            )

        if f_hi > 0:
            return ("positive", float("inf"))

        # Root lies in (1, hi); avoid the known root at exactly 1.
        lo = 1.0 + 1e-12

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = _bernoulli_contribution_at_q(
                observed_sum,
                p_group,
                mid,
            )

            if f_mid > 0:
                lo = mid
            else:
                hi = mid

            if hi - lo <= root_tol * max(1.0, abs(mid)):
                break

        return ("positive", float(0.5 * (lo + hi)))

    # residual < 0: find the non-unit root to the left of q=1.
    hi = 1.0 - 1e-12
    lo = 0.5

    f_lo = _bernoulli_contribution_at_q(
        observed_sum,
        p_group,
        lo,
    )

    while f_lo > 0 and lo > min_q:
        lo *= 0.5
        f_lo = _bernoulli_contribution_at_q(
            observed_sum,
            p_group,
            lo,
        )

    if f_lo > 0:
        return ("negative", 0.0)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _bernoulli_contribution_at_q(
            observed_sum,
            p_group,
            mid,
        )

        # score is positive near q=1 and non-positive below the root.
        if f_mid > 0:
            hi = mid
        else:
            lo = mid

        if hi - lo <= root_tol:
            break

    return ("negative", float(0.5 * (lo + hi)))


# ============================================================================
# Scanner v2
# ============================================================================

class AdaptiveStructuredScannerV2:
    """
    Hybrid adaptive search over interpretable conjunctions.

    Parameters
    ----------
    feature_types:
        Sequence containing "numeric" or "categorical" for each column.
        If None, all features are numeric.

    numeric_shortlist_per_direction:
        Number of strongest Newton-screened lower-tail and upper-tail threshold
        candidates exact-scored per numeric feature / parent beam state.

    numeric_anchor_quantiles:
        Additional distributional threshold anchors to protect against local
        approximation mismatch between Newton screening and exact gain.

    numeric_exact_pool:
        Maximum number of Newton-screened numeric thresholds exact-scored per
        feature / parent state.

    per_feature_exact_candidates:
        After exact scoring, retain this many candidate literals per feature
        before beam expansion.

    beam_width:
        Number of conjunction states retained at each depth.

    diverse_fraction:
        Fraction of the beam initially reserved for distinct last-added
        features, reducing crowd-out by near-duplicate candidates.

    max_per_feature_signature:
        Maximum number of states with the same set of restricted features.

    refine_top_n:
        Number of strongest states subjected to coordinate refinement.

    refinement_passes:
        Maximum coordinate-refinement sweeps.
    """

    def __init__(
        self,
        *,
        feature_types: Optional[Sequence[str]] = None,
        max_rule_features: int = 5,
        beam_width: int = 30,
        numeric_shortlist_per_direction: int = 8,
        numeric_anchor_quantiles: Sequence[float] = (
            0.20,
            0.40,
            0.60,
            0.80,
        ),
        numeric_exact_pool: int = 6,
        per_feature_exact_candidates: int = 2,
        diverse_fraction: float = 0.50,
        max_per_feature_signature: int = 3,
        refine_top_n: int = 4,
        refinement_passes: int = 1,
        min_size: int = 40,
        min_events: int = 3,
        min_nonevents: int = 3,
        l2: float = 0.0,
    ) -> None:
        self.feature_types = (
            None if feature_types is None else tuple(feature_types)
        )
        self.max_rule_features = int(max_rule_features)
        self.beam_width = int(beam_width)
        self.numeric_shortlist_per_direction = int(
            numeric_shortlist_per_direction
        )
        self.numeric_anchor_quantiles = tuple(
            float(q) for q in numeric_anchor_quantiles
        )
        self.numeric_exact_pool = int(numeric_exact_pool)
        self.per_feature_exact_candidates = int(
            per_feature_exact_candidates
        )
        self.diverse_fraction = float(diverse_fraction)
        self.max_per_feature_signature = int(
            max_per_feature_signature
        )
        self.refine_top_n = int(refine_top_n)
        self.refinement_passes = int(refinement_passes)
        self.min_size = int(min_size)
        self.min_events = int(min_events)
        self.min_nonevents = int(min_nonevents)
        self.l2 = float(l2)

        # Kept for compatibility with existing learner conventions.
        self.binner_ = None

        self.evaluated_exact_candidates_: int = 0
        self.screened_numeric_candidates_: int = 0

    def fit_binner(self, X: np.ndarray) -> "AdaptiveStructuredScannerV2":
        # No fixed global discretisation is required.
        X = np.asarray(X)
        if self.feature_types is None:
            self.feature_types_ = tuple(
                "numeric" for _ in range(X.shape[1])
            )
        else:
            if len(self.feature_types) != X.shape[1]:
                raise ValueError(
                    "feature_types length must equal number of columns."
                )
            bad = set(self.feature_types) - {
                "numeric",
                "categorical",
            }
            if bad:
                raise ValueError(
                    f"Unknown feature types: {sorted(bad)}"
                )
            self.feature_types_ = tuple(self.feature_types)
        return self

    # ------------------------------------------------------------------
    # Numeric literal proposals
    # ------------------------------------------------------------------

    def _numeric_candidates(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
        parent_mask: np.ndarray,
        j: int,
    ) -> List[Tuple[RuleLiteral, np.ndarray, SubgroupScore]]:
        idx = np.flatnonzero(parent_mask)

        if len(idx) < 2 * self.min_size:
            return []

        x = np.asarray(X[idx, j], dtype=float)
        yy = np.asarray(y[idx], dtype=float)
        zz = np.asarray(logits[idx], dtype=float)
        pp = sigmoid(zz)

        finite = np.isfinite(x)

        if finite.sum() < 2 * self.min_size:
            return []

        idx = idx[finite]
        x = x[finite]
        yy = yy[finite]
        pp = pp[finite]

        order = np.argsort(x, kind="mergesort")
        x = x[order]
        yy = yy[order]
        pp = pp[order]
        idx = idx[order]

        m = len(x)

        # Candidate split t means lower = [0:t], upper = [t:m].
        change = np.flatnonzero(x[:-1] < x[1:]) + 1

        if len(change) == 0:
            return []

        # Support constraints before scoring.
        change = change[
            (change >= self.min_size)
            & ((m - change) >= self.min_size)
        ]

        if len(change) == 0:
            return []

        g = pp - yy
        h = pp * (1.0 - pp)

        G = np.cumsum(g)
        H = np.cumsum(h)
        E = np.cumsum(yy)

        total_G = G[-1]
        total_H = H[-1]

        t = change

        lower_G = G[t - 1]
        lower_H = H[t - 1]

        upper_G = total_G - lower_G
        upper_H = total_H - lower_H

        lower_proxy = 0.5 * lower_G**2 / np.maximum(
            lower_H + max(self.l2, 1e-12),
            1e-12,
        )
        upper_proxy = 0.5 * upper_G**2 / np.maximum(
            upper_H + max(self.l2, 1e-12),
            1e-12,
        )

        self.screened_numeric_candidates_ += int(
            2 * len(change)
        )

        shortlist: set[Tuple[str, int]] = set()

        k = min(
            self.numeric_shortlist_per_direction,
            len(change),
        )

        if k > 0:
            low_top = np.argpartition(
                -lower_proxy,
                k - 1,
            )[:k]
            up_top = np.argpartition(
                -upper_proxy,
                k - 1,
            )[:k]

            for pos in low_top:
                shortlist.add(("le", int(pos)))
            for pos in up_top:
                shortlist.add(("gt", int(pos)))

        # Distributional anchors: nearest feasible cutpoint to requested
        # parent-subgroup quantiles.
        for q in self.numeric_anchor_quantiles:
            target = int(round(q * m))
            nearest = int(
                np.argmin(
                    np.abs(change - target)
                )
            )
            shortlist.add(("le", nearest))
            shortlist.add(("gt", nearest))

        # Rank all shortlisted proposals by the cheap Newton proxy and send
        # only a small pool to the exact likelihood scorer.
        ranked_shortlist = []

        for direction, pos in shortlist:
            proxy = (
                float(lower_proxy[pos])
                if direction == "le"
                else float(upper_proxy[pos])
            )
            ranked_shortlist.append(
                (proxy, direction, int(pos))
            )

        ranked_shortlist.sort(
            key=lambda z: z[0],
            reverse=True,
        )

        ranked_shortlist = ranked_shortlist[
            : self.numeric_exact_pool
        ]

        exact_candidates = []

        for _, direction, pos in ranked_shortlist:
            split_t = int(change[pos])
            threshold = float(
                0.5 * (
                    x[split_t - 1]
                    + x[split_t]
                )
            )

            if direction == "le":
                local_idx = idx[:split_t]
            else:
                local_idx = idx[split_t:]

            mask = np.zeros(len(X), dtype=bool)
            mask[local_idx] = True

            self.evaluated_exact_candidates_ += 1

            sc = _valid_score(
                y,
                logits,
                mask,
                min_size=self.min_size,
                min_events=self.min_events,
                min_nonevents=self.min_nonevents,
                l2=self.l2,
            )

            if sc is None:
                continue

            literal = NumericLiteral(
                feature=j,
                direction=direction,
                threshold=threshold,
            )
            exact_candidates.append(
                (literal, mask, sc)
            )

        exact_candidates.sort(
            key=lambda z: z[2].gain,
            reverse=True,
        )

        return exact_candidates[
            : self.per_feature_exact_candidates
        ]

    # ------------------------------------------------------------------
    # Categorical ALTSS-style conditional subset proposals
    # ------------------------------------------------------------------

    def _categorical_candidates(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
        parent_mask: np.ndarray,
        j: int,
    ) -> List[Tuple[RuleLiteral, np.ndarray, SubgroupScore]]:
        idx = np.flatnonzero(parent_mask)

        if len(idx) < self.min_size:
            return []

        values = np.asarray(X[idx, j])
        yy = np.asarray(y[idx], dtype=int)
        pp = sigmoid(np.asarray(logits[idx], dtype=float))

        levels = np.unique(values)

        if len(levels) <= 1:
            return []

        positive_roots = []
        negative_roots = []

        for level in levels:
            m = values == level

            interval = _category_positive_q_interval(
                yy[m],
                pp[m],
            )

            if interval is None:
                continue

            direction, root = interval

            if direction == "positive":
                positive_roots.append(
                    (float(root), float(level))
                )
            else:
                negative_roots.append(
                    (float(root), float(level))
                )

        candidate_sets: set[Tuple[float, ...]] = set()

        # q > 1: include categories whose positive interval extends beyond
        # the current q. This yields prefixes after sorting q_max descending.
        positive_roots.sort(
            key=lambda z: z[0],
            reverse=True,
        )
        ordered_pos = [
            level for _, level in positive_roots
        ]

        for k in range(1, len(ordered_pos) + 1):
            subset = tuple(
                sorted(ordered_pos[:k])
            )
            if 0 < len(subset) < len(levels):
                candidate_sets.add(subset)

        # q < 1: categories are added as q rises above their q_min roots.
        # Sorting q_min ascending gives the candidate prefixes.
        negative_roots.sort(
            key=lambda z: z[0],
        )
        ordered_neg = [
            level for _, level in negative_roots
        ]

        for k in range(1, len(ordered_neg) + 1):
            subset = tuple(
                sorted(ordered_neg[:k])
            )
            if 0 < len(subset) < len(levels):
                candidate_sets.add(subset)

        exact_candidates = []

        full_x = np.asarray(X[:, j])

        for allowed in candidate_sets:
            mask = parent_mask & np.isin(
                full_x,
                np.asarray(allowed),
            )

            self.evaluated_exact_candidates_ += 1

            sc = _valid_score(
                y,
                logits,
                mask,
                min_size=self.min_size,
                min_events=self.min_events,
                min_nonevents=self.min_nonevents,
                l2=self.l2,
            )

            if sc is None:
                continue

            literal = CategoricalLiteral(
                feature=j,
                allowed=tuple(allowed),
            )
            exact_candidates.append(
                (literal, mask, sc)
            )

        exact_candidates.sort(
            key=lambda z: z[2].gain,
            reverse=True,
        )

        return exact_candidates[
            : self.per_feature_exact_candidates
        ]

    def _feature_candidates(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
        parent_mask: np.ndarray,
        j: int,
    ) -> List[Tuple[RuleLiteral, np.ndarray, SubgroupScore]]:
        kind = self.feature_types_[j]

        if kind == "numeric":
            return self._numeric_candidates(
                X,
                y,
                logits,
                parent_mask,
                j,
            )

        return self._categorical_candidates(
            X,
            y,
            logits,
            parent_mask,
            j,
        )

    # ------------------------------------------------------------------
    # Beam diversity
    # ------------------------------------------------------------------

    @staticmethod
    def _signature(
        basis: AdaptiveRuleBasis,
    ) -> Tuple[int, ...]:
        return tuple(sorted(basis.literals))

    def _retain_diverse_beam(
        self,
        states: List[AdaptiveBeamState],
    ) -> List[AdaptiveBeamState]:
        if not states:
            return []

        states = sorted(
            states,
            key=lambda st: st.score.gain,
            reverse=True,
        )

        selected: List[AdaptiveBeamState] = []
        seen_keys = set()
        signature_counts: Dict[Tuple[int, ...], int] = {}
        last_feature_seen = set()

        reserve = min(
            self.beam_width,
            max(
                1,
                int(
                    round(
                        self.beam_width
                        * self.diverse_fraction
                    )
                ),
            ),
        )

        # Pass 1: reserve space for distinct last-added features.
        for st in states:
            if len(selected) >= reserve:
                break

            if st.last_feature in last_feature_seen:
                continue

            key = st.basis.key()
            if key in seen_keys:
                continue

            sig = self._signature(st.basis)

            selected.append(st)
            seen_keys.add(key)
            last_feature_seen.add(st.last_feature)
            signature_counts[sig] = (
                signature_counts.get(sig, 0) + 1
            )

        # Pass 2: fill globally, controlling repeated feature signatures.
        for st in states:
            if len(selected) >= self.beam_width:
                break

            key = st.basis.key()
            if key in seen_keys:
                continue

            sig = self._signature(st.basis)

            if (
                signature_counts.get(sig, 0)
                >= self.max_per_feature_signature
            ):
                continue

            selected.append(st)
            seen_keys.add(key)
            signature_counts[sig] = (
                signature_counts.get(sig, 0) + 1
            )

        return selected

    # ------------------------------------------------------------------
    # Coordinate refinement
    # ------------------------------------------------------------------

    def _mask_from_literals_except(
        self,
        X: np.ndarray,
        basis: AdaptiveRuleBasis,
        skip_feature: Optional[int] = None,
    ) -> np.ndarray:
        mask = np.ones(len(X), dtype=bool)

        for j, literal in basis.literals.items():
            if j == skip_feature:
                continue
            mask &= literal.transform(X)

        return mask

    def _refine_state(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
        state: AdaptiveBeamState,
    ) -> AdaptiveBeamState:
        basis = AdaptiveRuleBasis(
            literals=dict(state.basis.literals)
        )
        current_mask = basis.transform(X).astype(bool)
        current_score = _valid_score(
            y,
            logits,
            current_mask,
            min_size=self.min_size,
            min_events=self.min_events,
            min_nonevents=self.min_nonevents,
            l2=self.l2,
        )

        if current_score is None:
            return state

        for _ in range(self.refinement_passes):
            improved = False

            # Re-optimize each used feature conditional on all others.
            for j in list(sorted(basis.literals)):
                parent_mask = self._mask_from_literals_except(
                    X,
                    basis,
                    skip_feature=j,
                )

                proposals = self._feature_candidates(
                    X,
                    y,
                    logits,
                    parent_mask,
                    j,
                )

                if not proposals:
                    continue

                literal, mask, sc = proposals[0]

                if sc.gain > current_score.gain + 1e-10:
                    new_literals = dict(basis.literals)
                    new_literals[j] = literal
                    basis = AdaptiveRuleBasis(new_literals)
                    current_mask = mask
                    current_score = sc
                    improved = True

            # If room remains, try the strongest additional feature.
            if len(basis.literals) < self.max_rule_features:
                best_add = None

                for j in range(X.shape[1]):
                    if j in basis.literals:
                        continue

                    proposals = self._feature_candidates(
                        X,
                        y,
                        logits,
                        current_mask,
                        j,
                    )

                    if not proposals:
                        continue

                    cand = proposals[0]

                    if (
                        best_add is None
                        or cand[2].gain > best_add[3].gain
                    ):
                        best_add = (
                            j,
                            cand[0],
                            cand[1],
                            cand[2],
                        )

                if (
                    best_add is not None
                    and best_add[3].gain
                    > current_score.gain + 1e-10
                ):
                    j, literal, mask, sc = best_add
                    new_literals = dict(basis.literals)
                    new_literals[j] = literal
                    basis = AdaptiveRuleBasis(new_literals)
                    current_mask = mask
                    current_score = sc
                    improved = True

            if not improved:
                break

        return AdaptiveBeamState(
            basis=basis,
            mask=current_mask,
            score=current_score,
            last_feature=state.last_feature,
        )

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def scan(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
    ) -> AdaptiveScanResult:
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        logits = np.asarray(logits, dtype=float)

        self.fit_binner(X)

        self.evaluated_exact_candidates_ = 0
        self.screened_numeric_candidates_ = 0

        n, p = X.shape

        beam = [
            AdaptiveBeamState(
                basis=AdaptiveRuleBasis({}),
                mask=np.ones(n, dtype=bool),
                score=None,  # type: ignore[arg-type]
                last_feature=None,
            )
        ]

        best_by_depth: List[AdaptiveBeamState] = []
        all_final_candidates: List[AdaptiveBeamState] = []

        for depth in range(
            1,
            self.max_rule_features + 1,
        ):
            children: Dict[
                Tuple,
                AdaptiveBeamState,
            ] = {}

            for parent in beam:
                used = set(
                    parent.basis.literals
                )

                for j in range(p):
                    if j in used:
                        continue

                    proposals = self._feature_candidates(
                        X,
                        y,
                        logits,
                        parent.mask,
                        j,
                    )

                    for literal, mask, sc in proposals:
                        new_literals = dict(
                            parent.basis.literals
                        )
                        new_literals[j] = literal

                        basis = AdaptiveRuleBasis(
                            new_literals
                        )
                        key = basis.key()

                        state = AdaptiveBeamState(
                            basis=basis,
                            mask=mask,
                            score=sc,
                            last_feature=j,
                        )

                        old = children.get(key)

                        if (
                            old is None
                            or sc.gain > old.score.gain
                        ):
                            children[key] = state

            if not children:
                break

            beam = self._retain_diverse_beam(
                list(children.values())
            )

            if not beam:
                break

            best_by_depth.append(beam[0])
            all_final_candidates.extend(beam)

        if not all_final_candidates:
            raise RuntimeError(
                "No valid adaptive structured subgroup found."
            )

        # Refine strongest distinct states from across depths.
        ranked = sorted(
            all_final_candidates,
            key=lambda st: st.score.gain,
            reverse=True,
        )

        refine_pool = []
        seen = set()

        for st in ranked:
            key = st.basis.key()
            if key in seen:
                continue
            seen.add(key)
            refine_pool.append(st)

            if len(refine_pool) >= self.refine_top_n:
                break

        refined_states = [
            self._refine_state(
                X,
                y,
                logits,
                st,
            )
            for st in refine_pool
        ]

        candidates = ranked + refined_states

        best = max(
            candidates,
            key=lambda st: st.score.gain,
        )

        refined = any(
            best.basis.key() == st.basis.key()
            and best.score.gain >= st.score.gain
            for st in refined_states
        )

        return AdaptiveScanResult(
            gain=float(best.score.gain),
            delta=float(best.score.delta),
            basis=best.basis,
            mask=best.mask,
            score=best.score,
            depth=best.basis.n_features,
            evaluated_exact_candidates=int(
                self.evaluated_exact_candidates_
            ),
            screened_numeric_candidates=int(
                self.screened_numeric_candidates_
            ),
            refined=bool(refined),
        )


# ============================================================================
# Scan-only cross-fitted repair learner
# ============================================================================

@dataclass
class AdaptiveRepairRound:
    round_index: int
    discovery_gain: float
    selection_gain: float
    selected_delta: float
    active_terms: int
    rule_depth: int
    evaluated_exact_candidates: int
    screened_numeric_candidates: int


class AdaptiveScanRepairLearner:
    """
    Discovery -> independent selection -> full coefficient refit.

    This mirrors the existing strict scan-only learner but accepts the new
    AdaptiveRuleBasis instead of requiring ScanBasis / fixed bins.
    """

    def __init__(
        self,
        scanner: AdaptiveStructuredScannerV2,
        *,
        repair_rounds: int = 4,
        coefficient_l2: float = 1e-3,
        selection_gain_tolerance: float = 1.9207,
        selection_min_size: int = 25,
        selection_min_events: int = 2,
        selection_min_nonevents: int = 2,
    ) -> None:
        self.scanner = scanner
        self.repair_rounds = int(repair_rounds)
        self.coefficient_l2 = float(coefficient_l2)
        self.selection_gain_tolerance = float(
            selection_gain_tolerance
        )
        self.selection_min_size = int(selection_min_size)
        self.selection_min_events = int(selection_min_events)
        self.selection_min_nonevents = int(selection_min_nonevents)

        self.bases_: List[AdaptiveRuleBasis] = []
        self.coefficients_: np.ndarray = np.empty(0)
        self.history_: List[AdaptiveRepairRound] = []

    def _Phi(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if not self.bases_:
            return np.empty(
                (len(X), 0),
                dtype=float,
            )

        return np.column_stack(
            [
                b.transform(X)
                for b in self.bases_
            ]
        )

    def repair_logits(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
    ) -> np.ndarray:
        z = np.asarray(
            base_logits,
            dtype=float,
        )

        if not self.bases_:
            return z.copy()

        return (
            z
            + self._Phi(X)
            @ self.coefficients_
        )

    def fit(
        self,
        X_disc: np.ndarray,
        y_disc: np.ndarray,
        base_logits_disc: np.ndarray,
        X_sel: np.ndarray,
        y_sel: np.ndarray,
        base_logits_sel: np.ndarray,
    ) -> "AdaptiveScanRepairLearner":
        X_disc = np.asarray(X_disc)
        y_disc = np.asarray(y_disc, dtype=int)
        z0_disc = np.asarray(
            base_logits_disc,
            dtype=float,
        )

        X_sel = np.asarray(X_sel)
        y_sel = np.asarray(y_sel, dtype=int)
        z0_sel = np.asarray(
            base_logits_sel,
            dtype=float,
        )

        self.bases_ = []
        self.coefficients_ = np.empty(0)
        self.history_ = []

        for t in range(self.repair_rounds):
            z_disc = self.repair_logits(
                X_disc,
                z0_disc,
            )
            z_sel = self.repair_logits(
                X_sel,
                z0_sel,
            )

            result = self.scanner.scan(
                X_disc,
                y_disc,
                z_disc,
            )

            basis = result.basis
            sel_mask = basis.transform(
                X_sel
            ).astype(bool)

            sel_score = _valid_score(
                y_sel,
                z_sel,
                sel_mask,
                min_size=self.selection_min_size,
                min_events=self.selection_min_events,
                min_nonevents=self.selection_min_nonevents,
                l2=0.0,
            )

            if sel_score is None:
                break

            if (
                not np.isfinite(
                    sel_score.gain
                )
                or sel_score.gain
                < self.selection_gain_tolerance
            ):
                break

            self.bases_.append(basis)

            Phi_sel = self._Phi(X_sel)

            init = np.concatenate(
                [
                    self.coefficients_,
                    np.array(
                        [sel_score.delta],
                        dtype=float,
                    ),
                ]
            )

            fit = fit_residual_dictionary_with_offset(
                y_sel,
                z0_sel,
                Phi_sel,
                l2=self.coefficient_l2,
                init=init,
            )

            self.coefficients_ = fit.coefficients

            self.history_.append(
                AdaptiveRepairRound(
                    round_index=t + 1,
                    discovery_gain=float(
                        result.gain
                    ),
                    selection_gain=float(
                        sel_score.gain
                    ),
                    selected_delta=float(
                        sel_score.delta
                    ),
                    active_terms=len(
                        self.bases_
                    ),
                    rule_depth=int(
                        result.depth
                    ),
                    evaluated_exact_candidates=int(
                        result.evaluated_exact_candidates
                    ),
                    screened_numeric_candidates=int(
                        result.screened_numeric_candidates
                    ),
                )
            )

        return self
