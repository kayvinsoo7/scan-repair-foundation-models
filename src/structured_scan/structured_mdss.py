
"""
Structured beam MDSS oracle.

Motivation
----------
The initial coordinate-ascent scanner can fail after a strong tree model has
removed low-order residual structure. It also overfits if continuous variables
are allowed arbitrary disconnected unions of bins.

This oracle uses a more explicit subgroup class for continuous variables:

    tail mode:
        X_j <= threshold
        X_j > threshold

    interval mode:
        any contiguous interval of discretised bins

and searches conjunctions with beam search up to K restricted features.

This is deliberately aligned with the paper's structured subgroup family:
interpretable conjunctions of simple literals, not arbitrary disconnected
sets on continuous variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mdss_treescan_stage1 import (
    BinningModel,
    SubgroupScore,
    exact_subgroup_logit_correction,
    fit_quantile_binner,
)


def candidate_bin_subsets(
    levels: Sequence[int],
    *,
    mode: str = "tails",
) -> List[Tuple[int, ...]]:
    """
    Return proper non-empty candidate subsets for an ordered binned feature.

    mode="tails":
        lower and upper threshold sets only.

    mode="intervals":
        every contiguous proper interval.

    The full set is intentionally excluded because adding it does not create
    a feature restriction.
    """
    levels = tuple(sorted(int(v) for v in levels))
    K = len(levels)

    if K <= 1:
        return []

    candidates = set()

    if mode == "tails":
        for r in range(1, K):
            candidates.add(tuple(levels[:r]))  # lower tail
            candidates.add(tuple(levels[r:]))  # upper tail

    elif mode == "intervals":
        for a in range(K):
            for b in range(a, K):
                sub = tuple(levels[a:b+1])
                if 0 < len(sub) < K:
                    candidates.add(sub)
    else:
        raise ValueError("mode must be 'tails' or 'intervals'")

    return sorted(candidates, key=lambda z: (len(z), z))


@dataclass
class BeamState:
    restrictions: Dict[int, Tuple[int, ...]]
    mask: np.ndarray
    score: SubgroupScore


@dataclass
class BeamMDSSResult:
    gain: float
    delta: float
    restrictions: Dict[int, Tuple[int, ...]]
    mask: np.ndarray
    score: SubgroupScore
    depth: int
    evaluated_candidates: int


class StructuredBeamMDSS:
    """
    Beam search over conjunctions of simple binned literals.

    At depth r, each beam state has r restricted features. We expand it by one
    new feature/literal, score every child by exact Bernoulli/logistic gain,
    deduplicate equivalent rules, and retain the top `beam_width`.

    This is not claimed to be a globally exact combinatorial search. Its purpose
    is to avoid the single-coordinate entry barrier that can affect coordinate
    ascent after low-order structure has already been removed by the base model.
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
        l2: float = 0.0,
    ) -> None:
        self.n_bins = int(n_bins)
        self.max_rule_features = int(max_rule_features)
        self.beam_width = int(beam_width)
        self.literal_mode = literal_mode
        self.min_size = int(min_size)
        self.min_events = int(min_events)
        self.min_nonevents = int(min_nonevents)
        self.l2 = float(l2)

        self.binner_: Optional[BinningModel] = None

    def fit_binner(self, X: np.ndarray) -> "StructuredBeamMDSS":
        self.binner_ = fit_quantile_binner(X, self.n_bins)
        return self

    @staticmethod
    def _rule_key(restrictions: Dict[int, Tuple[int, ...]]) -> Tuple:
        return tuple(
            (int(j), tuple(int(v) for v in vals))
            for j, vals in sorted(restrictions.items())
        )

    def scan(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
    ) -> BeamMDSSResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        logits = np.asarray(logits, dtype=float)

        if self.binner_ is None:
            self.fit_binner(X)

        assert self.binner_ is not None
        Xb = self.binner_.transform(X)
        n, p = Xb.shape

        levels = [
            tuple(int(v) for v in np.unique(Xb[:, j]))
            for j in range(p)
        ]
        literals = [
            candidate_bin_subsets(levels[j], mode=self.literal_mode)
            for j in range(p)
        ]

        best_global: Optional[BeamState] = None
        best_depth = 0
        evaluated = 0

        # Beam at depth zero is represented by an implicit empty rule.
        beam_rules: List[Tuple[Dict[int, Tuple[int, ...]], np.ndarray]] = [
            ({}, np.ones(n, dtype=bool))
        ]

        for depth in range(1, self.max_rule_features + 1):
            child_states: Dict[Tuple, BeamState] = {}

            for restrictions, parent_mask in beam_rules:
                used = set(restrictions.keys())

                for j in range(p):
                    if j in used:
                        continue

                    for allowed in literals[j]:
                        mask = parent_mask & np.isin(Xb[:, j], allowed)

                        # Fast support rejection before Newton scoring.
                        n_mask = int(mask.sum())
                        if n_mask < self.min_size:
                            continue

                        evaluated += 1
                        sc = exact_subgroup_logit_correction(
                            y,
                            logits,
                            mask,
                            l2=self.l2,
                            min_size=self.min_size,
                            min_events=self.min_events,
                            min_nonevents=self.min_nonevents,
                        )
                        if sc is None:
                            continue

                        new_restrictions = dict(restrictions)
                        new_restrictions[j] = tuple(allowed)
                        key = self._rule_key(new_restrictions)

                        old = child_states.get(key)
                        if old is None or sc.gain > old.score.gain:
                            child_states[key] = BeamState(
                                restrictions=new_restrictions,
                                mask=mask,
                                score=sc,
                            )

            if not child_states:
                break

            ranked = sorted(
                child_states.values(),
                key=lambda st: st.score.gain,
                reverse=True,
            )

            beam = ranked[: self.beam_width]

            if best_global is None or beam[0].score.gain > best_global.score.gain:
                best_global = beam[0]
                best_depth = depth

            beam_rules = [
                (state.restrictions, state.mask)
                for state in beam
            ]

        if best_global is None:
            raise RuntimeError("No valid structured subgroup found.")

        return BeamMDSSResult(
            gain=float(best_global.score.gain),
            delta=float(best_global.score.delta),
            restrictions=dict(best_global.restrictions),
            mask=best_global.mask,
            score=best_global.score,
            depth=best_depth,
            evaluated_candidates=evaluated,
        )

    def describe_restrictions(
        self,
        result: BeamMDSSResult,
        feature_names: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if self.binner_ is None:
            raise RuntimeError("Binner not fitted.")

        lines = []
        for j, bins in sorted(result.restrictions.items()):
            name = feature_names[j] if feature_names is not None else f"X{j+1}"
            e = self.binner_.edges[j]

            lo_bin = min(bins)
            hi_bin = max(bins)
            lo = e[lo_bin]
            hi = e[hi_bin + 1]

            # Because beam literals are contiguous/tail subsets, one interval
            # is sufficient to describe them.
            lines.append(f"{name} in [{lo:.3f}, {hi:.3f})")

        return lines
