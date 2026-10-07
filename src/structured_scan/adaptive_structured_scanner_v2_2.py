
"""
Structured Scan v2.2
====================

Scanner v2.2 addresses the main weakness revealed by v2.1:

    stronger DISCOVERY search -> higher empirical gain
    but sometimes worse recovery/generalization.

The fix is cross-sample candidate selection.

Per repair round:
    1. Adaptive scanner proposes top-K diverse exact-likelihood rules.
    2. A protected v1-style fixed-bin rule is also proposed.
    3. All unique candidates are evaluated on independent SELECTION data.
    4. SELECTION chooses the candidate with the largest exact likelihood gain.
    5. Acceptance uses a within-round Bonferroni/chi-square gate:

           tau_K = 0.5 * chi2.ppf(1 - alpha/K, df=1)

       where K is the number of valid candidates evaluated that round.

       For K=1 and alpha=.05:
           tau_1 = 1.920729...
       i.e. the original strict gate.

    6. Accepted rule coefficients are fully refit jointly on SELECTION.

The exact Bernoulli/logistic likelihood objective is unchanged.

Important:
    v2.2 is not a globally exact conjunction optimizer.
    It is a stronger candidate-generation + independent candidate-selection
    procedure designed to reduce winner's-curse behaviour from DISCOVERY.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
from scipy.stats import chi2

from mdss_treescan_stage1 import (
    SubgroupScore,
    exact_subgroup_logit_correction,
)
from residual_repair_benchmark import (
    fit_residual_dictionary_with_offset,
)
from structured_mdss import StructuredBeamMDSS
from treescan_prototype import ScanBasis

from adaptive_structured_scanner_v2_1 import (
    AdaptiveStructuredScannerV2,
    AdaptiveRuleBasis,
    AdaptiveBeamState,
    AdaptiveScanResult,
)


# ============================================================================
# Candidate structures
# ============================================================================

@dataclass
class ProposedRule:
    basis: Any
    source: str
    discovery_gain: float
    discovery_delta: float
    discovery_depth: int
    discovery_mask: np.ndarray
    refined: bool = False


@dataclass
class SelectionCandidate:
    proposal: ProposedRule
    selection_score: SubgroupScore
    selection_mask: np.ndarray


@dataclass
class CandidateSelectionRound:
    round_index: int
    proposed_candidates: int
    valid_selection_candidates: int
    multiplicity_gate: float
    selected_source: Optional[str]
    selected_discovery_gain: float
    selected_selection_gain: float
    selected_delta: float
    selected_rule_depth: int
    active_terms: int
    adaptive_candidates_proposed: int
    legacy_candidate_proposed: int
    evaluated_exact_candidates: int
    screened_numeric_candidates: int


# ============================================================================
# Utility
# ============================================================================

def mask_jaccard(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    union = np.sum(a | b)

    if union == 0:
        return 1.0

    return float(
        np.sum(a & b) / union
    )


def within_round_multiplicity_gate(
    n_candidates: int,
    *,
    alpha: float = 0.05,
) -> float:
    """
    Bonferroni-adjusted fixed-rule likelihood-ratio gate.

    2 * DeltaL approximately follows chi-square_1 for one independently
    specified scalar correction. Candidate rules are fixed by DISCOVERY before
    being compared on SELECTION.

    This controls multiplicity among the K candidates examined in a single
    selection round. It does not claim global family-wise control across all
    stagewise repair rounds.
    """
    K = max(
        1,
        int(n_candidates),
    )

    return float(
        0.5
        * chi2.ppf(
            1.0 - alpha / K,
            df=1,
        )
    )


# ============================================================================
# Adaptive candidate-pool scanner
# ============================================================================

class AdaptiveStructuredScannerV22(
    AdaptiveStructuredScannerV2
):
    """
    Extends v2.1 by exposing a diverse shortlist instead of only the single
    highest DISCOVERY rule.
    """

    def __init__(
        self,
        *,
        candidate_pool_size: int = 4,
        candidate_mask_jaccard_max: float = 0.90,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.candidate_pool_size = int(
            candidate_pool_size
        )
        self.candidate_mask_jaccard_max = float(
            candidate_mask_jaccard_max
        )

    def _search_states(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
    ) -> Tuple[
        List[AdaptiveBeamState],
        List[AdaptiveBeamState],
    ]:
        """
        Return:
            ranked distinct states,
            refined states.
        """
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
                mask=np.ones(
                    n,
                    dtype=bool,
                ),
                score=None,  # type: ignore[arg-type]
                last_feature=None,
            )
        ]

        all_candidates: List[
            AdaptiveBeamState
        ] = []

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

                    proposals = (
                        self._feature_candidates(
                            X,
                            y,
                            logits,
                            parent.mask,
                            j,
                        )
                    )

                    for (
                        literal,
                        mask,
                        score,
                    ) in proposals:
                        literals = dict(
                            parent.basis.literals
                        )
                        literals[j] = literal

                        basis = AdaptiveRuleBasis(
                            literals
                        )
                        key = basis.key()

                        state = AdaptiveBeamState(
                            basis=basis,
                            mask=mask,
                            score=score,
                            last_feature=j,
                        )

                        old = children.get(
                            key
                        )

                        if (
                            old is None
                            or score.gain
                            > old.score.gain
                        ):
                            children[key] = (
                                state
                            )

            if not children:
                break

            beam = (
                self._retain_diverse_beam(
                    list(
                        children.values()
                    )
                )
            )

            if not beam:
                break

            all_candidates.extend(
                beam
            )

        if not all_candidates:
            raise RuntimeError(
                "No valid adaptive structured subgroup found."
            )

        # Deduplicate exact bases, keeping the highest-gain realization.
        by_key: Dict[
            Tuple,
            AdaptiveBeamState,
        ] = {}

        for st in all_candidates:
            key = st.basis.key()
            old = by_key.get(key)

            if (
                old is None
                or st.score.gain
                > old.score.gain
            ):
                by_key[key] = st

        ranked = sorted(
            by_key.values(),
            key=lambda st: st.score.gain,
            reverse=True,
        )

        # Refine the strongest states as in v2.1.
        refine_pool = ranked[
            : self.refine_top_n
        ]

        refined_states = [
            self._refine_state(
                X,
                y,
                logits,
                st,
            )
            for st in refine_pool
        ]

        return ranked, refined_states

    def scan_candidates(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
        *,
        n_candidates: Optional[int] = None,
    ) -> List[AdaptiveScanResult]:
        """
        Return a compact, mask-diverse candidate shortlist.

        Candidate generation uses DISCOVERY data only.
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        logits = np.asarray(logits, dtype=float)

        ranked, refined = self._search_states(
            X,
            y,
            logits,
        )

        # Merge original and refined states, deduplicating by basis.
        by_key: Dict[
            Tuple,
            AdaptiveBeamState,
        ] = {}

        for st in ranked + refined:
            key = st.basis.key()
            old = by_key.get(key)

            if (
                old is None
                or st.score.gain
                > old.score.gain
            ):
                by_key[key] = st

        states = sorted(
            by_key.values(),
            key=lambda st: st.score.gain,
            reverse=True,
        )

        K = (
            self.candidate_pool_size
            if n_candidates is None
            else int(n_candidates)
        )

        selected: List[
            AdaptiveBeamState
        ] = []

        # Pass 1: always keep the best state.
        if states:
            selected.append(
                states[0]
            )

        # Pass 2: favor different depths and sufficiently different masks.
        seen_depths = {
            states[0].basis.n_features
        } if states else set()

        for st in states[1:]:
            if len(selected) >= K:
                break

            depth = st.basis.n_features

            if depth in seen_depths:
                continue

            if all(
                mask_jaccard(
                    st.mask,
                    old.mask,
                )
                <= self.candidate_mask_jaccard_max
                for old in selected
            ):
                selected.append(st)
                seen_depths.add(depth)

        # Pass 3: fill remaining capacity by mask diversity regardless of depth.
        for st in states:
            if len(selected) >= K:
                break

            if any(
                st.basis.key()
                == old.basis.key()
                for old in selected
            ):
                continue

            if all(
                mask_jaccard(
                    st.mask,
                    old.mask,
                )
                <= self.candidate_mask_jaccard_max
                for old in selected
            ):
                selected.append(st)

        # Pass 4: if diversity constraints leave unused slots, fill by gain.
        for st in states:
            if len(selected) >= K:
                break

            if any(
                st.basis.key()
                == old.basis.key()
                for old in selected
            ):
                continue

            selected.append(st)

        refined_keys = {
            st.basis.key()
            for st in refined
        }

        results = []

        for st in selected:
            results.append(
                AdaptiveScanResult(
                    gain=float(
                        st.score.gain
                    ),
                    delta=float(
                        st.score.delta
                    ),
                    basis=st.basis,
                    mask=st.mask,
                    score=st.score,
                    depth=st.basis.n_features,
                    evaluated_exact_candidates=int(
                        self.evaluated_exact_candidates_
                    ),
                    screened_numeric_candidates=int(
                        self.screened_numeric_candidates_
                    ),
                    refined=(
                        st.basis.key()
                        in refined_keys
                    ),
                )
            )

        return results

    def scan(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
    ) -> AdaptiveScanResult:
        return self.scan_candidates(
            X,
            y,
            logits,
            n_candidates=1,
        )[0]


# ============================================================================
# Protected v1 candidate
# ============================================================================

def propose_legacy_fixed_bin_rule(
    X: np.ndarray,
    y: np.ndarray,
    logits: np.ndarray,
    *,
    max_rule_features: int,
    beam_width: int,
    min_size: int,
    min_events: int,
    min_nonevents: int,
    n_bins: int = 5,
) -> ProposedRule:
    """
    Protected simple-path candidate from the original v1 scanner.
    """
    scanner = StructuredBeamMDSS(
        n_bins=n_bins,
        max_rule_features=max_rule_features,
        beam_width=beam_width,
        literal_mode="tails",
        min_size=min_size,
        min_events=min_events,
        min_nonevents=min_nonevents,
    )

    result = scanner.scan(
        X,
        y,
        logits,
    )

    basis = ScanBasis(
        restrictions=dict(
            result.restrictions
        ),
        binner=scanner.binner_,
    )

    return ProposedRule(
        basis=basis,
        source="legacy_fixed5",
        discovery_gain=float(
            result.gain
        ),
        discovery_delta=float(
            result.delta
        ),
        discovery_depth=int(
            result.depth
        ),
        discovery_mask=np.asarray(
            result.mask,
            dtype=bool,
        ),
        refined=False,
    )


# ============================================================================
# Candidate-selected repair learner
# ============================================================================

class CandidateSelectedScanRepairLearner:
    """
    v2.2 stagewise repair.

    Adaptive top-K + protected legacy candidate are generated on DISCOVERY.
    Independent SELECTION chooses among them using exact likelihood gain.
    """

    def __init__(
        self,
        scanner: AdaptiveStructuredScannerV22,
        *,
        repair_rounds: int = 4,
        adaptive_candidates: int = 4,
        include_legacy_candidate: bool = True,
        legacy_beam_width: int = 20,
        legacy_n_bins: int = 5,
        coefficient_l2: float = 1e-3,
        selection_alpha: float = 0.05,
        selection_min_size: int = 25,
        selection_min_events: int = 2,
        selection_min_nonevents: int = 2,
    ) -> None:
        self.scanner = scanner
        self.repair_rounds = int(
            repair_rounds
        )
        self.adaptive_candidates = int(
            adaptive_candidates
        )
        self.include_legacy_candidate = bool(
            include_legacy_candidate
        )
        self.legacy_beam_width = int(
            legacy_beam_width
        )
        self.legacy_n_bins = int(
            legacy_n_bins
        )
        self.coefficient_l2 = float(
            coefficient_l2
        )
        self.selection_alpha = float(
            selection_alpha
        )
        self.selection_min_size = int(
            selection_min_size
        )
        self.selection_min_events = int(
            selection_min_events
        )
        self.selection_min_nonevents = int(
            selection_min_nonevents
        )

        self.bases_: List[Any] = []
        self.coefficients_ = np.empty(
            0,
            dtype=float,
        )
        self.history_: List[
            CandidateSelectionRound
        ] = []

        self.last_round_candidates_: List[
            SelectionCandidate
        ] = []

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
                basis.transform(X)
                for basis in self.bases_
            ]
        ).astype(float)

    def repair_logits(
        self,
        X: np.ndarray,
        base_logits: np.ndarray,
    ) -> np.ndarray:
        z0 = np.asarray(
            base_logits,
            dtype=float,
        )

        if not self.bases_:
            return z0.copy()

        return (
            z0
            + self._Phi(X)
            @ self.coefficients_
        )

    @staticmethod
    def _mask_key(
        mask: np.ndarray,
    ) -> bytes:
        return np.packbits(
            np.asarray(
                mask,
                dtype=np.uint8,
            )
        ).tobytes()

    def _active_selection_mask_keys(
        self,
        X_sel: np.ndarray,
    ) -> set[bytes]:
        return {
            self._mask_key(
                basis.transform(
                    X_sel
                ).astype(bool)
            )
            for basis in self.bases_
        }

    def _propose(
        self,
        X_disc: np.ndarray,
        y_disc: np.ndarray,
        z_disc: np.ndarray,
    ) -> Tuple[
        List[ProposedRule],
        int,
        int,
    ]:
        adaptive = self.scanner.scan_candidates(
            X_disc,
            y_disc,
            z_disc,
            n_candidates=self.adaptive_candidates,
        )

        proposals = [
            ProposedRule(
                basis=res.basis,
                source="adaptive",
                discovery_gain=float(
                    res.gain
                ),
                discovery_delta=float(
                    res.delta
                ),
                discovery_depth=int(
                    res.depth
                ),
                discovery_mask=np.asarray(
                    res.mask,
                    dtype=bool,
                ),
                refined=bool(
                    res.refined
                ),
            )
            for res in adaptive
        ]

        n_adaptive = len(
            proposals
        )
        n_legacy = 0

        if self.include_legacy_candidate:
            try:
                legacy = (
                    propose_legacy_fixed_bin_rule(
                        X_disc,
                        y_disc,
                        z_disc,
                        max_rule_features=(
                            self.scanner.max_rule_features
                        ),
                        beam_width=(
                            self.legacy_beam_width
                        ),
                        min_size=(
                            self.scanner.min_size
                        ),
                        min_events=(
                            self.scanner.min_events
                        ),
                        min_nonevents=(
                            self.scanner.min_nonevents
                        ),
                        n_bins=(
                            self.legacy_n_bins
                        ),
                    )
                )
                proposals.append(
                    legacy
                )
                n_legacy = 1
            except RuntimeError:
                pass

        # Deduplicate by DISCOVERY membership, preserving the stronger
        # discovery proposal. If identical adaptive/legacy masks occur, keep
        # the source with larger discovery gain.
        by_mask: Dict[
            bytes,
            ProposedRule,
        ] = {}

        for prop in proposals:
            key = self._mask_key(
                prop.discovery_mask
            )
            old = by_mask.get(
                key
            )

            if (
                old is None
                or prop.discovery_gain
                > old.discovery_gain
            ):
                by_mask[key] = prop

        unique = sorted(
            by_mask.values(),
            key=lambda p: p.discovery_gain,
            reverse=True,
        )

        return (
            unique,
            n_adaptive,
            n_legacy,
        )

    def fit(
        self,
        X_disc: np.ndarray,
        y_disc: np.ndarray,
        base_logits_disc: np.ndarray,
        X_sel: np.ndarray,
        y_sel: np.ndarray,
        base_logits_sel: np.ndarray,
    ) -> "CandidateSelectedScanRepairLearner":
        X_disc = np.asarray(
            X_disc
        )
        y_disc = np.asarray(
            y_disc,
            dtype=int,
        )
        z0_disc = np.asarray(
            base_logits_disc,
            dtype=float,
        )

        X_sel = np.asarray(
            X_sel
        )
        y_sel = np.asarray(
            y_sel,
            dtype=int,
        )
        z0_sel = np.asarray(
            base_logits_sel,
            dtype=float,
        )

        self.bases_ = []
        self.coefficients_ = np.empty(
            0,
            dtype=float,
        )
        self.history_ = []
        self.last_round_candidates_ = []

        for t in range(
            self.repair_rounds
        ):
            z_disc = self.repair_logits(
                X_disc,
                z0_disc,
            )
            z_sel = self.repair_logits(
                X_sel,
                z0_sel,
            )

            proposals, n_adaptive, n_legacy = (
                self._propose(
                    X_disc,
                    y_disc,
                    z_disc,
                )
            )

            active_keys = (
                self._active_selection_mask_keys(
                    X_sel
                )
            )

            valid: List[
                SelectionCandidate
            ] = []

            seen_selection_masks = set()

            for prop in proposals:
                sel_mask = (
                    prop.basis.transform(
                        X_sel
                    ).astype(bool)
                )

                key = self._mask_key(
                    sel_mask
                )

                if (
                    key in seen_selection_masks
                    or key in active_keys
                ):
                    continue

                seen_selection_masks.add(
                    key
                )

                score = (
                    exact_subgroup_logit_correction(
                        y_sel,
                        z_sel,
                        sel_mask,
                        l2=0.0,
                        min_size=(
                            self.selection_min_size
                        ),
                        min_events=(
                            self.selection_min_events
                        ),
                        min_nonevents=(
                            self.selection_min_nonevents
                        ),
                    )
                )

                if score is None:
                    continue

                valid.append(
                    SelectionCandidate(
                        proposal=prop,
                        selection_score=score,
                        selection_mask=sel_mask,
                    )
                )

            self.last_round_candidates_ = (
                valid
            )

            K = len(
                valid
            )

            if K == 0:
                self.history_.append(
                    CandidateSelectionRound(
                        round_index=t + 1,
                        proposed_candidates=len(
                            proposals
                        ),
                        valid_selection_candidates=0,
                        multiplicity_gate=np.nan,
                        selected_source=None,
                        selected_discovery_gain=np.nan,
                        selected_selection_gain=np.nan,
                        selected_delta=np.nan,
                        selected_rule_depth=0,
                        active_terms=len(
                            self.bases_
                        ),
                        adaptive_candidates_proposed=n_adaptive,
                        legacy_candidate_proposed=n_legacy,
                        evaluated_exact_candidates=int(
                            self.scanner.evaluated_exact_candidates_
                        ),
                        screened_numeric_candidates=int(
                            self.scanner.screened_numeric_candidates_
                        ),
                    )
                )
                break

            gate = (
                within_round_multiplicity_gate(
                    K,
                    alpha=self.selection_alpha,
                )
            )

            best = max(
                valid,
                key=lambda c: (
                    c.selection_score.gain
                ),
            )

            if (
                not np.isfinite(
                    best.selection_score.gain
                )
                or best.selection_score.gain
                < gate
            ):
                self.history_.append(
                    CandidateSelectionRound(
                        round_index=t + 1,
                        proposed_candidates=len(
                            proposals
                        ),
                        valid_selection_candidates=K,
                        multiplicity_gate=float(
                            gate
                        ),
                        selected_source=None,
                        selected_discovery_gain=float(
                            best.proposal.discovery_gain
                        ),
                        selected_selection_gain=float(
                            best.selection_score.gain
                        ),
                        selected_delta=float(
                            best.selection_score.delta
                        ),
                        selected_rule_depth=int(
                            best.proposal.discovery_depth
                        ),
                        active_terms=len(
                            self.bases_
                        ),
                        adaptive_candidates_proposed=n_adaptive,
                        legacy_candidate_proposed=n_legacy,
                        evaluated_exact_candidates=int(
                            self.scanner.evaluated_exact_candidates_
                        ),
                        screened_numeric_candidates=int(
                            self.scanner.screened_numeric_candidates_
                        ),
                    )
                )
                break

            self.bases_.append(
                best.proposal.basis
            )

            Phi_sel = self._Phi(
                X_sel
            )

            init = np.concatenate(
                [
                    self.coefficients_,
                    np.array(
                        [
                            best.selection_score.delta
                        ],
                        dtype=float,
                    ),
                ]
            )

            fit = (
                fit_residual_dictionary_with_offset(
                    y_sel,
                    z0_sel,
                    Phi_sel,
                    l2=self.coefficient_l2,
                    init=init,
                )
            )

            self.coefficients_ = (
                fit.coefficients
            )

            self.history_.append(
                CandidateSelectionRound(
                    round_index=t + 1,
                    proposed_candidates=len(
                        proposals
                    ),
                    valid_selection_candidates=K,
                    multiplicity_gate=float(
                        gate
                    ),
                    selected_source=(
                        best.proposal.source
                    ),
                    selected_discovery_gain=float(
                        best.proposal.discovery_gain
                    ),
                    selected_selection_gain=float(
                        best.selection_score.gain
                    ),
                    selected_delta=float(
                        best.selection_score.delta
                    ),
                    selected_rule_depth=int(
                        best.proposal.discovery_depth
                    ),
                    active_terms=len(
                        self.bases_
                    ),
                    adaptive_candidates_proposed=n_adaptive,
                    legacy_candidate_proposed=n_legacy,
                    evaluated_exact_candidates=int(
                        self.scanner.evaluated_exact_candidates_
                    ),
                    screened_numeric_candidates=int(
                        self.scanner.screened_numeric_candidates_
                    ),
                )
            )

        return self
