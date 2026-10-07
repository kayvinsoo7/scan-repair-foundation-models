
"""
Structured Scan v2.3
====================

v2.3 keeps the v2.2 candidate-generation machinery unchanged and replaces
multiplicity-adjusted candidate selection with stability selection.

Per repair round
----------------
1. DISCOVERY proposes:
      - top-K diverse adaptive candidates;
      - protected v1 fixed-five-bin candidate.

2. SELECTION is split stratifiably into two deterministic halves A and B.

3. For every candidate, compute exact likelihood correction on both halves:
      Delta_A, delta_A, p_A
      Delta_B, delta_B, p_B

4. A candidate is "stable" if:
      - both half-sample scores are valid;
      - both gains are > 0;
      - sign(delta_A) == sign(delta_B).

5. Stable candidates are ranked by:
      stability_score =
          min(Delta_A / n_A, Delta_B / n_B)

   This rewards a correction that reproduces across both halves rather than a
   rule that is excellent in only one validation half.

6. The winning stable candidate is re-evaluated on the FULL SELECTION set.
   Its nominal likelihood-ratio p-value is

      p_full = P(ChiSq_1 >= 2 * Delta_full).

   Acceptance criterion:
      p_full < alpha  (default 0.05)

Important
---------
The full-selection p-value is used as a practical nominal validation criterion.
Because the candidate is selected using the same SELECTION sample via its two
halves, it should NOT be described as a formal post-selection inferential
p-value or family-wise error-controlled test.

The exact Bernoulli/logistic objective and candidate search are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

from mdss_treescan_stage1 import (
    SubgroupScore,
    exact_subgroup_logit_correction,
)
from residual_repair_benchmark import (
    fit_residual_dictionary_with_offset,
)

from adaptive_structured_scanner_v2_2 import (
    ProposedRule,
    SelectionCandidate,
    AdaptiveStructuredScannerV22,
    propose_legacy_fixed_bin_rule,
)


# ============================================================================
# Helpers
# ============================================================================

def nominal_lr_pvalue(gain: float) -> float:
    """
    Nominal 1-df likelihood-ratio p-value from exact log-likelihood gain.
    """
    if not np.isfinite(gain) or gain < 0:
        return np.nan

    return float(
        chi2.sf(
            2.0 * float(gain),
            df=1,
        )
    )


def stratified_validation_halves(
    y: np.ndarray,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split SELECTION indices into two approximately equal stratified halves.
    """
    y = np.asarray(y, dtype=int)
    idx = np.arange(len(y))

    try:
        a, b = train_test_split(
            idx,
            test_size=0.5,
            random_state=int(seed),
            stratify=y,
        )
    except ValueError:
        # Fallback for unusually small / degenerate samples.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(idx)
        mid = len(perm) // 2
        a = perm[:mid]
        b = perm[mid:]

    return (
        np.asarray(a, dtype=int),
        np.asarray(b, dtype=int),
    )


def _same_nonzero_sign(
    a: float,
    b: float,
) -> bool:
    if (
        not np.isfinite(a)
        or not np.isfinite(b)
        or abs(a) <= 1e-12
        or abs(b) <= 1e-12
    ):
        return False

    return bool(
        np.sign(a) == np.sign(b)
    )


@dataclass
class StableCandidate:
    proposal: ProposedRule

    score_a: SubgroupScore
    score_b: SubgroupScore

    mask_a: np.ndarray
    mask_b: np.ndarray

    n_a: int
    n_b: int

    p_a: float
    p_b: float

    gain_per_obs_a: float
    gain_per_obs_b: float

    stability_score: float
    delta_consistency_ratio: float
    support_consistency_ratio: float


@dataclass
class StabilitySelectionRound:
    round_index: int

    proposed_candidates: int
    valid_half_candidates: int
    stable_candidates: int

    selected_source: Optional[str]

    selected_discovery_gain: float

    selected_half_gain_a: float
    selected_half_gain_b: float

    selected_half_p_a: float
    selected_half_p_b: float

    selected_stability_score: float
    selected_delta_a: float
    selected_delta_b: float
    selected_delta_consistency_ratio: float
    selected_support_consistency_ratio: float

    full_selection_gain: float
    full_selection_delta: float
    full_selection_p: float

    selected_rule_depth: int
    active_terms: int

    adaptive_candidates_proposed: int
    legacy_candidate_proposed: int

    evaluated_exact_candidates: int
    screened_numeric_candidates: int


# ============================================================================
# v2.3 learner
# ============================================================================

class StabilitySelectedScanRepairLearner:
    """
    Candidate pool -> two-half stability ranking -> nominal full-selection p.

    Search:
        identical in spirit to v2.2.

    Selection:
        different from v2.2.
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
        alpha: float = 0.05,
        validation_split_seed: int = 9917,
        selection_min_size: int = 25,
        selection_min_events: int = 2,
        selection_min_nonevents: int = 2,
        half_min_size: Optional[int] = None,
        half_min_events: int = 1,
        half_min_nonevents: int = 1,
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

        self.alpha = float(
            alpha
        )

        self.validation_split_seed = int(
            validation_split_seed
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

        self.half_min_size = (
            max(
                10,
                int(
                    round(
                        self.selection_min_size
                        / 2
                    )
                ),
            )
            if half_min_size is None
            else int(half_min_size)
        )

        self.half_min_events = int(
            half_min_events
        )
        self.half_min_nonevents = int(
            half_min_nonevents
        )

        self.bases_: List[Any] = []
        self.coefficients_ = np.empty(
            0,
            dtype=float,
        )

        self.history_: List[
            StabilitySelectionRound
        ] = []

        self.last_stable_candidates_: List[
            StableCandidate
        ] = []

    # ------------------------------------------------------------------
    # Prediction basis
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Candidate pool
    # ------------------------------------------------------------------

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

        # DISCOVERY-mask deduplication.
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

    # ------------------------------------------------------------------
    # Stability scoring
    # ------------------------------------------------------------------

    def _score_candidate_halves(
        self,
        proposal: ProposedRule,
        X_sel: np.ndarray,
        y_sel: np.ndarray,
        z_sel: np.ndarray,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
    ) -> Optional[StableCandidate]:
        full_mask = proposal.basis.transform(
            X_sel
        ).astype(bool)

        mask_a = full_mask[
            idx_a
        ]
        mask_b = full_mask[
            idx_b
        ]

        score_a = (
            exact_subgroup_logit_correction(
                y_sel[idx_a],
                z_sel[idx_a],
                mask_a,
                l2=0.0,
                min_size=self.half_min_size,
                min_events=self.half_min_events,
                min_nonevents=self.half_min_nonevents,
            )
        )

        score_b = (
            exact_subgroup_logit_correction(
                y_sel[idx_b],
                z_sel[idx_b],
                mask_b,
                l2=0.0,
                min_size=self.half_min_size,
                min_events=self.half_min_events,
                min_nonevents=self.half_min_nonevents,
            )
        )

        if (
            score_a is None
            or score_b is None
        ):
            return None

        if (
            score_a.gain <= 0
            or score_b.gain <= 0
        ):
            return None

        if not _same_nonzero_sign(
            score_a.delta,
            score_b.delta,
        ):
            return None

        n_a = len(
            idx_a
        )
        n_b = len(
            idx_b
        )

        gain_per_obs_a = float(
            score_a.gain / n_a
        )
        gain_per_obs_b = float(
            score_b.gain / n_b
        )

        stability_score = float(
            min(
                gain_per_obs_a,
                gain_per_obs_b,
            )
        )

        abs_da = abs(
            float(
                score_a.delta
            )
        )
        abs_db = abs(
            float(
                score_b.delta
            )
        )

        delta_consistency_ratio = float(
            min(
                abs_da,
                abs_db,
            )
            / max(
                abs_da,
                abs_db,
                1e-12,
            )
        )

        support_a = float(
            mask_a.mean()
        )
        support_b = float(
            mask_b.mean()
        )

        support_consistency_ratio = float(
            min(
                support_a,
                support_b,
            )
            / max(
                support_a,
                support_b,
                1e-12,
            )
        )

        return StableCandidate(
            proposal=proposal,
            score_a=score_a,
            score_b=score_b,
            mask_a=mask_a,
            mask_b=mask_b,
            n_a=n_a,
            n_b=n_b,
            p_a=nominal_lr_pvalue(
                score_a.gain
            ),
            p_b=nominal_lr_pvalue(
                score_b.gain
            ),
            gain_per_obs_a=(
                gain_per_obs_a
            ),
            gain_per_obs_b=(
                gain_per_obs_b
            ),
            stability_score=(
                stability_score
            ),
            delta_consistency_ratio=(
                delta_consistency_ratio
            ),
            support_consistency_ratio=(
                support_consistency_ratio
            ),
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_disc: np.ndarray,
        y_disc: np.ndarray,
        base_logits_disc: np.ndarray,
        X_sel: np.ndarray,
        y_sel: np.ndarray,
        base_logits_sel: np.ndarray,
    ) -> "StabilitySelectedScanRepairLearner":
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
        self.last_stable_candidates_ = []

        idx_a, idx_b = (
            stratified_validation_halves(
                y_sel,
                seed=self.validation_split_seed,
            )
        )

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

            (
                proposals,
                n_adaptive,
                n_legacy,
            ) = self._propose(
                X_disc,
                y_disc,
                z_disc,
            )

            stable: List[
                StableCandidate
            ] = []

            valid_half_candidates = 0

            for prop in proposals:
                candidate = (
                    self._score_candidate_halves(
                        prop,
                        X_sel,
                        y_sel,
                        z_sel,
                        idx_a,
                        idx_b,
                    )
                )

                if candidate is not None:
                    stable.append(
                        candidate
                    )
                    valid_half_candidates += 1

            self.last_stable_candidates_ = (
                stable
            )

            if not stable:
                self.history_.append(
                    StabilitySelectionRound(
                        round_index=t + 1,
                        proposed_candidates=len(
                            proposals
                        ),
                        valid_half_candidates=0,
                        stable_candidates=0,
                        selected_source=None,
                        selected_discovery_gain=np.nan,
                        selected_half_gain_a=np.nan,
                        selected_half_gain_b=np.nan,
                        selected_half_p_a=np.nan,
                        selected_half_p_b=np.nan,
                        selected_stability_score=np.nan,
                        selected_delta_a=np.nan,
                        selected_delta_b=np.nan,
                        selected_delta_consistency_ratio=np.nan,
                        selected_support_consistency_ratio=np.nan,
                        full_selection_gain=np.nan,
                        full_selection_delta=np.nan,
                        full_selection_p=np.nan,
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

            best = max(
                stable,
                key=lambda c: (
                    c.stability_score,
                    c.delta_consistency_ratio,
                    c.support_consistency_ratio,
                    c.proposal.discovery_gain,
                ),
            )

            full_mask = (
                best.proposal.basis.transform(
                    X_sel
                ).astype(bool)
            )

            full_score = (
                exact_subgroup_logit_correction(
                    y_sel,
                    z_sel,
                    full_mask,
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

            if full_score is None:
                full_gain = np.nan
                full_delta = np.nan
                full_p = np.nan
            else:
                full_gain = float(
                    full_score.gain
                )
                full_delta = float(
                    full_score.delta
                )
                full_p = nominal_lr_pvalue(
                    full_score.gain
                )

            accepted = bool(
                full_score is not None
                and np.isfinite(
                    full_p
                )
                and full_p < self.alpha
            )

            if not accepted:
                self.history_.append(
                    StabilitySelectionRound(
                        round_index=t + 1,
                        proposed_candidates=len(
                            proposals
                        ),
                        valid_half_candidates=(
                            valid_half_candidates
                        ),
                        stable_candidates=len(
                            stable
                        ),
                        selected_source=None,
                        selected_discovery_gain=float(
                            best.proposal.discovery_gain
                        ),
                        selected_half_gain_a=float(
                            best.score_a.gain
                        ),
                        selected_half_gain_b=float(
                            best.score_b.gain
                        ),
                        selected_half_p_a=float(
                            best.p_a
                        ),
                        selected_half_p_b=float(
                            best.p_b
                        ),
                        selected_stability_score=float(
                            best.stability_score
                        ),
                        selected_delta_a=float(
                            best.score_a.delta
                        ),
                        selected_delta_b=float(
                            best.score_b.delta
                        ),
                        selected_delta_consistency_ratio=float(
                            best.delta_consistency_ratio
                        ),
                        selected_support_consistency_ratio=float(
                            best.support_consistency_ratio
                        ),
                        full_selection_gain=(
                            full_gain
                        ),
                        full_selection_delta=(
                            full_delta
                        ),
                        full_selection_p=(
                            full_p
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
                            full_score.delta
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
                StabilitySelectionRound(
                    round_index=t + 1,
                    proposed_candidates=len(
                        proposals
                    ),
                    valid_half_candidates=(
                        valid_half_candidates
                    ),
                    stable_candidates=len(
                        stable
                    ),
                    selected_source=(
                        best.proposal.source
                    ),
                    selected_discovery_gain=float(
                        best.proposal.discovery_gain
                    ),
                    selected_half_gain_a=float(
                        best.score_a.gain
                    ),
                    selected_half_gain_b=float(
                        best.score_b.gain
                    ),
                    selected_half_p_a=float(
                        best.p_a
                    ),
                    selected_half_p_b=float(
                        best.p_b
                    ),
                    selected_stability_score=float(
                        best.stability_score
                    ),
                    selected_delta_a=float(
                        best.score_a.delta
                    ),
                    selected_delta_b=float(
                        best.score_b.delta
                    ),
                    selected_delta_consistency_ratio=float(
                        best.delta_consistency_ratio
                    ),
                    selected_support_consistency_ratio=float(
                        best.support_consistency_ratio
                    ),
                    full_selection_gain=float(
                        full_score.gain
                    ),
                    full_selection_delta=float(
                        full_score.delta
                    ),
                    full_selection_p=float(
                        full_p
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
