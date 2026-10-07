
"""
Stage 1 implementation for the Tree–Scan project.

What is implemented here:
1. Controlled synthetic binary-outcome generator with:
   - interaction order k
   - subgroup prevalence s
   - subgroup log-odds correction delta
   - overall event prevalence pi
2. Exact Bernoulli/logistic subgroup correction:
   - optimal constant logit shift for a supplied subgroup
   - exact reduction in logistic loss
3. A practical multidimensional subset scan (MDSS) oracle:
   - discretises continuous variables
   - searches Cartesian subgroups by coordinate ascent
   - each coordinate update is exact over the discretised bins
   - multiple feature-order restarts
4. A canonical demonstration.

This is intentionally the "mechanism" layer only.
XGBoost / competitive Tree–Scan boosting comes next.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Basic numerical utilities
# ---------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically safe logit."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def logistic_loss_from_logits(y: np.ndarray, logits: np.ndarray) -> float:
    """Summed binary negative log-likelihood."""
    y = np.asarray(y, dtype=float)
    logits = np.asarray(logits, dtype=float)
    return float(np.sum(np.logaddexp(0.0, logits) - y * logits))


# ---------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticConfig:
    n: int = 10_000
    p: int = 30
    interaction_order: int = 5          # k
    subgroup_prevalence: float = 0.08   # s
    subgroup_log_odds: float = float(np.log(4.0))  # delta
    event_prevalence: float = 0.10      # pi
    n_background: int = 6
    background_coefficients: Tuple[float, ...] = (
        1.0, -0.8, 0.7, -0.6, 0.5, -0.4
    )
    seed: int = 123


@dataclass
class SyntheticData:
    X: np.ndarray
    y: np.ndarray
    true_logits: np.ndarray
    true_probabilities: np.ndarray
    base_logits: np.ndarray
    base_probabilities: np.ndarray
    subgroup_mask: np.ndarray
    subgroup_threshold: float
    intercept: float
    background_feature_indices: np.ndarray


def _solve_intercept(
    raw_signal: np.ndarray,
    target_prevalence: float,
    lower: float = -20.0,
    upper: float = 20.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """
    Solve mean(sigmoid(b0 + raw_signal)) = target_prevalence
    by bisection. No scipy dependency is needed.
    """
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError("target_prevalence must lie strictly between 0 and 1.")

    lo, hi = lower, upper
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        mean_p = float(np.mean(sigmoid(mid + raw_signal)))
        if abs(mean_p - target_prevalence) < tol:
            return mid
        if mean_p < target_prevalence:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_structured_residual_data(cfg: SyntheticConfig) -> SyntheticData:
    """
    Generate:
        logit P(Y=1|X) = b0 + f_base(X) + delta * I(X in S*)

    S* is a k-way conjunction:
        X_1 > c, ..., X_k > c

    where c = 1 - s^(1/k), making P(S*) = s in expectation
    when X_j ~ Uniform(0,1).

    The returned `base_logits` omit the planted subgroup term. They therefore
    represent a deliberately misspecified-but-otherwise-correct base predictor.
    """
    if cfg.interaction_order < 1:
        raise ValueError("interaction_order must be >= 1.")
    if cfg.interaction_order >= cfg.p:
        raise ValueError("interaction_order must be smaller than p.")
    if not 0.0 < cfg.subgroup_prevalence < 1.0:
        raise ValueError("subgroup_prevalence must lie in (0,1).")
    if cfg.n_background > len(cfg.background_coefficients):
        raise ValueError("Need at least n_background coefficients.")
    if cfg.interaction_order + cfg.n_background > cfg.p:
        raise ValueError("Not enough features for non-overlapping planted/background features.")

    rng = np.random.default_rng(cfg.seed)
    X = rng.uniform(0.0, 1.0, size=(cfg.n, cfg.p))

    k = cfg.interaction_order
    s = cfg.subgroup_prevalence
    threshold = 1.0 - s ** (1.0 / k)

    subgroup_mask = np.all(X[:, :k] > threshold, axis=1)

    bg_idx = np.arange(k, k + cfg.n_background)
    beta = np.asarray(cfg.background_coefficients[: cfg.n_background], dtype=float)

    # Center Uniform(0,1) covariates to approximately [-1,1].
    background_signal = (2.0 * X[:, bg_idx] - 1.0) @ beta

    raw_true_signal = (
        background_signal
        + cfg.subgroup_log_odds * subgroup_mask.astype(float)
    )

    intercept = _solve_intercept(raw_true_signal, cfg.event_prevalence)

    true_logits = intercept + raw_true_signal
    true_prob = sigmoid(true_logits)

    # "Base" predictor knows the background signal but not the planted subgroup.
    base_logits = intercept + background_signal
    base_prob = sigmoid(base_logits)

    y = rng.binomial(1, true_prob).astype(int)

    return SyntheticData(
        X=X,
        y=y,
        true_logits=true_logits,
        true_probabilities=true_prob,
        base_logits=base_logits,
        base_probabilities=base_prob,
        subgroup_mask=subgroup_mask,
        subgroup_threshold=threshold,
        intercept=intercept,
        background_feature_indices=bg_idx,
    )


# ---------------------------------------------------------------------
# Exact subgroup correction / exact Bernoulli likelihood gain
# ---------------------------------------------------------------------

@dataclass
class SubgroupScore:
    gain: float
    delta: float
    size: int
    events: int
    nonevents: int
    mean_observed: float
    mean_predicted_before: float
    mean_predicted_after: float


def exact_subgroup_logit_correction(
    y: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray,
    *,
    l2: float = 0.0,
    delta_bound: float = 12.0,
    min_size: int = 25,
    min_events: int = 2,
    min_nonevents: int = 2,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> Optional[SubgroupScore]:
    """
    Find the exact optimal common logit correction delta for a subgroup.

    Maximises:
        Delta(S,delta;F)
          = L(F) - L(F + delta * I_S) - (l2/2) delta^2

    The first-order condition is:
        sum_{i in S} [y_i - sigmoid(F_i + delta)] - l2*delta = 0.

    Newton's method is safe here because the objective is concave in delta.
    We clip delta to a finite bound for numerical robustness.
    """
    y = np.asarray(y, dtype=float)
    logits = np.asarray(logits, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    idx = np.flatnonzero(mask)
    n = idx.size
    if n < min_size:
        return None

    ys = y[idx]
    zs = logits[idx]
    events = int(ys.sum())
    nonevents = int(n - events)

    if events < min_events or nonevents < min_nonevents:
        return None

    delta = 0.0

    for _ in range(max_iter):
        p = sigmoid(zs + delta)
        grad = float(np.sum(ys - p) - l2 * delta)
        curvature = float(np.sum(p * (1.0 - p)) + l2)

        if curvature <= 0:
            break

        step = grad / curvature
        new_delta = float(np.clip(delta + step, -delta_bound, delta_bound))

        if abs(new_delta - delta) < tol:
            delta = new_delta
            break
        delta = new_delta

    before = np.logaddexp(0.0, zs) - ys * zs
    z_after = zs + delta
    after = np.logaddexp(0.0, z_after) - ys * z_after

    gain = float(np.sum(before - after) - 0.5 * l2 * delta * delta)

    p_before = sigmoid(zs)
    p_after = sigmoid(z_after)

    return SubgroupScore(
        gain=max(0.0, gain),
        delta=float(delta),
        size=int(n),
        events=events,
        nonevents=nonevents,
        mean_observed=float(np.mean(ys)),
        mean_predicted_before=float(np.mean(p_before)),
        mean_predicted_after=float(np.mean(p_after)),
    )


# ---------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------

@dataclass
class BinningModel:
    edges: List[np.ndarray]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape, dtype=np.int16)
        for j, e in enumerate(self.edges):
            # Internal edges only; digitize returns bins 0..K-1.
            out[:, j] = np.digitize(X[:, j], e[1:-1], right=False)
        return out


def fit_quantile_binner(X: np.ndarray, n_bins: int = 5) -> BinningModel:
    X = np.asarray(X, dtype=float)
    edges: List[np.ndarray] = []
    qs = np.linspace(0.0, 1.0, n_bins + 1)

    for j in range(X.shape[1]):
        e = np.quantile(X[:, j], qs)
        e = np.unique(e)
        if len(e) < 2:
            # Degenerate column.
            e = np.array([X[:, j].min() - 1e-12, X[:, j].max() + 1e-12])
        else:
            e = e.astype(float)
            e[0] -= 1e-12
            e[-1] += 1e-12
        edges.append(e)

    return BinningModel(edges=edges)


# ---------------------------------------------------------------------
# Practical MDSS oracle
# ---------------------------------------------------------------------

@dataclass
class MDSSResult:
    gain: float
    delta: float
    mask: np.ndarray
    restrictions: Dict[int, Tuple[int, ...]]
    score: SubgroupScore
    restart: int
    passes: int


def _all_nonempty_subsets(levels: Sequence[int]) -> Iterable[Tuple[int, ...]]:
    """
    Enumerate all non-empty subsets. This is deliberately exact within a
    coordinate. Keep n_bins modest (e.g. 4-6).
    """
    levels = tuple(levels)
    for r in range(1, len(levels) + 1):
        for subset in combinations(levels, r):
            yield subset


class BinnedMDSS:
    """
    Coordinate-ascent MDSS over discretised features.

    For a given feature, holding all other feature restrictions fixed, we
    exhaustively evaluate every non-empty subset of that feature's bins.
    This makes each coordinate update exact for modest n_bins.

    Across dimensions, the search is non-convex. We therefore use multiple
    feature-order restarts.

    This is a practical research implementation, not yet an LTSS-optimised
    production scanner.
    """

    def __init__(
        self,
        *,
        n_bins: int = 5,
        n_restarts: int = 8,
        max_passes: int = 8,
        min_size: int = 50,
        min_events: int = 3,
        min_nonevents: int = 3,
        max_rule_features: Optional[int] = None,
        l2: float = 0.0,
        seed: int = 123,
    ) -> None:
        self.n_bins = n_bins
        self.n_restarts = n_restarts
        self.max_passes = max_passes
        self.min_size = min_size
        self.min_events = min_events
        self.min_nonevents = min_nonevents
        self.max_rule_features = max_rule_features
        self.l2 = l2
        self.seed = seed
        self.binner_: Optional[BinningModel] = None

    def fit_binner(self, X: np.ndarray) -> "BinnedMDSS":
        self.binner_ = fit_quantile_binner(X, self.n_bins)
        return self

    def _mask_from_restrictions(
        self,
        Xb: np.ndarray,
        restrictions: Dict[int, Tuple[int, ...]],
        skip_feature: Optional[int] = None,
    ) -> np.ndarray:
        mask = np.ones(Xb.shape[0], dtype=bool)
        for j, allowed in restrictions.items():
            if j == skip_feature:
                continue
            mask &= np.isin(Xb[:, j], allowed)
            if not np.any(mask):
                break
        return mask

    def scan(
        self,
        X: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
    ) -> MDSSResult:
        if self.binner_ is None:
            self.fit_binner(X)

        assert self.binner_ is not None
        Xb = self.binner_.transform(X)
        n, p = Xb.shape
        rng = np.random.default_rng(self.seed)

        levels_by_feature = [
            tuple(int(v) for v in np.unique(Xb[:, j]))
            for j in range(p)
        ]

        best_global: Optional[MDSSResult] = None

        for restart in range(self.n_restarts):
            restrictions: Dict[int, Tuple[int, ...]] = {}

            # Restart 0: deterministic, all dimensions initially unrestricted.
            # Later restarts: seed one randomly chosen feature with a random
            # non-full subset to escape some coordinate-ascent local optima.
            if restart > 0:
                j0 = int(rng.integers(0, p))
                lv = levels_by_feature[j0]
                if len(lv) > 1:
                    subset_size = int(rng.integers(1, len(lv)))
                    subset = tuple(sorted(rng.choice(lv, size=subset_size, replace=False)))
                    restrictions[j0] = subset

            previous_gain = -np.inf
            passes_done = 0

            # Different feature order at each restart.
            feature_order = np.arange(p)
            if restart > 0:
                rng.shuffle(feature_order)

            for pass_idx in range(self.max_passes):
                passes_done = pass_idx + 1
                changed = False

                for j in feature_order:
                    j = int(j)
                    base_mask = self._mask_from_restrictions(
                        Xb, restrictions, skip_feature=j
                    )

                    best_for_feature_score: Optional[SubgroupScore] = None
                    best_subset: Optional[Tuple[int, ...]] = None

                    # Enforce the subgroup-class complexity cap G_{K,m}.
                    # If K other dimensions are already restricted, a currently
                    # unrestricted dimension cannot introduce an additional
                    # restriction in this coordinate update.
                    full_levels_j = levels_by_feature[j]
                    restricted_elsewhere = sum(
                        1 for f in restrictions.keys() if f != j
                    )
                    j_currently_restricted = j in restrictions

                    if (
                        self.max_rule_features is not None
                        and not j_currently_restricted
                        and restricted_elsewhere >= self.max_rule_features
                    ):
                        candidate_subsets = [full_levels_j]
                    else:
                        candidate_subsets = _all_nonempty_subsets(full_levels_j)

                    for subset in candidate_subsets:
                        candidate_mask = base_mask & np.isin(Xb[:, j], subset)

                        sc = exact_subgroup_logit_correction(
                            y,
                            logits,
                            candidate_mask,
                            l2=self.l2,
                            min_size=self.min_size,
                            min_events=self.min_events,
                            min_nonevents=self.min_nonevents,
                        )
                        if sc is None:
                            continue

                        if (
                            best_for_feature_score is None
                            or sc.gain > best_for_feature_score.gain + 1e-12
                        ):
                            best_for_feature_score = sc
                            best_subset = subset

                    if best_subset is None:
                        continue

                    full = levels_by_feature[j]
                    old = restrictions.get(j, full)

                    # Store no restriction when all bins are allowed.
                    if set(best_subset) == set(full):
                        restrictions.pop(j, None)
                    else:
                        restrictions[j] = tuple(best_subset)

                    new = restrictions.get(j, full)
                    if tuple(old) != tuple(new):
                        changed = True

                current_mask = self._mask_from_restrictions(Xb, restrictions)
                current_score = exact_subgroup_logit_correction(
                    y,
                    logits,
                    current_mask,
                    l2=self.l2,
                    min_size=self.min_size,
                    min_events=self.min_events,
                    min_nonevents=self.min_nonevents,
                )

                current_gain = 0.0 if current_score is None else current_score.gain

                if current_gain <= previous_gain + 1e-10 and not changed:
                    break
                previous_gain = current_gain

            final_mask = self._mask_from_restrictions(Xb, restrictions)
            final_score = exact_subgroup_logit_correction(
                y,
                logits,
                final_mask,
                l2=self.l2,
                min_size=self.min_size,
                min_events=self.min_events,
                min_nonevents=self.min_nonevents,
            )
            if final_score is None:
                continue

            result = MDSSResult(
                gain=final_score.gain,
                delta=final_score.delta,
                mask=final_mask,
                restrictions=dict(restrictions),
                score=final_score,
                restart=restart,
                passes=passes_done,
            )

            if best_global is None or result.gain > best_global.gain:
                best_global = result

        if best_global is None:
            raise RuntimeError("No valid subgroup found. Relax subgroup constraints.")

        return best_global

    def describe_restrictions(
        self,
        result: MDSSResult,
        feature_names: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if self.binner_ is None:
            raise RuntimeError("Binner has not been fitted.")

        desc: List[str] = []
        for j, bins in sorted(result.restrictions.items()):
            name = feature_names[j] if feature_names is not None else f"X{j+1}"
            e = self.binner_.edges[j]
            intervals = []
            for b in bins:
                lo = e[b]
                hi = e[b + 1]
                intervals.append(f"[{lo:.3f}, {hi:.3f})")
            desc.append(f"{name} in " + " U ".join(intervals))
        return desc


# ---------------------------------------------------------------------
# Demo / diagnostics
# ---------------------------------------------------------------------

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.sum(a | b)
    if union == 0:
        return 1.0
    return float(np.sum(a & b) / union)


def canonical_demo() -> dict:
    cfg = SyntheticConfig(
        n=10_000,
        p=30,
        interaction_order=5,
        subgroup_prevalence=0.08,
        subgroup_log_odds=float(np.log(4.0)),
        event_prevalence=0.10,
        seed=20260926,
    )

    data = generate_structured_residual_data(cfg)

    # Score the known true subgroup under the deliberately misspecified
    # background-only predictor.
    true_score = exact_subgroup_logit_correction(
        data.y,
        data.base_logits,
        data.subgroup_mask,
        min_size=50,
        min_events=3,
        min_nonevents=3,
    )

    scanner = BinnedMDSS(
        n_bins=5,
        n_restarts=6,
        max_passes=6,
        min_size=50,
        min_events=3,
        min_nonevents=3,
        max_rule_features=5,
        seed=20260926,
    )

    discovered = scanner.scan(data.X, data.y, data.base_logits)

    return {
        "config": cfg,
        "data": data,
        "true_score": true_score,
        "scanner": scanner,
        "discovered": discovered,
        "jaccard": jaccard(data.subgroup_mask, discovered.mask),
        "descriptions": scanner.describe_restrictions(discovered),
    }


if __name__ == "__main__":
    out = canonical_demo()

    cfg = out["config"]
    data = out["data"]
    true_score = out["true_score"]
    discovered = out["discovered"]

    print("=== Canonical structured-residual experiment ===")
    print(f"n                         : {cfg.n}")
    print(f"p                         : {cfg.p}")
    print(f"interaction order k       : {cfg.interaction_order}")
    print(f"target subgroup support s : {cfg.subgroup_prevalence:.3f}")
    print(f"realized subgroup support : {data.subgroup_mask.mean():.3f}")
    print(f"true subgroup threshold   : {data.subgroup_threshold:.4f}")
    print(f"target event prevalence   : {cfg.event_prevalence:.3f}")
    print(f"realized event prevalence : {data.y.mean():.3f}")
    print(f"true planted delta        : {cfg.subgroup_log_odds:.4f}")
    print()

    if true_score is not None:
        print("--- True planted subgroup under base predictor ---")
        print(f"exact gain                : {true_score.gain:.4f}")
        print(f"estimated delta           : {true_score.delta:.4f}")
        print(f"group size                : {true_score.size}")
        print(f"observed event rate       : {true_score.mean_observed:.4f}")
        print(f"predicted before          : {true_score.mean_predicted_before:.4f}")
        print(f"predicted after           : {true_score.mean_predicted_after:.4f}")
        print()

    print("--- MDSS discovered subgroup ---")
    print(f"exact gain                : {discovered.gain:.4f}")
    print(f"estimated delta           : {discovered.delta:.4f}")
    print(f"group size                : {discovered.score.size}")
    print(f"observed event rate       : {discovered.score.mean_observed:.4f}")
    print(f"predicted before          : {discovered.score.mean_predicted_before:.4f}")
    print(f"predicted after           : {discovered.score.mean_predicted_after:.4f}")
    print(f"Jaccard with true subgroup: {out['jaccard']:.4f}")
    print(f"restart / passes          : {discovered.restart} / {discovered.passes}")
    print("restrictions:")
    for line in out["descriptions"]:
        print("  -", line)
