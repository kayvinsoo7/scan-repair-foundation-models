
"""
Prototype Fully-Corrective Competitive Tree–Scan learner.

Requires:
    mdss_treescan_stage1.py

Core algorithm:
1. Maintain an additive predictor F(x) = beta0 + sum_j theta_j phi_j(x).
2. At each round:
   a. Ask XGBoost for ONE shallow tree, using the current logits as base_margin.
   b. Ask MDSS for ONE structured subgroup rule on the same current logits.
   c. Re-score both candidates using exact logistic loss reduction.
   d. Add the better candidate to the active dictionary.
   e. Fully re-optimise all active coefficients under penalised logistic loss.
3. Stop when no candidate gives enough exact gain.

This is a research prototype: transparent, testable, and intentionally not yet
micro-optimised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize

from mdss_treescan_stage1 import (
    BinnedMDSS,
    BinningModel,
    MDSSResult,
    exact_subgroup_logit_correction,
    fit_quantile_binner,
    generate_structured_residual_data,
    logistic_loss_from_logits,
    sigmoid,
    SyntheticConfig,
)


# ---------------------------------------------------------------------
# Exact one-dimensional gain for a general candidate direction h(x)
# ---------------------------------------------------------------------

@dataclass
class DirectionScore:
    gain: float
    alpha: float


def exact_direction_gain(
    y: np.ndarray,
    logits: np.ndarray,
    direction: np.ndarray,
    *,
    l2: float = 0.0,
    alpha_bound: float = 12.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> DirectionScore:
    """
    Compute:
        max_alpha L(F) - L(F + alpha*h) - (l2/2) alpha^2

    by Newton optimisation of the exact 1D logistic objective.
    """
    y = np.asarray(y, dtype=float)
    z = np.asarray(logits, dtype=float)
    h = np.asarray(direction, dtype=float)

    if np.allclose(h, 0.0):
        return DirectionScore(gain=0.0, alpha=0.0)

    alpha = 0.0

    for _ in range(max_iter):
        p = sigmoid(z + alpha * h)

        # derivative of GAIN wrt alpha
        grad_gain = float(np.sum(h * (y - p)) - l2 * alpha)

        # positive curvature magnitude for -second derivative of gain
        curvature = float(np.sum((h * h) * p * (1.0 - p)) + l2)

        if curvature <= 0:
            break

        step = grad_gain / curvature
        new_alpha = float(np.clip(alpha + step, -alpha_bound, alpha_bound))

        if abs(new_alpha - alpha) < tol:
            alpha = new_alpha
            break
        alpha = new_alpha

    before = logistic_loss_from_logits(y, z)
    after = logistic_loss_from_logits(y, z + alpha * h)
    gain = before - after - 0.5 * l2 * alpha * alpha

    return DirectionScore(gain=max(0.0, float(gain)), alpha=float(alpha))


# ---------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------

@dataclass
class TreeBasis:
    booster: xgb.Booster

    def transform(self, X: np.ndarray) -> np.ndarray:
        d = xgb.DMatrix(np.asarray(X, dtype=float))
        # base_score=0.5 -> zero margin; without base_margin the prediction
        # is the tree contribution itself.
        return self.booster.predict(d, output_margin=True).astype(float)


@dataclass
class ScanBasis:
    restrictions: dict
    binner: BinningModel

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xb = self.binner.transform(np.asarray(X, dtype=float))
        mask = np.ones(Xb.shape[0], dtype=bool)
        for j, allowed in self.restrictions.items():
            mask &= np.isin(Xb[:, int(j)], tuple(allowed))
            if not np.any(mask):
                break
        return mask.astype(float)


Basis = Union[TreeBasis, ScanBasis]


# ---------------------------------------------------------------------
# Full coefficient refit
# ---------------------------------------------------------------------

@dataclass
class CoefficientFit:
    intercept: float
    coefficients: np.ndarray
    loss: float
    success: bool


def fit_active_dictionary(
    y: np.ndarray,
    Phi: np.ndarray,
    *,
    l2: float = 1e-4,
    init: Optional[np.ndarray] = None,
) -> CoefficientFit:
    """
    Convex penalised logistic refit over a FIXED active dictionary.

    Phi shape: (n_samples, n_basis)
    Parameter vector: [intercept, coefficients...]
    Intercept is not penalised.
    """
    y = np.asarray(y, dtype=float)
    Phi = np.asarray(Phi, dtype=float)

    n_basis = Phi.shape[1]

    if init is None or len(init) != n_basis + 1:
        ybar = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
        beta0 = np.log(ybar / (1.0 - ybar))
        x0 = np.zeros(n_basis + 1, dtype=float)
        x0[0] = beta0
    else:
        x0 = np.asarray(init, dtype=float).copy()

    def fun_and_grad(beta: np.ndarray):
        intercept = beta[0]
        coef = beta[1:]
        logits = intercept + Phi @ coef
        p = sigmoid(logits)

        loss = (
            logistic_loss_from_logits(y, logits)
            + 0.5 * l2 * float(np.dot(coef, coef))
        )

        residual = p - y
        grad0 = float(np.sum(residual))
        grad_coef = Phi.T @ residual + l2 * coef
        grad = np.concatenate(([grad0], grad_coef))
        return loss, grad

    res = minimize(
        lambda b: fun_and_grad(b),
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )

    beta = res.x
    return CoefficientFit(
        intercept=float(beta[0]),
        coefficients=np.asarray(beta[1:], dtype=float),
        loss=float(res.fun),
        success=bool(res.success),
    )


# ---------------------------------------------------------------------
# One-tree XGBoost proposal oracle
# ---------------------------------------------------------------------

def propose_xgb_tree(
    X: np.ndarray,
    y: np.ndarray,
    current_logits: np.ndarray,
    *,
    max_depth: int = 2,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    seed: int = 123,
) -> TreeBasis:
    """
    Ask XGBoost for exactly one tree using current_logits as base_margin.

    XGBoost therefore computes its logistic gradients/Hessians around the
    CURRENT Tree–Scan predictor, not around a fresh intercept-only model.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    current_logits = np.asarray(current_logits, dtype=float)

    dtrain = xgb.DMatrix(X, label=y, base_margin=current_logits)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": int(max_depth),
        "eta": 1.0,                   # candidate shape; outer model rescales it
        "min_child_weight": float(min_child_weight),
        "lambda": float(reg_lambda),
        "gamma": float(gamma),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "base_score": 0.5,            # zero default margin for later transform()
        "seed": int(seed),
        "verbosity": 0,
    }

    booster = xgb.train(params, dtrain, num_boost_round=1)
    return TreeBasis(booster=booster)


# ---------------------------------------------------------------------
# Competitive Tree–Scan model
# ---------------------------------------------------------------------

@dataclass
class RoundRecord:
    round_index: int
    tree_gain: float
    tree_alpha: float
    scan_gain: float
    scan_delta: float
    chosen: str
    active_basis_count: int
    train_loss_after_refit: float


class CompetitiveTreeScan:
    def __init__(
        self,
        *,
        max_rounds: int = 12,
        tree_max_depth: int = 2,
        tree_min_child_weight: float = 1.0,
        tree_reg_lambda: float = 1.0,
        coefficient_l2: float = 1e-4,
        gain_tolerance: float = 1e-3,
        mdss_n_bins: int = 5,
        mdss_n_restarts: int = 6,
        mdss_max_passes: int = 6,
        mdss_min_size: int = 50,
        mdss_min_events: int = 3,
        mdss_min_nonevents: int = 3,
        mdss_max_rule_features: Optional[int] = 5,
        validation_patience: int = 3,
        seed: int = 123,
        verbose: bool = True,
    ) -> None:
        self.max_rounds = max_rounds
        self.tree_max_depth = tree_max_depth
        self.tree_min_child_weight = tree_min_child_weight
        self.tree_reg_lambda = tree_reg_lambda
        self.coefficient_l2 = coefficient_l2
        self.gain_tolerance = gain_tolerance
        self.validation_patience = validation_patience
        self.seed = seed
        self.verbose = verbose

        self.scanner = BinnedMDSS(
            n_bins=mdss_n_bins,
            n_restarts=mdss_n_restarts,
            max_passes=mdss_max_passes,
            min_size=mdss_min_size,
            min_events=mdss_min_events,
            min_nonevents=mdss_min_nonevents,
            max_rule_features=mdss_max_rule_features,
            l2=0.0,
            seed=seed,
        )

        self.bases_: List[Basis] = []
        self.intercept_: float = 0.0
        self.coefficients_: np.ndarray = np.empty(0)
        self.history_: List[RoundRecord] = []

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        if len(self.bases_) == 0:
            return np.empty((len(X), 0), dtype=float)
        return np.column_stack([b.transform(X) for b in self.bases_])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if len(self.bases_) == 0:
            return np.full(X.shape[0], self.intercept_, dtype=float)
        Phi = self._design_matrix(X)
        return self.intercept_ + Phi @ self.coefficients_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CompetitiveTreeScan":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X_val is not None:
            X_val = np.asarray(X_val, dtype=float)
        if y_val is not None:
            y_val = np.asarray(y_val, dtype=int)

        # Fit binning once and freeze it for all scan rounds.
        self.scanner.fit_binner(X)

        ybar = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
        self.intercept_ = float(np.log(ybar / (1.0 - ybar)))
        self.coefficients_ = np.empty(0)
        self.bases_ = []
        self.history_ = []

        best_val_loss = np.inf
        best_snapshot = None
        rounds_without_val_improvement = 0

        for t in range(self.max_rounds):
            current_logits = self.decision_function(X)

            # ---- Tree proposal ----
            tree_basis = propose_xgb_tree(
                X,
                y,
                current_logits,
                max_depth=self.tree_max_depth,
                min_child_weight=self.tree_min_child_weight,
                reg_lambda=self.tree_reg_lambda,
                seed=self.seed + t,
            )
            tree_direction = tree_basis.transform(X)
            tree_score = exact_direction_gain(
                y, current_logits, tree_direction
            )

            # ---- MDSS proposal ----
            scan_result = self.scanner.scan(X, y, current_logits)
            scan_basis = ScanBasis(
                restrictions=dict(scan_result.restrictions),
                binner=self.scanner.binner_,
            )
            # scan_result.gain/delta are already exact for this indicator.
            scan_gain = float(scan_result.gain)
            scan_delta = float(scan_result.delta)

            # ---- Competition ----
            if tree_score.gain > scan_gain:
                chosen = "tree"
                chosen_basis: Basis = tree_basis
                best_gain = tree_score.gain
            else:
                chosen = "scan"
                chosen_basis = scan_basis
                best_gain = scan_gain

            if self.verbose:
                print(
                    f"round {t+1:02d} | "
                    f"tree gain={tree_score.gain:9.4f} | "
                    f"scan gain={scan_gain:9.4f} | "
                    f"choose={chosen}"
                )

            if best_gain < self.gain_tolerance:
                if self.verbose:
                    print(
                        f"Stopping: best exact candidate gain {best_gain:.6g} "
                        f"< tolerance {self.gain_tolerance:.6g}"
                    )
                break

            self.bases_.append(chosen_basis)

            # ---- Fully corrective coefficient refit ----
            Phi = self._design_matrix(X)

            old_beta = np.concatenate(
                ([self.intercept_], self.coefficients_)
            )
            # New column -> initialise its coefficient to the exact 1D candidate
            # coefficient for a better starting point.
            if chosen == "tree":
                new_coef = tree_score.alpha
            else:
                new_coef = scan_delta

            init = np.concatenate((old_beta, [new_coef]))

            fit = fit_active_dictionary(
                y,
                Phi,
                l2=self.coefficient_l2,
                init=init,
            )

            self.intercept_ = fit.intercept
            self.coefficients_ = fit.coefficients

            self.history_.append(
                RoundRecord(
                    round_index=t + 1,
                    tree_gain=float(tree_score.gain),
                    tree_alpha=float(tree_score.alpha),
                    scan_gain=float(scan_gain),
                    scan_delta=float(scan_delta),
                    chosen=chosen,
                    active_basis_count=len(self.bases_),
                    train_loss_after_refit=float(fit.loss),
                )
            )

            # Validation controls model selection; the audit/test set remains
            # untouched. We snapshot the best active dictionary and coefficients.
            if X_val is not None and y_val is not None:
                val_logits = self.decision_function(X_val)
                val_loss = logistic_loss_from_logits(y_val, val_logits) / len(y_val)

                if self.verbose:
                    print(f"          validation logloss={val_loss:.6f}")

                if val_loss < best_val_loss - 1e-8:
                    best_val_loss = val_loss
                    best_snapshot = (
                        list(self.bases_),
                        float(self.intercept_),
                        self.coefficients_.copy(),
                        list(self.history_),
                    )
                    rounds_without_val_improvement = 0
                else:
                    rounds_without_val_improvement += 1

                if rounds_without_val_improvement >= self.validation_patience:
                    if self.verbose:
                        print(
                            f"Stopping after {self.validation_patience} rounds "
                            "without validation improvement."
                        )
                    break

        if best_snapshot is not None:
            (
                self.bases_,
                self.intercept_,
                self.coefficients_,
                self.history_,
            ) = best_snapshot

        return self


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)

    return {
        "logloss": float(log_loss(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def split_indices(n: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(0.60 * n))
    n_val = int(round(0.20 * n))
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


def demo_integrated_model():
    """
    Moderate-size mechanism demo. It is not the final benchmark experiment.
    """
    cfg = SyntheticConfig(
        n=9_000,
        p=30,
        interaction_order=5,
        subgroup_prevalence=0.08,
        subgroup_log_odds=float(np.log(4.0)),
        event_prevalence=0.10,
        seed=20260926,
    )
    data = generate_structured_residual_data(cfg)
    tr, va, te = split_indices(cfg.n, seed=20260926)

    model = CompetitiveTreeScan(
        max_rounds=6,
        tree_max_depth=2,
        coefficient_l2=1e-4,
        gain_tolerance=1e-2,
        mdss_n_bins=5,
        mdss_n_restarts=4,
        mdss_max_passes=5,
        mdss_min_size=35,
        mdss_min_events=3,
        mdss_min_nonevents=3,
        mdss_max_rule_features=5,
        validation_patience=2,
        seed=20260926,
        verbose=True,
    )

    model.fit(data.X[tr], data.y[tr], data.X[va], data.y[va])

    ptr = model.predict_proba(data.X[tr])
    pva = model.predict_proba(data.X[va])
    pte = model.predict_proba(data.X[te])

    return {
        "config": cfg,
        "data": data,
        "splits": (tr, va, te),
        "model": model,
        "train_metrics": binary_metrics(data.y[tr], ptr),
        "val_metrics": binary_metrics(data.y[va], pva),
        "test_metrics": binary_metrics(data.y[te], pte),
    }


if __name__ == "__main__":
    out = demo_integrated_model()
    print("\nTrain:", out["train_metrics"])
    print("Val  :", out["val_metrics"])
    print("Test :", out["test_metrics"])
