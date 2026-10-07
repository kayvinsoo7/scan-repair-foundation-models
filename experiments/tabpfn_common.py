"""Frozen TabPFN wrapper with two backends, chosen by environment variables BEFORE import.

TABPFN_BACKEND = "modern"  (default) the `tabpfn` package (pip install tabpfn==9.1.0); the paper uses TabPFN-2.5.
TABPFN_VERSION = "v2.5" | "v2.6" | "v3" | "v3.5" | "v3.5-fast"   (modern only; default v2.5)
TABPFN_DEVICE  = "cuda" | "mps" | "cpu" | "auto"                  (modern only; auto = NVIDIA GPU, else Apple GPU, else CPU)
TABPFN_CHUNK   = rows per prediction call (default 10000; lower it if memory runs out)
TABPFN_FIT_MODE= optional TabPFN fit_mode, e.g. "fit_with_cache" (faster on large GPUs, uses more memory)
TABPFN_CONTEXT = "1000" (default) | "full" | any integer
    How many labelled rows the model sees as its context (its "training data").
    "full" = every row of the training split (modern models accept 50k rows (v2.5) to 1M (v3)).
The model's weights never change: fitting only stores the context, so the model stays frozen.
"""
import os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
BACKEND = os.environ.get("TABPFN_BACKEND", "modern").lower()
VERSION = os.environ.get("TABPFN_VERSION", "v2.5")
DEVICE = os.environ.get("TABPFN_DEVICE", "auto")
CONTEXT = os.environ.get("TABPFN_CONTEXT", "1000").lower()
CHUNK = int(os.environ.get("TABPFN_CHUNK", "0")) or None          # rows per predict call (lower it if you run out of memory)
FIT_MODE = os.environ.get("TABPFN_FIT_MODE", "")                  # e.g. "fit_with_cache" on big GPUs (faster, more memory)
MODEL_PATH = os.environ.get("TABPFN_MODEL_PATH", "")              # testing only: use a local checkpoint file
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")      # allow CPU runs above 1,000 rows (slow, but permitted)

if BACKEND == "v1":
    raise ValueError("TabPFN v1 (ICLR 2023) is not used in this paper and is not included in this repository")
elif BACKEND == "modern":
    from tabpfn import TabPFNClassifier as _Modern
    from tabpfn.constants import ModelVersion as _MV
else:
    raise ValueError(f"TABPFN_BACKEND must be 'v1' or 'modern', got {BACKEND!r}")
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def describe():
    if BACKEND == "v1": return f"TabPFN v1 (ICLR 2023), context={CONTEXT}"
    return f"TabPFN {VERSION} (tabpfn package), device={DEVICE}, context={CONTEXT}"

def context_rows(rng, pool):
    """Rows of `pool` used as the model's context: a random subset of TABPFN_CONTEXT rows, or all of them."""
    pool = np.asarray(pool)
    if CONTEXT == "full": return pool
    k = int(CONTEXT)
    if BACKEND == "v1" and k > 1000: raise ValueError("TabPFN v1 supports at most 1,000 context rows")
    return pool if k >= len(pool) else rng.choice(pool, k, replace=False)

class FrozenTabPFN:
    """A pre-trained transformer that predicts by in-context learning; weights never change."""
    def __init__(self, Xc, yc, n_ens=4, seed=0, categorical_idx=None):
        Xc = np.asarray(Xc, float); yc = np.asarray(yc, int)
        if BACKEND == "v1":
            self.clf = _V1(device="cpu", N_ensemble_configurations=n_ens, seed=seed)
        else:
            kw = dict(device=DEVICE, random_state=seed)
            if categorical_idx: kw["categorical_features_indices"] = list(categorical_idx)
            if FIT_MODE: kw["fit_mode"] = FIT_MODE
            if MODEL_PATH: kw["model_path"] = MODEL_PATH
            self.clf = _Modern.create_default_for_version(_MV(VERSION), **kw)
        self.clf.fit(Xc, yc)
    def logits(self, X, chunk=None):
        X = np.asarray(X, float); chunk = chunk or CHUNK or (2000 if BACKEND == "v1" else 10000)
        p = np.concatenate([self.clf.predict_proba(X[i:i+chunk])[:, 1] for i in range(0, len(X), chunk)])
        p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def platt(z_fit, y_fit):
    m = LogisticRegression(C=1e6).fit(z_fit[:, None], y_fit); return lambda z: m.decision_function(z[:, None])
def isotonic(z_fit, y_fit):
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6).fit(z_fit, y_fit)
    return lambda z: (lambda p: np.log(p/(1-p)))(np.clip(m.predict(z), 1e-6, 1-1e-6))
