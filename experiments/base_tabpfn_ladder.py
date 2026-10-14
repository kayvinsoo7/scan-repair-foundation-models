"""STAGE 1 (GPU recommended): run the frozen TabPFN at every context size of the ladder and SAVE its raw outputs.

Nothing is corrected here. For each dataset, seed and context size we store TabPFN's logits on the
discovery, selection and test rows, plus timings, in one small .npz file. Every correction method
(Platt, isotonic, the scan, ...) then runs later from these files on a CPU (corrections.py), so methods,
metrics and tables can be changed without touching the GPU again.

Contexts are NESTED: per seed we shuffle the training split once and give the model the first k rows,
so the 1,000-row context contains the 500-row one, and "full" contains them all. The only thing that
changes going up the ladder is how much labelled data the model sees.

Usage:  python ladder_logits.py                      # everything (resumes: existing files are skipped)
Environment (all optional):
  LADDER    = "500 1000 1500 2000 3000 4000 6000 8000 10000 full"   context sizes; sizes >= the training split collapse to "full"
  DATASETS  = "synth8 synth1 credit"      synth8 = planted subgroup (odds ratio 8), synth1 = no signal, credit = Default of Credit Card Clients
  SEEDS     = number of seeds per dataset (default 10; QUICK=1 -> 5)
  OUT       = output folder (default ladder_logits/<backend-version>)
plus the TABPFN_* variables of tabpfn_common.py (backend, version, device, chunk).
"""
import os, sys, time, json, platform, numpy as np
os.environ.setdefault("TABPFN_BACKEND", "modern")
os.environ["TABPFN_CONTEXT"] = "full"                    # context selection is done here, not in tabpfn_common
from common import *
import tabpfn_common as TC
from ladder_data import load_dataset, DATASET_SEEDS
from ladder_env import env_info

LADDER = os.environ.get("LADDER", "500 1000 1500 2000 3000 4000 6000 8000 10000 full").split()
DATASETS = os.environ.get("DATASETS", "synth8 synth1 credit").split()
NSEEDS = int(os.environ.get("SEEDS", "5" if os.environ.get("QUICK") == "1" else "10"))
TAG = "v1" if TC.BACKEND == "v1" else TC.VERSION
OUT = os.environ.get("OUT", str(ARTISTRY / "logits" / "tabpfn25_ladder"))
os.makedirs(OUT, exist_ok=True)

def sizes_for(pool_n):
    """Ladder rungs for a training split of pool_n rows: numeric sizes below pool_n, then 'full'. Deduplicated, ordered."""
    out = []
    for s in LADDER:
        k = pool_n if s == "full" else int(s)
        if TC.BACKEND == "v1" and k > 1000: continue      # TabPFN v1 cannot take more than 1,000 context rows
        lab = "full" if k >= pool_n else str(k)
        if lab not in out: out.append(lab)
    return out

INFO = env_info()
print("Stage 1:", TC.describe().replace("context=full", "context ladder " + " ".join(LADDER)), "|", INFO["device"], flush=True)
json.dump(INFO, open(os.path.join(OUT, "environment.json"), "w"), indent=1)

warmed = False
for ds in DATASETS:
    for seed in DATASET_SEEDS[ds][:NSEEDS]:
        D = load_dataset(ds, seed)
        perm = np.random.default_rng(seed).permutation(D.train)          # one shuffle per seed -> nested contexts
        for lab in sizes_for(len(D.train)):
            f = os.path.join(OUT, f"{ds}_seed{seed}_ctx{lab}.npz")
            if os.path.exists(f): continue
            k = len(perm) if lab == "full" else int(lab)
            ctx = perm[:k]
            if not warmed:      # load weights / compile kernels once, outside the timed region
                W = TC.FrozenTabPFN(D.X[ctx[:200]], D.y[ctx[:200]], seed=0, categorical_idx=D.cat_idx); W.logits(D.X[D.test[:50]]); warmed = True
            t0 = time.perf_counter(); M = TC.FrozenTabPFN(D.X[ctx], D.y[ctx], seed=D.model_seed, categorical_idx=D.cat_idx); t_fit = time.perf_counter() - t0
            z, t_pred = {}, {}
            for part in ("discovery", "selection", "test"):
                t0 = time.perf_counter(); z[part] = M.logits(D.X[getattr(D, part)]); t_pred[part] = time.perf_counter() - t0
            n_pred = sum(len(getattr(D, p)) for p in t_pred)
            np.savez_compressed(f, zd=z["discovery"], zs=z["selection"], zt=z["test"], ctx_idx=ctx,
                                yd=D.y[D.discovery], ys=D.y[D.selection], yt=D.y[D.test],
                                fit_seconds=t_fit, predict_seconds=sum(t_pred.values()), rows_predicted=n_pred,
                                predict_seconds_test=t_pred["test"], context_rows=k, info=json.dumps(INFO))
            print(f"  {ds:7s} seed {seed:5d} ctx {lab:>5s} ({k:5d} rows): fit {t_fit:6.1f}s, predict {n_pred} rows {sum(t_pred.values()):7.1f}s", flush=True)
        del D
print("Stage 1 done ->", OUT)
