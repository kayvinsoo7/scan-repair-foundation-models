"""Clean speed measurement: seconds for TabPFN to predict 1,000 rows at each context size, on THIS machine.
Uses the real Credit Default features. Run it on each device you want in the paper (Colab GPU, Colab CPU, your Mac).

  python time_ladder.py                       # auto device (NVIDIA GPU -> Apple GPU -> CPU)
  TABPFN_DEVICE=cpu python time_ladder.py     # force the CPU
Environment: LADDER (same default as ladder_logits.py), REPEATS (default 2; the median is kept), MAX_SECONDS
(default 1800: larger contexts are skipped once one measurement takes longer than this).
Writes timing_<device>.csv next to this script (or into $OUT)."""
import os, time, json, re, numpy as np, pandas as pd
os.environ.setdefault("TABPFN_BACKEND", "modern"); os.environ["TABPFN_CONTEXT"] = "full"
from common import *
import tabpfn_common as TC
from ladder_data import load_dataset
import ladder_env as LE

LADDER = os.environ.get("LADDER", "500 1000 1500 2000 3000 4000 6000 8000 10000 full").split()
REPEATS = int(os.environ.get("REPEATS", "2")); MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "1800"))
D = load_dataset("credit", 42001)
perm = np.random.default_rng(0).permutation(D.train); test = D.test[:1000]
info = LE.env_info(); dev = info["device"]
print(f"Timing TabPFN {info['version']} on {dev}", flush=True)
W = TC.FrozenTabPFN(D.X[perm[:200]], D.y[perm[:200]], seed=0, categorical_idx=D.cat_idx); W.logits(D.X[test[:50]])   # warm-up
rows = []
for s in LADDER:
    k = len(perm) if s == "full" else int(s)
    if k > len(perm) or (TC.BACKEND == "v1" and k > 1000): continue
    fit, pred = [], []
    for r in range(REPEATS):
        t = time.perf_counter(); M = TC.FrozenTabPFN(D.X[perm[:k]], D.y[perm[:k]], seed=r, categorical_idx=D.cat_idx); fit.append(time.perf_counter() - t)
        t = time.perf_counter(); M.logits(D.X[test]); pred.append(time.perf_counter() - t)
    rows.append(dict(device=dev, version=info["version"], context=s, context_rows=k, fit_seconds=float(np.median(fit)),
                     predict_seconds_per_1000=float(np.median(pred)), total_seconds_per_1000=float(np.median(np.add(fit, pred)))))
    print(f"  context {k:6d}: fit {rows[-1]['fit_seconds']:7.2f}s  predict 1,000 rows {rows[-1]['predict_seconds_per_1000']:7.2f}s", flush=True)
    out = os.path.join(os.environ.get("OUT", str(ARTISTRY / "timing")), "timing_" + re.sub(r"[^A-Za-z0-9]+", "_", dev).strip("_") + ".csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    if rows[-1]["total_seconds_per_1000"] > MAX_SECONDS: print("  (stopping: next context would be too slow)"); break
print("Saved", out)
