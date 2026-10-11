"""Seconds for TabICLv2 to score 1,000 rows at each context size, on THIS machine (same protocol as timing_tabpfn.py:
real Credit Default features, nested contexts, fit + predict, median of REPEATS).
  python timing_tabicl.py                                   # GPU if available (Colab T4 in the paper)
  DEVICE=cpu TABICL_MODEL_PATH=<random-weight ckpt> python timing_tabicl.py
      CPU timing with a random-weight checkpoint of the same architecture (make_timing_checkpoint_tabicl.py):
      identical computation, meaningless predictions.
Env: LADDER, REPEATS (2), MAX_SECONDS (1800), TABICL_BATCH (the CPU timing used 1 to fit in 7 GB of memory), OUT (default <artifacts>/timing). Writes timing_tabicl_<device>.csv."""
import os, time, re, platform, numpy as np, pandas as pd
from common import *
from ladder_data import load_dataset
import torch
from tabicl import TabICLClassifier
LADDER = os.environ.get("LADDER", "500 1000 1500 2000 3000 4000 6000 8000 10000 full").split()
REPEATS = int(os.environ.get("REPEATS", "2")); MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "1800"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
dev = ("cuda:" + torch.cuda.get_device_name(0)) if DEVICE.startswith("cuda") else f"cpu ({torch.get_num_threads()} threads)"
KW = dict(device=DEVICE, checkpoint_version="tabicl-classifier-v2-20260212.ckpt")
if os.environ.get("TABICL_MODEL_PATH"): KW.update(model_path=os.environ["TABICL_MODEL_PATH"], allow_auto_download=False)
if os.environ.get("TABICL_BATCH"): KW.update(batch_size=int(os.environ["TABICL_BATCH"]))   # ensemble members per batch; 1 = least memory (CPU)
D = load_dataset("credit", 42001)
perm = np.random.default_rng(0).permutation(D.train); test = D.test[:1000]
def frame(idx):
    df = pd.DataFrame(D.X[idx], columns=D.feature_names)
    for c in D.categorical_names: df[c] = df[c].astype(int).astype("category")
    return df
w = TabICLClassifier(**KW, random_state=0); w.fit(frame(perm[:200]), D.y[perm[:200]]); w.predict_proba(frame(test[:50]))  # warm-up
print("Timing TabICLv2 on", dev, flush=True); rows = []
out = os.path.join(os.environ.get("OUT", str(ARTIFACTS / "timing")), "timing_tabicl_" + os.environ.get("TAG", "") + re.sub(r"[^A-Za-z0-9]+", "_", dev).strip("_") + ".csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
for s in LADDER:
    k = len(perm) if s == "full" else int(s)
    if k > len(perm): continue
    fit, pred = [], []
    for r in range(REPEATS):
        t = time.perf_counter(); m = TabICLClassifier(**KW, random_state=r); m.fit(frame(perm[:k]), D.y[perm[:k]]); fit.append(time.perf_counter() - t)
        t = time.perf_counter(); m.predict_proba(frame(test)); pred.append(time.perf_counter() - t)
    rows.append(dict(device=dev, version="tabicl-v2", context=s, context_rows=k, fit_seconds=float(np.median(fit)),
                     predict_seconds_per_1000=float(np.median(pred)), total_seconds_per_1000=float(np.median(np.add(fit, pred)))))
    print(f"  context {k:6d}: fit {rows[-1]['fit_seconds']:7.2f}s  predict 1,000 rows {rows[-1]['predict_seconds_per_1000']:7.2f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    if rows[-1]["total_seconds_per_1000"] > MAX_SECONDS: print("  (stopping: next context would be too slow)"); break
print("Saved", out)
