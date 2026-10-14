"""Discovery-size arm for the foundation models (run on a GPU): TabPFN-2.5 or TabICLv2 with the whole 6,000-row
training split as context, predicting all 9,600 discovery + selection + test rows of synth8x4d4 / synth1x4d4.
The 2,400-row variant (d1) is the first 2,400 discovery rows of the same predictions (make_discovery_small.py).
Usage: MODEL=tabicl|tabpfn python base_fm_discovery.py      (env: OUT)"""
import os, time, json, platform, numpy as np, pandas as pd
MODEL = os.environ.get("MODEL", "tabicl")
if MODEL == "tabpfn": os.environ.setdefault("TABPFN_BACKEND", "modern"); os.environ["TABPFN_CONTEXT"] = "full"
from common import *
from ladder_data import load_dataset, DATASET_SEEDS
import torch
OUT = os.environ.get("OUT", str(ARTISTRY / "logits" / "discovery" / MODEL)); os.makedirs(OUT, exist_ok=True)
dev = ("cuda:" + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu:" + platform.machine()
info = dict(backend=MODEL, device=dev, torch=torch.__version__); print(MODEL, "on", dev, flush=True)
def make(D, seed):
    if MODEL == "tabpfn":
        import tabpfn_common as TC
        return TC.FrozenTabPFN(D.X[D.train], D.y[D.train], seed=seed)
    from tabicl import TabICLClassifier
    kw = dict(model_path=os.environ["TABICL_MODEL_PATH"], allow_auto_download=False) if os.environ.get("TABICL_MODEL_PATH") \
         else dict(checkpoint_version="tabicl-classifier-v2-20260212.ckpt")
    m = TabICLClassifier(**kw, random_state=seed); m.fit(pd.DataFrame(D.X[D.train]), D.y[D.train]); return m
def logits(m, X):
    if MODEL == "tabpfn": return m.logits(X)
    p = np.clip(m.predict_proba(pd.DataFrame(X))[:, list(m.classes_).index(1)], 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
for ds in ["synth8x4d4", "synth1x4d4"]:
    for seed in DATASET_SEEDS[ds]:
        f = os.path.join(OUT, f"{ds}_seed{seed}_ctxfull.npz")
        if os.path.exists(f): continue
        D = load_dataset(ds, seed); t0 = time.perf_counter(); m = make(D, seed); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); z = {p: logits(m, D.X[getattr(D, p)]) for p in ("discovery", "selection", "test")}
        tp = time.perf_counter() - t0
        np.savez_compressed(f, zd=z["discovery"], zs=z["selection"], zt=z["test"], ctx_idx=D.train, yd=D.y[D.discovery],
                            ys=D.y[D.selection], yt=D.y[D.test], fit_seconds=t_fit, predict_seconds=tp,
                            rows_predicted=sum(len(getattr(D, p)) for p in ("discovery", "selection", "test")),
                            predict_seconds_test=0.0, context_rows=len(D.train), info=json.dumps(info))
        print(f"  {MODEL} {ds} seed {seed}: {t_fit + tp:.1f}s", flush=True)
print("done ->", OUT)
