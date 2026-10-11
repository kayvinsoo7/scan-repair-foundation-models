"""TabICLv2 (Qu et al., ICML 2026) as a frozen base model on exactly the ladder's splits; saves logits in the
ladder's .npz format so corrections.py applies every correction unchanged.
The model's context is the whole training split (the same rows TabPFN gets at full context), default settings,
categorical columns passed as pandas 'category'. GPU recommended (Colab T4: well under an hour).
Usage: python base_tabicl.py            (env: OUT, DATASETS, SEEDS, CONTEXT=full|<rows>)"""
import os, time, json, platform, numpy as np, pandas as pd
from common import *
from ladder_data import load_dataset, DATASET_SEEDS
import torch
from tabicl import TabICLClassifier

OUT = os.environ.get("OUT", str(ARTIFACTS / "logits" / "tabicl")); os.makedirs(OUT, exist_ok=True)
DATASETS = os.environ.get("DATASETS", "synth8 synth1 credit").split()
NSEEDS = int(os.environ.get("SEEDS", "10"))
# one context ("full" = whole training split, as in Table I) or a ladder of nested contexts, e.g.
# LADDER="500 1000 1500 2000 3000 4000 6000 8000 10000 full" (same nested rows as the TabPFN-2.5 ladder)
CONTEXTS = os.environ.get("LADDER", os.environ.get("CONTEXT", "full")).split()
CKPT = os.environ.get("TABICL_CHECKPOINT", "tabicl-classifier-v2-20260212.ckpt")
KW = dict(checkpoint_version=CKPT)
if os.environ.get("TABICL_MODEL_PATH"): KW = dict(model_path=os.environ["TABICL_MODEL_PATH"], allow_auto_download=False)   # testing only
dev = ("cuda:" + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else \
      ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu:" + platform.machine())
info = dict(backend="tabicl", version=CKPT, device=dev, torch=torch.__version__)
print("TabICL on", dev, "| checkpoint", CKPT, flush=True)
json.dump(info, open(os.path.join(OUT, "environment.json"), "w"), indent=1)

def frame(D, idx):
    df = pd.DataFrame(D.X[idx], columns=D.feature_names)
    for c in D.categorical_names: df[c] = df[c].astype(int).astype("category")
    return df

warmed = False
for ds in DATASETS:
    for seed in DATASET_SEEDS[ds][:NSEEDS]:
        D = load_dataset(ds, seed)
        perm = np.random.default_rng(seed).permutation(D.train)       # same nested order as the TabPFN ladder
        for CONTEXT in CONTEXTS:
            if CONTEXT != "full" and int(CONTEXT) >= len(perm): continue    # e.g. 8k/10k on the 6k synthetic training split
            ctx = perm if CONTEXT == "full" else perm[:int(CONTEXT)]
            lab = "full" if len(ctx) == len(D.train) else str(len(ctx))
            f = os.path.join(OUT, f"{ds}_seed{seed}_ctx{lab}.npz")
            if os.path.exists(f): continue
            if not warmed:
                w = TabICLClassifier(**KW, random_state=0); w.fit(frame(D, ctx[:200]), D.y[ctx[:200]])
                w.predict_proba(frame(D, D.test[:50])); warmed = True
            t0 = time.perf_counter(); m = TabICLClassifier(**KW, random_state=D.model_seed)
            m.fit(frame(D, ctx), D.y[ctx]); t_fit = time.perf_counter() - t0
            z, tp = {}, 0.0
            for part in ("discovery", "selection", "test"):
                t0 = time.perf_counter(); p = m.predict_proba(frame(D, getattr(D, part)))[:, list(m.classes_).index(1)]
                tp += time.perf_counter() - t0; p = np.clip(p, 1e-6, 1 - 1e-6); z[part] = np.log(p / (1 - p))
            np.savez_compressed(f, zd=z["discovery"], zs=z["selection"], zt=z["test"], ctx_idx=ctx, yd=D.y[D.discovery],
                                ys=D.y[D.selection], yt=D.y[D.test], fit_seconds=t_fit, predict_seconds=tp,
                                rows_predicted=len(D.discovery) + len(D.selection) + len(D.test), predict_seconds_test=0.0,
                                context_rows=len(ctx), info=json.dumps(info))
            print(f"  {ds:7s} seed {seed:5d} ctx {lab}: fit {t_fit:5.1f}s predict {tp:6.1f}s", flush=True)
print("TabICL done ->", OUT)
