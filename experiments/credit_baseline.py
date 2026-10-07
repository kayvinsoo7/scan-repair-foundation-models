"""Sanity check: reproduce the original study's Credit Default XGBoost result (base LL 0.4341). Paper, Reproducibility:
"reproduces the earlier tree-model baseline (test LL 0.4338 versus 0.4341)". Writes <artifacts>/systems/credit_baseline.csv."""
import os
import json, time, numpy as np, pandas as pd
from common import *
os.chdir(out_dir(ARTIFACTS / "systems"))   # all outputs (and credit_xgb_params.json) live here
import fastexact as F; F.install()
bundle = ext.load_credit_default(DATA, allow_download=False)
split0 = ext.make_extended_split(bundle, seed=42001)
params, _ = ext.select_xgb_params_extended(bundle, split0, seed=42001)
print("params", params, flush=True); json.dump(params, open("credit_xgb_params.json", "w"))
rows = []
for seed in range(42001, 42011):
    split = ext.make_extended_split(bundle, seed=seed)
    base = core.fit_xgb_base(bundle, split, params, seed=seed)
    Xd, Xs, Xt = (bundle.X.iloc[s] for s in (split.discovery, split.selection, split.test))
    yd, ys, yt = (bundle.y[s] for s in (split.discovery, split.selection, split.test))
    zd, zs, zt = base.logits(Xd), base.logits(Xs), base.logits(Xt)
    prep = core.CompactScanPreprocessor(bundle.numeric_columns, bundle.categorical_columns); prep.fit(bundle.X.iloc[split.train])
    L = core.make_realdata_scanner(prep.feature_types, discovery_n=len(Xd), selection_n=len(Xs), seed=seed)
    t = time.perf_counter(); L.fit(prep.transform(Xd), yd, zd, prep.transform(Xs), ys, zs); sec = time.perf_counter() - t
    zr = L.repair_logits(prep.transform(Xt), zt)
    rows.append(dict(seed=seed, base_ll=ll(yt, core.sigmoid(zt)), repair_ll=ll(yt, core.sigmoid(zr)), terms=len(L.bases_), sec=sec,
                     n_train=len(split.train), n_disc=len(yd), n_sel=len(ys), n_test=len(yt)))
    print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv("credit_baseline.csv", index=False)
print("MEAN base_ll %.6f  delta %.6f  interv %.1f  (paper: 0.434147, -0.000095, 0.3)" % (d.base_ll.mean(), (d.base_ll - d.repair_ll).mean(), (d.terms > 0).mean()))
