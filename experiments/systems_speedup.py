"""Runtime of the frozen scanner in three implementations: original NumPy, compiled scorer,
compiled + batched search. Same data and seed; fitted repairs must be identical."""
import os
import sys, time, cProfile, pstats, numpy as np, pandas as pd
from common import *
os.chdir(out_dir(ARTISTRY / "systems"))   # all outputs (and credit_xgb_params.json) live here
import fastexact as F, fastscan as FS, profile_scan as P, numba
NT = numba.get_num_threads()
def set_mode(mode):
    numba.set_num_threads(1 if mode == "batched_1t" else NT)
    for mod in list(sys.modules.values()):
        if mod is None: continue
        cur = getattr(mod, "exact_subgroup_logit_correction", None)
        if cur in (F._orig, F.fast_exact): mod.exact_subgroup_logit_correction = F._orig if mode == "original" else F.fast_exact
    FS.install() if mode.startswith("batched") else FS.uninstall()
# JIT warm-up of all kernels
F.fast_exact(np.array([0., 1.] * 30), np.zeros(60), np.ones(60, bool)); set_mode("batched"); P.run(n=3000, p=12, seed=9)
import hashlib
def fingerprint(L, X):
    phi = L._Phi(np.asarray(X)); h = hashlib.md5(np.ascontiguousarray(phi).tobytes()).hexdigest()[:12]
    return ";".join(f"{c:.12f}" for c in np.asarray(L.coefficients_, float)) + "|" + h
rows = []
for n, p in [(6000, 30), (12000, 30), (24000, 30), (48000, 30), (12000, 60), (12000, 120)]:
    for mode in ("original", "compiled", "batched_1t", "batched"):
        set_mode(mode); el, L, d, sp = P.run(n=n, p=p, seed=1)
        rows.append(dict(n_discovery=len(sp.discovery), p=p, impl=mode, seconds=el, terms=len(L.bases_), fp=fingerprint(L, d.X)))
        print({k: v for k, v in rows[-1].items() if k != "fp"}, flush=True)
        pd.DataFrame(rows).to_csv("cost4_scaling.csv", index=False)
# component breakdown at the default size
prof = {}
for mode in ("original", "compiled", "batched"):
    set_mode(mode); cond, d, sp, m = P.setup(n=12000, p=30, seed=1)
    Lr = P.make_final_scanner(cond, discovery_n=len(sp.discovery), selection_n=len(sp.selection))
    pr = cProfile.Profile(); pr.enable()
    Lr.fit(d.X[sp.discovery], d.y[sp.discovery], m.logits(d.X[sp.discovery]), d.X[sp.selection], d.y[sp.selection], m.logits(d.X[sp.selection]))
    pr.disable(); st = pstats.Stats(pr)
    cum = lambda s: sum(v[3] for k, v in st.stats.items() if s in f"{k[0]}:{k[2]}")
    prof[mode] = dict(total=cum("v2_3.py:fit"), legacy_fixed_bin=cum(":propose_legacy_fixed_bin_rule"),
                      adaptive_search=cum(":scan_candidates"), selection_and_refit=cum("v2_3.py:fit") - cum("v2_3.py:_propose"))
pd.DataFrame(prof).T.to_csv("cost4_breakdown.csv"); print(pd.DataFrame(prof).T.round(2))
# real-data equivalence (categorical + numeric features): Credit Default, XGBoost base, seeds where the paper intervened
bundle = ext.load_credit_default(DATA, allow_download=False); import json
params = json.load(open("credit_xgb_params.json")); real = []
for seed in (42008, 42009, 42003):
    split = ext.make_extended_split(bundle, seed=seed); base = core.fit_xgb_base(bundle, split, params, seed=seed)
    prep = core.CompactScanPreprocessor(bundle.numeric_columns, bundle.categorical_columns); prep.fit(bundle.X.iloc[split.train])
    Xd, Xs = (prep.transform(bundle.X.iloc[s]) for s in (split.discovery, split.selection))
    zd, zs = base.logits(bundle.X.iloc[split.discovery]), base.logits(bundle.X.iloc[split.selection])
    for mode in ("original", "batched"):
        set_mode(mode); L = core.make_realdata_scanner(prep.feature_types, discovery_n=len(zd), selection_n=len(zs), seed=seed)
        t = time.perf_counter(); L.fit(Xd, bundle.y[split.discovery], zd, Xs, bundle.y[split.selection], zs)
        real.append(dict(seed=seed, impl=mode, seconds=time.perf_counter() - t, terms=len(L.bases_), fp=fingerprint(L, Xs)))
        print({k: v for k, v in real[-1].items() if k != "fp"}, flush=True)
pd.DataFrame(real).to_csv("cost4_realdata_equivalence.csv", index=False)
