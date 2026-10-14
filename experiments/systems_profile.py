"""Runtime: where the time goes, and how it scales; original vs compiled exact scorer.
Same data, same seed, same frozen scanner; outputs checked identical."""
import os
import sys, time, cProfile, pstats, io, numpy as np, pandas as pd
from common import *
os.chdir(out_dir(ARTISTRY / "systems"))   # all outputs (and credit_xgb_params.json) live here
import fastexact as F, profile_scan as P
import mdss_treescan_stage1 as M
def set_scorer(fast):
    for mod in list(sys.modules.values()):
        if mod is None: continue
        cur = getattr(mod, "exact_subgroup_logit_correction", None)
        if cur in (F._orig, F.fast_exact): mod.exact_subgroup_logit_correction = F.fast_exact if fast else F._orig
F.fast_exact(np.array([0.,1.]*30), np.zeros(60), np.ones(60, bool))  # JIT warm-up
rows = []
for n, p in [(6000,30),(12000,30),(24000,30),(48000,30),(12000,60),(12000,120)]:
    for fast in (False, True):
        set_scorer(fast)
        el, L, d, sp = P.run(n=n, p=p, seed=1)
        rows.append(dict(n=n, n_discovery=len(sp.discovery), p=p, scorer="compiled" if fast else "original", seconds=el,
                         terms=len(L.bases_), coefs=";".join(f"{c:.10f}" for c in np.asarray(L.coefficients_, float))))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv("cost_scaling.csv", index=False)
# breakdown at the default size
prof = {}
for fast in (False, True):
    set_scorer(fast)
    cond, d, sp, m = P.setup(n=12000, p=30, seed=1)
    Lr = P.make_final_scanner(cond, discovery_n=len(sp.discovery), selection_n=len(sp.selection))
    pr = cProfile.Profile(); pr.enable()
    Lr.fit(d.X[sp.discovery], d.y[sp.discovery], m.logits(d.X[sp.discovery]), d.X[sp.selection], d.y[sp.selection], m.logits(d.X[sp.selection]))
    pr.disable(); st = pstats.Stats(pr)
    def cum(fn_sub):
        return sum(v[3] for k, v in st.stats.items() if fn_sub in f"{k[0]}:{k[2]}")
    total = sum(v[2] for v in st.stats.values())
    prof["compiled" if fast else "original"] = dict(
        total=cum("v2_3.py:fit"),
        exact_scorer=cum(":exact_subgroup_logit_correction") + cum("fastexact.py:fast_exact"),
        legacy_fixed_bin=cum(":propose_legacy_fixed_bin_rule"),
        adaptive_search=cum(":scan_candidates"),
        selection_and_refit=cum("v2_3.py:fit") - cum("v2_3.py:_propose"))
pd.DataFrame(prof).T.to_csv("cost_breakdown.csv"); print(pd.DataFrame(prof).T)
