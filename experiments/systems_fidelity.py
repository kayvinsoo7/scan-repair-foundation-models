"""(1) Does the G^2/2H screen miss the best exact threshold?  (2) Does the logit-histogram score preserve rankings?"""
import os
import sys, numpy as np, pandas as pd, json
from common import *
os.chdir(out_dir(ARTIFACTS / "systems"))   # all outputs (and credit_xgb_params.json) live here
import fastexact as F; F.install()
import profile_scan as P
from fastscore import LogitBins, batched_gain, all_threshold_gains
from mdss_treescan_stage1 import exact_subgroup_logit_correction as exact_ref
exact = F.fast_exact
gaps = []
for seed in range(1, 6):
    cond, d, sp, m = P.setup(seed=seed)
    X = d.X[sp.discovery]; y = d.y[sp.discovery].astype(float); z = m.logits(X)
    L = P.make_final_scanner(cond, discovery_n=len(y), selection_n=len(sp.selection)); sc = L.scanner; sc.fit_binner(X)
    lb = LogitBins(z, 64); rng = np.random.default_rng(seed)
    parents = [np.ones(len(y), bool)] + [np.all(X[:, :k] > d.subgroup_threshold, axis=1) for k in range(1, 5)]
    for _ in range(15):
        j = rng.integers(X.shape[1]); parents.append(X[:, j] > rng.uniform(0.1, 0.5))
    for pm in parents:
        for j in range(X.shape[1]):
            cur = sc._numeric_candidates(X, y, z, pm, j)
            r = all_threshold_gains(X[:, j], y, z, lb.bin, lb.B, np.flatnonzero(pm), sc.min_size)
            if r is None or not cur: continue
            ch, thr, gl, gh, _, _, idx = r
            best = 0.0
            for g, dr in ((gl, "le"), (gh, "gt")):
                for i in np.argsort(-g)[:3]:
                    mk = np.zeros(len(y), bool); mk[idx[:ch[i]] if dr == "le" else idx[ch[i]:]] = True
                    s = exact(y, z, mk, min_size=sc.min_size, min_events=3, min_nonevents=3)
                    if s: best = max(best, s.gain)
            got = cur[0][2].gain
            gaps.append(dict(seed=seed, best=best, got=got, gap=max(best - got, 0.0)))
g = pd.DataFrame(gaps); g.to_csv("fidelity_screen.csv", index=False)
cap = np.where(g.best > 0, g.got / g.best, 1.0)
out = dict(states=len(g), miss_small=float((g.gap > 1e-3).mean()), miss_large=float((g.gap > 0.5).mean()),
           max_gap=float(g.gap.max()), p05_captured=float(np.quantile(cap, 0.05)))
# histogram fidelity
cond, d, sp, m = P.setup(seed=1); X = d.X[sp.discovery]; y = d.y[sp.discovery].astype(float); z = m.logits(X)
rng = np.random.default_rng(0); masks = []
while len(masks) < 2000:
    k = rng.integers(1, 6); f = rng.choice(X.shape[1], k, replace=False); t = rng.uniform(0.05, 0.6, k); s_ = rng.random(k) < 0.5
    mk = np.ones(len(y), bool)
    for j, tt, ss in zip(f, t, s_): mk &= (X[:, j] > tt) if ss else (X[:, j] <= 1 - tt)
    if 40 <= mk.sum() and 3 <= y[mk].sum() <= mk.sum() - 3: masks.append(mk)
eg = np.array([exact_ref(y, z, mk, min_size=1, min_events=0, min_nonevents=0).gain for mk in masks])
lb = LogitBins(z, 64); S = [lb.stats(y, z, mk) for mk in masks]
hg, _ = batched_gain(np.array([s[0] for s in S]), np.array([s[1] for s in S]), np.array([s[2] for s in S]))
out.update(hist_n=len(masks), hist_max_err=float(np.abs(hg - eg).max()), hist_same_best=bool(np.argmax(hg) == np.argmax(eg)),
           hist_top50_overlap=len(set(np.argsort(-eg)[:50]) & set(np.argsort(-hg)[:50])))
json.dump(out, open("fidelity.json", "w"), indent=1); print(out)
