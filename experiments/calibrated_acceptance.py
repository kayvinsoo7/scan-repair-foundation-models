"""Calibrated acceptance test for Structured Scan Repair (Monte-Carlo null).

Question. The scan picks its candidate rule on the discovery rows, ranks the survivors with the two selection
halves, and then applies a nominal 1-df likelihood-ratio test (alpha = 0.05) on the same selection rows. Because the
selection rows were already used to choose the rule, the nominal test may reject a true null more often than 5%.
How often, and does a test calibrated by simulation change the repairs?

Null model. If the frozen model F is correct, y ~ Bernoulli(sigmoid(F(x))) independently. For one configuration
(dataset, seed, context) we draw B replicates y* on the discovery + selection rows from the frozen logits, run the
scan's FIRST round exactly as in the paper but with alpha = 1 (so the selected candidate's statistic is always
recorded), and keep  S* = 2 * full-selection gain  of the selected candidate (0 if no candidate passes the two-half
stability check, which also means "no correction").

Calibrated test. T = the ceil(0.95 (B + 1))-th smallest of the B null statistics; a candidate is accepted when its
statistic exceeds T, i.e. alpha_cal = chi2.sf(T, 1) (alpha_cal = 1 when T = 0: then the stability check alone already
keeps the null rejection rate at or below 5%). The real labels are then refitted with alpha_cal, all other settings
unchanged; the same alpha_cal is used in every repair round (only round 1 is simulated).

Settings fixed before running: TabPFN-2.5 ladder logits; synth8 (planted subgroup, odds x8) and synth1 (no planted
effect); seeds 1-10; contexts 2000 and full; B = 200; SSR only (no Platt step).

Outputs (in <artistry>/systems/calibrated_acceptance/ unless OUT is set):
  null/<ds>_seed<s>_ctx<c>.csv   one row per null replicate (resumable: finished replicates are skipped)
  fits/<ds>_seed<s>_ctx<c>.csv   the real-label refit, nominal and calibrated
  summary.csv                    all fits, one row per configuration and test
Usage: python calibrated_acceptance.py            (SHARD=0/2 and SHARD=1/2 split the configurations over 2 processes)
       FAST=0 B=3 ONLY=synth8_seed1_ctx2000 ...   time a few replicates with the original (uncompiled) scorer
"""
import os, sys, glob, time, math, warnings, logging, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from common import *
FAST = os.environ.get("FAST", "1") == "1"
if FAST:
    import fastexact as F; F.install()
    import fastscan as FS; FS.install()
from scipy.stats import chi2
from final_scanner_block_b import make_final_scanner
from ladder_data import load_dataset

B = int(os.environ.get("B", "200"))
DATASETS, SEEDS, CONTEXTS = ("synth8", "synth1"), range(1, 11), ("2000", "full")
LOGITS = ARTISTRY / "logits" / "tabpfn25_ladder"
OUT = out_dir(os.environ.get("OUT", ARTISTRY / "systems" / "calibrated_acceptance"))
for d in ("null", "fits"): (OUT / d).mkdir(exist_ok=True)
sig = lambda z: 1 / (1 + np.exp(-z))

def jac(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 0.0

def first_round_stat(D, Xd, yd, zd, Xs, ys, zs):
    L = make_final_scanner(D.cond, discovery_n=len(zd), selection_n=len(zs)); L.alpha = 1.0; L.repair_rounds = 1
    L.fit(Xd, yd, zd, Xs, ys, zs)
    g = L.history_[0].full_selection_gain if L.history_ else np.nan
    return 2.0 * float(g) if np.isfinite(g) and g > 0 else 0.0, L.history_[0].selected_rule_depth if L.history_ else 0

def refit(D, alpha, Xd, yd, zd, Xs, ys, zs, Xt, yt, zt):
    L = make_final_scanner(D.cond, discovery_n=len(zd), selection_n=len(zs)); L.alpha = alpha
    t = time.perf_counter(); L.fit(Xd, yd, zd, Xs, ys, zs); sec = time.perf_counter() - t
    zr = L.repair_logits(Xt, zt); Phi = L._Phi(Xt) if L.bases_ else np.zeros((len(zt), 0))
    tr = D.truth[D.test]; p = sig(zr)
    r = dict(alpha=alpha, terms=len(L.bases_), fit_seconds=sec, test_ll=ll(yt, p), frozen_test_ll=ll(yt, sig(zt)),
             planted_abs_bias=abs(float(np.mean(yt[tr] - p[tr]))), frozen_planted_abs_bias=abs(float(np.mean(yt[tr] - sig(zt)[tr]))))
    r["gain_x1e3"] = 1e3 * (r["frozen_test_ll"] - r["test_ll"])
    r["best_term_jaccard"] = max([jac(c > 0.5, tr) for c in Phi.T], default=0.0) if D.odds > 1 else np.nan
    r["first_round_stat"] = 2.0 * float(L.history_[0].full_selection_gain) if L.history_ and np.isfinite(L.history_[0].full_selection_gain) else 0.0
    return r

configs = [(ds, s, c) for ds in DATASETS for s in SEEDS for c in CONTEXTS]
if os.environ.get("ONLY"): configs = [k for k in configs if f"{k[0]}_seed{k[1]}_ctx{k[2]}" in os.environ["ONLY"].split(",")]
if os.environ.get("SHARD"):
    i, n = map(int, os.environ["SHARD"].split("/")); configs = configs[i::n]
print(f"{len(configs)} configurations, B = {B}, fast scorer = {FAST}", flush=True)

for ds, seed, ctx in configs:
    tag = f"{ds}_seed{seed}_ctx{ctx}"
    D = load_dataset(ds, seed); Z = np.load(LOGITS / f"{tag}.npz")
    zd, zs, zt = (Z[k].astype(float) for k in ("zd", "zs", "zt"))
    yd, ys, yt = D.y[D.discovery], D.y[D.selection], D.y[D.test]
    assert (Z["yd"] == yd).all() and (Z["ys"] == ys).all() and (Z["yt"] == yt).all()
    Xd, Xs, Xt = D.X[D.discovery], D.X[D.selection], D.X[D.test]
    # ---- null replicates (resumable) ----
    nf = OUT / "null" / f"{tag}.csv" if FAST else OUT / "null_original_scorer" / f"{tag}.csv"
    nf.parent.mkdir(exist_ok=True)
    done = pd.read_csv(nf) if nf.exists() else pd.DataFrame(columns=["b", "stat", "depth", "seconds"])
    have = set(done["b"].astype(int))
    for b in range(B):
        if b in have: continue
        rng = np.random.default_rng([20271113, int(ds[5:]), seed, 0 if ctx == "full" else int(ctx), b])
        yd_s, ys_s = (rng.random(len(zd)) < sig(zd)).astype(int), (rng.random(len(zs)) < sig(zs)).astype(int)
        t = time.perf_counter(); st, dep = first_round_stat(D, Xd, yd_s, zd, Xs, ys_s, zs); sec = time.perf_counter() - t
        pd.DataFrame([dict(b=b, stat=st, depth=dep, seconds=sec)]).to_csv(nf, mode="a", header=not nf.exists(), index=False)
    if not FAST: print(f"  {tag}: original scorer, {B} replicates timed", flush=True); continue
    null = pd.read_csv(nf).drop_duplicates("b").sort_values("b")
    assert len(null) >= B
    s = np.sort(null["stat"].to_numpy()[:B]); k = min(B, math.ceil(0.95 * (B + 1)))
    T = float(s[k - 1]); alpha_cal = float(chi2.sf(T, 1)) if T > 0 else 1.0
    nominal_T = float(chi2.isf(0.05, 1))
    common_ = dict(dataset=ds, seed=seed, context=ctx, B=B, null_threshold=T, nominal_threshold=nominal_T,
                   null_reject_rate_nominal=float(np.mean(s > nominal_T)), null_frac_zero=float(np.mean(s == 0)),
                   null_seconds_per_replicate=float(null["seconds"].mean()))
    rows = [dict(common_, test="nominal", **refit(D, 0.05, Xd, yd, zd, Xs, ys, zs, Xt, yt, zt)),
            dict(common_, test="calibrated", **refit(D, alpha_cal, Xd, yd, zd, Xs, ys, zs, Xt, yt, zt))]
    pd.DataFrame(rows).to_csv(OUT / "fits" / f"{tag}.csv", index=False)
    print(f"  {tag}: null rej(nominal)={common_['null_reject_rate_nominal']:.3f}  T={T:.2f} alpha_cal={alpha_cal:.4f}  "
          f"terms {rows[0]['terms']}->{rows[1]['terms']}  gain {rows[0]['gain_x1e3']:+.2f}->{rows[1]['gain_x1e3']:+.2f}", flush=True)

fits = sorted(glob.glob(str(OUT / "fits" / "*.csv")))
if fits: pd.concat([pd.read_csv(f) for f in fits]).to_csv(OUT / "summary.csv", index=False)
print("done ->", OUT)
