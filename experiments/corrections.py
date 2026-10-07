"""Apply every post-hoc correction to saved base-model logits (any base model; CPU only).

All methods see exactly the same labelled rows: DISCOVERY + SELECTION (the scan searches on discovery and
validates on selection; methods with their own validation protocol use it on the same rows). Test rows are
used only for evaluation.

Methods
  frozen                       TabPFN as is
  temperature                  temperature scaling of the logit (1 parameter)
  Platt                        logistic recalibration of the logit (2 parameters)
  beta                         beta calibration (3 parameters)
  isotonic                     isotonic regression on the logit
  histogram binning            15 equal-mass bins of the logit
  residual tree                one shallow tree (depth <= 3, chosen on selection) adding log-odds offsets: the "why not a tree?" baseline
  scan                         Structured Scan Repair, exactly as in the paper (alpha = 0.05)
  scan a=0.01                  same, stricter acceptance test
  Platt->scan                  Platt fitted on discovery, then the scan on the recalibrated model's residuals
  boosted residual (XGB)       gradient-boosted trees on the features starting from TabPFN's logit, early-stopped on selection
  oracle subgroup              synthetic only: the true planted subgroup with its best log-odds shift (upper bound for one rule)

Usage:  python corrections.py <logits folder> <output folder>
Writes <output>/parts/*.csv (one per stage-1 file; reruns skip finished ones), <output>/posthoc.csv (all rows)
and <output>/rows/*.npz (per-row test logits of every method, for the attribution analysis)."""
import os, sys, glob, time, json, re, warnings, logging, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
os.environ.setdefault("TABPFN_BACKEND", "none")
from common import *
import fastexact as F; F.install()
import fastscan as FS; FS.install()
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from structured_scan_calibration_benchmark import fit_beta_calibration
from final_scanner_block_b import make_final_scanner
import final_realdata_suite as fr
from ladder_data import load_dataset

IN = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(IN, "posthoc")
AUDIT = os.environ.get("AUDIT", "1") == "1"; N_PLACEBO = int(os.environ.get("N_PLACEBO", "500"))
for d in ("parts", "rows"): os.makedirs(os.path.join(OUT, d), exist_ok=True)

sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def jac(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 0.0
def ece(y, p, bins=15):
    o = np.argsort(p); e = 0.0
    for b in np.array_split(o, bins): e += len(b) * abs(y[b].mean() - p[b].mean())
    return e / len(y)

# ---------------- methods: each returns (test logits, info dict) ----------------
def m_platt(c):
    m = LogisticRegression(C=1e6).fit(c.zfit[:, None], c.yfit); return m.decision_function(c.zt[:, None]), dict(n_params=2)
def m_beta(c):
    f = fit_beta_calibration(sig(c.zfit), c.yfit); return logit(f(sig(c.zt))), dict(n_params=3)
def m_temperature(c):
    """Temperature scaling (Guo et al. 2017): divide the logit by one fitted constant."""
    from scipy.optimize import minimize_scalar
    z, y = c.zfit, c.yfit
    nll = lambda logT: float(np.mean(np.logaddexp(0, z / np.exp(logT)) - y * z / np.exp(logT)))
    T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))
    return c.zt / T, dict(n_params=1, temperature=T)
def m_histbin(c, bins=15):
    """Histogram binning (Zadrozny & Elkan 2001): equal-mass bins of the logit, each bin predicts its event rate."""
    edges = np.unique(np.quantile(c.zfit, np.linspace(0, 1, bins + 1))[1:-1])
    b_fit, b_t = np.searchsorted(edges, c.zfit), np.searchsorted(edges, c.zt)
    rate = np.array([(c.yfit[b_fit == k].sum() + 0.5) / ((b_fit == k).sum() + 1.0) for k in range(len(edges) + 1)])
    return logit(rate[b_t]), dict(n_params=int(len(edges) + 1))
def m_tree(c):
    """One shallow residual tree: leaves are log-odds offsets added to TabPFN's logit (Newton leaf values),
    grown on DISCOVERY; depth 1-3 chosen by SELECTION log-loss (depth 0 = no change if nothing helps)."""
    D = c.D; best = (ll(c.ys, sig(c.zs)), 0, None)
    dtr = xgb.DMatrix(D.X[D.discovery], label=c.yd, base_margin=c.zd)
    pd_ = sig(c.zd)                                                       # same minimum leaf size as the scan's minimum subgroup
    min_size = max(35, round(0.015 * len(c.yd))) if D.kind == "synth" else max(40, round(0.01 * len(c.yd)))
    mcw = min_size * float(np.mean(pd_ * (1 - pd_)))                      # ... expressed as the hessian weight xgboost uses
    for depth in (1, 2, 3):
        bst = xgb.train(dict(objective="binary:logistic", max_depth=depth, eta=1.0, min_child_weight=mcw,
                             reg_lambda=1.0, seed=D.seed, verbosity=0, nthread=1), dtr, num_boost_round=1)
        zs = bst.predict(xgb.DMatrix(D.X[D.selection], base_margin=c.zs), output_margin=True)
        s = ll(c.ys, sig(zs))
        if s < best[0]: best = (s, depth, bst)
    if best[2] is None: return c.zt, dict(n_params=0, depth=0)
    zt = best[2].predict(xgb.DMatrix(D.X[D.test], base_margin=c.zt), output_margin=True)
    return zt, dict(n_params=int(best[2].get_dump()[0].count("leaf=")), depth=best[1])
def m_oracle(c):
    """Oracle subgroup repair (synthetic only): the TRUE planted subgroup with its best single log-odds shift,
    fitted by exact Bernoulli maximum likelihood on discovery+selection. The most a one-rule repair could achieve."""
    from scipy.optimize import brentq
    D = c.D; m_fit = D.truth[np.concatenate([D.discovery, D.selection])]; m_t = D.truth[D.test]
    z, y = c.zfit[m_fit], c.yfit[m_fit]
    score = lambda d: float(np.sum(y - sig(z + d)))
    delta = brentq(score, -10, 10) if score(-10) > 0 > score(10) else 0.0
    return c.zt + delta * m_t, dict(n_params=1, oracle_delta=delta)
def m_isotonic(c):
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(c.zfit, c.yfit)
    return logit(m.predict(c.zt)), dict(n_params=int(len(m.X_thresholds_)))
def m_xgb(c):
    D = c.D
    dtr = xgb.DMatrix(D.X[D.discovery], label=c.yd, base_margin=c.zd); dva = xgb.DMatrix(D.X[D.selection], label=c.ys, base_margin=c.zs)
    bst = xgb.train(dict(objective="binary:logistic", eval_metric="logloss", max_depth=3, eta=0.05, min_child_weight=5, subsample=0.8,
                         colsample_bytree=0.8, seed=D.seed, verbosity=0, nthread=1), dtr, num_boost_round=400, evals=[(dva, "sel")],
                    early_stopping_rounds=30, verbose_eval=False)
    it = bst.best_iteration + 1
    zt = bst.predict(xgb.DMatrix(D.X[D.test], base_margin=c.zt), iteration_range=(0, it), output_margin=True)
    leaves = sum(t.count("leaf=") for t in bst.get_dump()[:it])
    return zt, dict(n_params=int(leaves), trees=int(it))
def mc_frame(D, idx, z):
    df = pd.DataFrame(D.X[idx], columns=D.feature_names); df["pred"] = sig(z); return df
def scan_inputs(c):
    """Scanner features: raw numeric matrix for the synthetic data, the paper's compact preprocessor for real data."""
    D = c.D
    if D.kind == "synth":
        mk = lambda: make_final_scanner(D.cond, discovery_n=len(D.discovery), selection_n=len(D.selection))
        return mk, D.X[D.discovery], D.X[D.selection], D.X[D.test], None
    prep = core.CompactScanPreprocessor(D.numeric_names, D.categorical_names); prep.fit(D.frame.iloc[D.train])
    Sd, Ss, St = (prep.transform(D.frame.iloc[s]) for s in (D.discovery, D.selection, D.test))
    mk = lambda: core.make_realdata_scanner(prep.feature_types, discovery_n=len(D.discovery), selection_n=len(D.selection), seed=D.seed)
    return mk, Sd, Ss, St, prep
def make_scan(alpha, two_stage):
    def run(c):
        mk, Sd, Ss, St, prep = c.scan_in
        zd, zs, zt = c.zd, c.zs, c.zt
        if two_stage:                                       # Platt on DISCOVERY only, so selection stays untouched
            m = LogisticRegression(C=1e6).fit(zd[:, None], c.yd); zd, zs, zt = (m.decision_function(z[:, None]) for z in (zd, zs, zt))
        L = mk(); L.alpha = alpha
        L.fit(Sd, c.yd, zd, Ss, c.ys, zs)
        zr = L.repair_logits(St, zt)
        Phi = L._Phi(St) if L.bases_ else np.zeros((len(zt), 0))
        info = dict(n_params=len(L.bases_) + (2 if two_stage else 0), terms=len(L.bases_),
                    median_support=float(np.median(Phi.mean(0))) if Phi.shape[1] else np.nan,
                    near_global_terms=int((Phi.mean(0) >= 0.95).sum()) if Phi.shape[1] else 0)
        if c.D.truth is not None and c.D.odds > 1:
            info["best_term_jaccard"] = max([jac(col > 0.5, c.D.truth[c.D.test]) for col in Phi.T], default=0.0)
        if AUDIT and not two_stage and L.bases_ and prep is not None:
            mech = fr.classify_mechanism(ext.mechanistic_term_audit(L, prep, St, c.yt, zt, dataset=c.D.name, seed=c.D.seed,
                                                                   fixed_test=True, n_placebo=N_PLACEBO))
            info["audited_pass"] = int(mech.localized_mechanistic_success.sum())
        c.extra_rows[f"mask_{alpha}_{int(two_stage)}"] = (Phi.sum(1) > 0) if Phi.shape[1] else np.zeros(len(zt), bool)
        return zr, info
    return run

METHODS = [("frozen", lambda c: (c.zt, dict(n_params=0))),
           ("temperature", m_temperature), ("Platt", m_platt), ("beta", m_beta), ("isotonic", m_isotonic), ("histogram binning", m_histbin),
           ("residual tree", m_tree),
           ("boosted residual (XGB)", m_xgb),
           ("scan", make_scan(0.05, False)), ("scan a=0.01", make_scan(0.01, False)), ("Platt->scan", make_scan(0.05, True)),
           ("oracle subgroup", m_oracle)]                       # synthetic only (skipped on real data)
ONLY = os.environ.get("METHODS")
if ONLY: METHODS = [m for m in METHODS if m[0] in ONLY.split(",")]

def metrics(c, z):
    D = c.D; p = sig(z); yt = c.yt
    r = dict(test_ll=ll(yt, p), brier=brier_score_loss(yt, p), auroc=roc_auc_score(yt, p), ece=ece(yt, p))
    if D.truth is not None:
        tr = D.truth[D.test]
        r["planted_abs_bias"] = abs(float(np.mean(yt[tr] - p[tr]))) if tr.any() else np.nan
        r["oracle_ll"] = ll(yt, sig(D.true_logits[D.test]))
    return r

class Ctx: pass
files = sorted(glob.glob(os.path.join(IN, os.environ.get("PATTERN", "*_seed*_ctx*.npz"))))   # PATTERN: rerun a subset
if os.environ.get("SHARD"):                      # e.g. SHARD=0/2 and SHARD=1/2 in two processes (set NUMBA_NUM_THREADS=1)
    i, n = map(int, os.environ["SHARD"].split("/")); files = files[i::n]
print(f"Stage 2: {len(files)} stage-1 files in {IN}; methods: {', '.join(m for m, _ in METHODS)}", flush=True)
cache = {}
for f in files:
    ds, seed, lab = re.match(r"(.+)_seed(\d+)_ctx(\w+)\.npz", os.path.basename(f)).groups(); seed = int(seed)
    part = os.path.join(OUT, "parts", os.path.basename(f).replace(".npz", ".csv"))
    if os.path.exists(part): continue
    Z = np.load(f)
    if (ds, seed) not in cache: cache.clear(); cache[(ds, seed)] = load_dataset(ds, seed)
    c = Ctx(); c.D = D = cache[(ds, seed)]
    c.zd, c.zs, c.zt = Z["zd"].astype(float), Z["zs"].astype(float), Z["zt"].astype(float)
    c.yd, c.ys, c.yt = D.y[D.discovery], D.y[D.selection], D.y[D.test]
    assert (Z["yd"] == c.yd).all() and (Z["ys"] == c.ys).all() and (Z["yt"] == c.yt).all(), f"labels in {f} do not match the split"
    c.zfit, c.yfit = np.concatenate([c.zd, c.zs]), np.concatenate([c.yd, c.ys])
    c.extra_rows = {}; c.scan_in = scan_inputs(c)
    info = json.loads(str(Z["info"])) if "info" in Z else {}
    base = dict(dataset=ds, seed=seed, context=lab, context_rows=int(Z["context_rows"]), tabpfn_version=info.get("version", "?"),
                tabpfn_device=info.get("device", "?"), tabpfn_fit_seconds=float(Z["fit_seconds"]),
                tabpfn_predict_seconds=float(Z["predict_seconds"]), tabpfn_rows_predicted=int(Z["rows_predicted"]))
    rows, store = [], {}
    for name, fn in METHODS:
        if name == "oracle subgroup" and D.truth is None: continue
        t = time.perf_counter(); z, inf = fn(c); sec = time.perf_counter() - t
        store[name] = np.asarray(z, np.float32)
        rows.append(dict(base, method=name, method_seconds=sec, **inf, **metrics(c, z)))
    np.savez_compressed(os.path.join(OUT, "rows", os.path.basename(f)), yt=c.yt, test_idx=D.test,
                        **{k: v for k, v in store.items()}, **c.extra_rows)
    pd.DataFrame(rows).to_csv(part, index=False)
    fz = rows[0]["test_ll"]
    print(f"  {ds:7s} seed {seed:5d} ctx {lab:>5s}: " + "  ".join(f"{r['method']}={1e3*(fz-r['test_ll']):+.1f}" for r in rows[1:]), flush=True)
parts = sorted(glob.glob(os.path.join(OUT, "parts", "*.csv")))
if parts: pd.concat([pd.read_csv(p) for p in parts]).to_csv(os.path.join(OUT, "posthoc.csv"), index=False)
print("Stage 2 done ->", OUT)
