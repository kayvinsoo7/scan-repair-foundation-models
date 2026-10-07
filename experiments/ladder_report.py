"""STAGE 3: every table and figure we might want from the context ladder. Pick what goes in the paper later.

  python ladder_report.py <stage-2 folder (contains posthoc.csv, rows/)> [<out folder>] [timing_*.csv ...]

Gains are test log-loss improvements over the frozen model at the SAME context, x1e-3 (positive = better).
Outputs (CSV for every table, plus Markdown for reading and LaTeX for the paper):
  T1_gain           gain per dataset x context x method: mean, 95% CI, paired t-test p, wins/harms
  T2_absolute       test log-loss, AUROC, ECE, multicalibration error (MCE sigma) per method x context
  T3_recovery       how much of the full-context improvement each CPU fix recovers at a small context
  T4_cost           TabPFN seconds per context (stage-1 device) + optional per-device timings + post-hoc CPU seconds
  T5_size           size of each fix (parameters / leaves / rules) and CPU seconds
  T6_scan           scan diagnostics: terms, audited rules, support, near-global, planted-subgroup Jaccard
  T7_planted        |bias| on the planted subgroup (synthetic, ground truth known)
  T8_attribution    does the scan point at the rows that more context later fixes? (per-row analysis)
  T9_reprompt       the compute alternative: re-prompting TabPFN with more context. For each fix at context c, the
                    context the frozen model needs to match it ("equivalent context") and the TabPFN seconds that costs
  T0_headline       per dataset x context, the paper's core measures in one row: base LL, repaired LL, subgroup residual
                    before/after, scan intervention rate, oracle subgroup gain and the share of it the scan gets,
                    detected-rule Jaccard, harm count, TabPFN and scan seconds
  fig_ladder_<ds>   log-loss vs context rows and vs TabPFN seconds
"""
import os, sys, glob, re, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith(".csv") else os.path.join(IN, "report")
TIMING = [a for a in sys.argv[2:] if a.endswith(".csv")]
os.makedirs(OUT, exist_ok=True)
P = pd.read_csv(os.path.join(IN, "posthoc.csv"))
P["ctx_order"] = np.where(P.context == "full", 10**9, pd.to_numeric(P.context, errors="coerce"))
DS_NAME = {"synth8": "Synthetic, planted subgroup (OR 8)", "synth1": "Synthetic, no signal (OR 1)", "credit": "Credit Default"}
METHODS = ["frozen", "temperature", "Platt", "beta", "isotonic", "histogram binning", "residual tree", "scan", "scan a=0.01", "Platt->scan",
           "boosted residual (XGB)", "oracle subgroup", "MCGrad"]
METHODS = [m for m in METHODS if m in set(P.method)] + sorted(set(P.method) - set(METHODS))
COLOR = {"frozen": "#52514e", "Platt": "#2a78d6", "isotonic": "#eb6834", "scan": "#1baf7a", "residual tree": "#eda100",
         "Platt->scan": "#e87ba4", "temperature": "#008300", "histogram binning": "#4a3aa7", "scan a=0.01": "#e34948",
         "beta": "#8a8a8a", "boosted residual (XGB)": "#b0b0b0", "oracle subgroup": "#000000", "MCGrad": "#c8c8c8"}
MARK = {"frozen": "o", "Platt": "s", "isotonic": "^", "scan": "D", "residual tree": "v", "Platt->scan": "P", "temperature": "x",
        "histogram binning": "*", "scan a=0.01": "d", "beta": "1", "boosted residual (XGB)": "2", "oracle subgroup": "_", "MCGrad": "3"}
def ctxs(df): return list(df.sort_values("ctx_order").context.drop_duplicates())

def save(df, name, floatfmt=".2f", index=False):
    df.to_csv(os.path.join(OUT, name + ".csv"), index=index)
    with open(os.path.join(OUT, name + ".md"), "w") as fh:
        try: fh.write(df.to_markdown(index=index, floatfmt=floatfmt))
        except ImportError: fh.write(df.to_string(index=index))
    try: df.to_latex(os.path.join(OUT, name + ".tex"), index=index, float_format=lambda x: f"{x:{floatfmt}}", na_rep="--")
    except Exception: pass

def paired(a, b):
    """mean of (a-b) x1e3, 95% CI, paired t p-value, wins, harms."""
    d = (np.asarray(a) - np.asarray(b)) * 1e3; n = len(d)
    if n == 0: return dict(n=0)
    m = d.mean(); h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    p = stats.ttest_1samp(d, 0).pvalue if n > 1 and d.std() > 0 else np.nan
    return dict(n=n, gain=m, ci_lo=m - h, ci_hi=m + h, p=p, wins=int((d > 1e-9).sum()), harms=int((d < -1e-9).sum()))


# ---- T0 headline: the core measures, one row per dataset x context ----
devs = P[P.method == "frozen"].groupby("dataset").tabpfn_device.nunique()
for ds, n in devs.items():
    if n > 1: print(f"WARNING: {ds} was run on {n} different devices; TabPFN seconds mix devices:", sorted(P[P.dataset == ds].tabpfn_device.unique()))
t0 = []
for (ds, c), g in P.groupby(["dataset", "context"]):
    w = g.pivot(index="seed", columns="method", values="test_ll"); sc = g[g.method == "scan"]; fz = g[g.method == "frozen"]
    r = dict(dataset=ds, context=c, context_rows=int(fz.context_rows.iloc[0]), seeds=len(w), base_ll=w["frozen"].mean(),
             repaired_ll=w["scan"].mean(), scan_gain=1e3 * (w["frozen"] - w["scan"]).mean(),
             scan_harmed=f"{int(((w['frozen'] - w['scan']) < -1e-9).sum())}/{len(w)}",
             scan_intervened=f"{int((sc.terms > 0).sum())}/{len(sc)}", scan_rules=sc.terms.mean(),
             tabpfn_seconds=(fz.tabpfn_fit_seconds + fz.tabpfn_predict_seconds).mean(), scan_cpu_seconds=sc.method_seconds.mean())
    if "planted_abs_bias" in g and g.planted_abs_bias.notna().any():
        r["subgroup_residual_frozen"] = fz.planted_abs_bias.mean(); r["subgroup_residual_scan"] = sc.planted_abs_bias.mean()
    if "oracle subgroup" in w:
        og = (w["frozen"] - w["oracle subgroup"]); r["oracle_subgroup_gain"] = 1e3 * og.mean()
        r["scan_share_of_oracle_pct"] = 100 * (w["frozen"] - w["scan"]).mean() / og.mean() if og.mean() > 1e-3 else np.nan   # undefined when there is nothing planted (OR 1)
    if "best_term_jaccard" in sc and sc.best_term_jaccard.notna().any(): r["detected_rule_jaccard"] = sc.best_term_jaccard.mean()
    if "audited_pass" in sc and sc.audited_pass.notna().any(): r["audited_rules"] = f"{int(sc.audited_pass.sum())}/{int(sc.terms.sum())}"
    t0.append(r)
T0 = pd.DataFrame(t0).sort_values(["dataset", "context_rows"]); save(T0, "T0_headline", ".3f")

W = P.pivot_table(index=["dataset", "context", "seed"], columns="method", values="test_ll")

# ---- T1 gain vs frozen at the same context, and vs Platt ----
t1 = []
for (ds, c), g in W.groupby(level=[0, 1]):
    for m in METHODS[1:]:
        if m in g: t1.append(dict(dataset=ds, context=c, method=m, **paired(g["frozen"], g[m]),
                                  **{f"vs_Platt_{k}": v for k, v in paired(g["Platt"], g[m]).items() if k in ("gain", "p")} if "Platt" in g else {}))
T1 = pd.DataFrame(t1); T1["ctx_order"] = T1.context.map(lambda c: 10**9 if c == "full" else int(c))
T1 = T1.sort_values(["dataset", "ctx_order", "method"]).drop(columns="ctx_order"); save(T1, "T1_gain_long", ".3f")
for ds in T1.dataset.unique():
    w = T1[T1.dataset == ds].assign(cell=lambda d: d.apply(lambda r: f"{r.gain:+.1f} [{r.ci_lo:+.1f},{r.ci_hi:+.1f}]" + ("*" if r.p < 0.05 else ""), axis=1))
    w = w.pivot(index="context", columns="method", values="cell").reindex(ctxs(P[P.dataset == ds]))[[m for m in METHODS[1:] if m in set(w.method)]]
    save(w.reset_index(), f"T1_gain_{ds}")

# ---- T2 absolute metrics ----
agg = {k: "mean" for k in ["test_ll", "auroc", "ece", "brier", "mce_sigma", "mce_relative"] if k in P}
T2 = P.groupby(["dataset", "context", "method"]).agg(agg).reset_index()
T2["ctx_order"] = T2.context.map(lambda c: 10**9 if c == "full" else int(c)); T2 = T2.sort_values(["dataset", "ctx_order", "method"]).drop(columns="ctx_order")
save(T2, "T2_absolute_long", ".4f")
for key in agg:
    for ds in T2.dataset.unique():
        w = T2[T2.dataset == ds].pivot(index="context", columns="method", values=key).reindex(ctxs(P[P.dataset == ds]))
        save(w[[m for m in METHODS if m in w]].reset_index(), f"T2_{key}_{ds}", ".4f")

# ---- T3 recovery: share of the full-context improvement recovered by a CPU fix at context c ----
t3 = []
for ds in W.index.get_level_values(0).unique():
    if "full" not in W.loc[ds].index.get_level_values(0): continue
    full = W.loc[(ds, "full")]["frozen"]
    for c in ctxs(P[P.dataset == ds]):
        if c == "full": continue
        g = W.loc[(ds, c)]; s = g.index.intersection(full.index); gap = (g.loc[s, "frozen"] - full.loc[s])
        for m in METHODS[1:]:
            if m not in g: continue
            got = g.loc[s, "frozen"] - g.loc[s, m]
            t3.append(dict(dataset=ds, context=c, method=m, full_context_gain=1e3 * gap.mean(), fix_gain=1e3 * got.mean(),
                           recovered_pct=100 * got.mean() / gap.mean() if gap.mean() > 1e-6 else np.nan,
                           fix_beats_full_seeds=f"{int((g.loc[s, m] < full.loc[s]).sum())}/{len(s)}",
                           fix_minus_full=1e3 * (g.loc[s, m] - full.loc[s]).mean()))
T3 = pd.DataFrame(t3)
if len(T3):
    save(T3, "T3_recovery_long")
    for ds in T3.dataset.unique():
        w = T3[T3.dataset == ds].pivot(index="context", columns="method", values="recovered_pct").reindex([c for c in ctxs(P[P.dataset == ds]) if c != "full"])
        w.insert(0, "full-context gain (x1e-3)", T3[T3.dataset == ds].groupby("context").full_context_gain.first())
        save(w.reset_index(), f"T3_recovery_{ds}", ".0f")

# ---- T4 cost ----
F = P[P.method == "frozen"].groupby(["dataset", "context"]).agg(context_rows=("context_rows", "first"), device=("tabpfn_device", "first"),
        tabpfn_fit_s=("tabpfn_fit_seconds", "mean"), tabpfn_predict_s=("tabpfn_predict_seconds", "mean"), rows_predicted=("tabpfn_rows_predicted", "first")).reset_index()
F["tabpfn_s_per_1000_rows"] = 1000 * (F.tabpfn_fit_s + F.tabpfn_predict_s) / F.rows_predicted
F = F.sort_values(["dataset", "context_rows"]); save(F, "T4_cost_tabpfn")
if TIMING:
    TT = pd.concat([pd.read_csv(t) for t in TIMING]); save(TT, "T4_cost_devices")
    w = TT.pivot_table(index="context_rows", columns="device", values="total_seconds_per_1000"); save(w.reset_index(), "T4_cost_devices_wide")
T5 = P.groupby(["dataset", "method"]).agg(median_params=("n_params", "median"), median_cpu_seconds=("method_seconds", "median"),
                                          max_cpu_seconds=("method_seconds", "max")).reset_index()
save(T5, "T5_size_and_cpu_seconds")

# ---- T6 scan diagnostics, T7 planted bias ----
S = P[P.method.isin(["scan", "scan a=0.01", "Platt->scan"])]
cols = {"terms": ("terms", "mean"), "intervened": ("terms", lambda x: f"{int((x > 0).sum())}/{len(x)}"),
        "median_support": ("median_support", "median"), "near_global_terms": ("near_global_terms", "sum")}
if "audited_pass" in S: cols["audited_pass"] = ("audited_pass", lambda x: x.sum(min_count=1))
if "best_term_jaccard" in S: cols["best_term_jaccard"] = ("best_term_jaccard", "mean")
T6 = S.groupby(["dataset", "context", "method"]).agg(**cols).reset_index(); save(T6, "T6_scan_diagnostics")
if "planted_abs_bias" in P and (P.dataset == "synth8").any():
    w = P[P.dataset == "synth8"].pivot_table(index="context", columns="method", values="planted_abs_bias").reindex(ctxs(P[P.dataset == "synth8"]))
    save(w[[m for m in METHODS if m in w]].reset_index(), "T7_planted_bias", ".3f")

# ---- T8 attribution: does the scan at context c point at what more context later fixes? ----
def rowll(y, z):
    p = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
t8 = []
for f in sorted(glob.glob(os.path.join(IN, "rows", "*_ctxfull.npz"))):
    ds, seed = re.match(r"(.+)_seed(\d+)_ctxfull", os.path.basename(f)).groups(); Rf = np.load(f); y = Rf["yt"]; lf = rowll(y, Rf["frozen"])
    rng = np.random.default_rng(int(seed))
    for g in sorted(glob.glob(os.path.join(IN, "rows", f"{ds}_seed{seed}_ctx*.npz"))):
        c = re.search(r"_ctx(\w+)\.npz", g).group(1)
        if c == "full": continue
        R = np.load(g); assert (R["yt"] == y).all()
        lc = rowll(y, R["frozen"]); imp = lc - lf                        # per-row improvement from more context
        dz_full = Rf["frozen"].astype(float) - R["frozen"].astype(float)  # how more context moved each logit
        for m, key in [("scan", "mask_0.05_0"), ("residual tree", None), ("Platt", None), ("isotonic", None)]:
            if m not in R: continue
            corr = R[m].astype(float) - R["frozen"].astype(float)
            mask = R[key].astype(bool) if key else np.abs(corr) > 1e-6
            r = dict(dataset=ds, seed=int(seed), context=c, method=m, flagged_share=mask.mean())
            if mask.any() and imp.sum() > 0:
                r["improvement_share_in_flagged"] = imp[mask].sum() / imp.sum()
                r["concentration"] = r["improvement_share_in_flagged"] / mask.mean()
                rand = rng.random(len(y)) < mask.mean()
                r["concentration_random_mask"] = (imp[rand].sum() / imp.sum()) / max(rand.mean(), 1e-9)
            if mask.sum() > 2:
                r["direction_corr"] = float(np.corrcoef(corr[mask], dz_full[mask])[0, 1]) if corr[mask].std() > 0 and dz_full[mask].std() > 0 else np.nan
                r["sign_agreement"] = float(np.mean(np.sign(corr[mask]) == np.sign(dz_full[mask])))
                denom = imp[mask].sum()
                r["fix_reproduces_pct_of_full_in_flagged"] = 100 * (lc[mask] - rowll(y, R[m].astype(float))[mask]).sum() / denom if denom > 0 else np.nan
            t8.append(r)
T8 = pd.DataFrame(t8)
if len(T8):
    save(T8, "T8_attribution_long", ".3f")
    T8s = T8.groupby(["dataset", "context", "method"]).agg(seeds=("seed", "count"), flagged_share=("flagged_share", "mean"),
            **{k: (k, "mean") for k in ["concentration", "concentration_random_mask", "direction_corr", "sign_agreement",
                                         "fix_reproduces_pct_of_full_in_flagged"] if k in T8}).reset_index()
    save(T8s, "T8_attribution", ".2f")


# ---- T9 re-prompting: how much more context would the frozen model need to match each CPU fix? ----
t9 = []
for ds in P.dataset.unique():
    d = P[P.dataset == ds]
    fr_ = d[d.method == "frozen"].groupby("context_rows").agg(ll=("test_ll", "mean"), sec=("tabpfn_predict_seconds", "mean"),
                                                               fs=("tabpfn_fit_seconds", "mean")).sort_index()
    fr_["sec"] = fr_.sec + fr_.fs
    ks, lls, secs = fr_.index.to_numpy(float), fr_.ll.to_numpy(), fr_.sec.to_numpy()
    for (m, k), g in d[d.method != "frozen"].groupby(["method", "context_rows"]):
        target = g.test_ll.mean(); here = fr_.loc[k]
        eq, eq_sec = np.nan, np.nan
        for i in range(len(ks) - 1):                      # first context where the frozen curve reaches the fix (log-linear interpolation)
            if ks[i] < k: continue
            if lls[i] <= target: eq, eq_sec = ks[i], secs[i]; break
            if lls[i + 1] <= target:
                w = (lls[i] - target) / (lls[i] - lls[i + 1])
                eq = float(np.exp(np.log(ks[i]) + w * (np.log(ks[i + 1]) - np.log(ks[i])))); eq_sec = float(secs[i] + w * (secs[i + 1] - secs[i])); break
        if eq != eq and target < lls[-1]: eq = np.inf          # better than the frozen model at the largest context run
        t9.append(dict(dataset=ds, method=m, context_rows=int(k), fix_ll=target, frozen_ll_here=here.ll,
                       equivalent_context_rows=eq, context_multiplier=eq / k if eq == eq else np.nan,
                       tabpfn_seconds_here=here.sec, tabpfn_seconds_equivalent=eq_sec, fix_cpu_seconds=g.method_seconds.mean(),
                       full_context_frozen_ll=lls[-1], beats_full_context=bool(target < lls[-1])))
T9 = pd.DataFrame(t9)
if len(T9):
    save(T9.sort_values(["dataset", "method", "context_rows"]), "T9_reprompt_long", ".3f")
    for ds in T9.dataset.unique():
        w = T9[T9.dataset == ds].pivot(index="context_rows", columns="method", values="equivalent_context_rows")
        w = w[[m for m in METHODS if m in w]].apply(lambda col: col.map(lambda v: "> full" if v == np.inf else ("" if v != v else f"{v:,.0f}")))
        save(w.reset_index(), f"T9_equivalent_context_{ds}")

# ---- figures: log-loss vs context rows, and vs TabPFN seconds (two panels, never a dual axis) ----
FIG_METHODS = os.environ.get("FIG_METHODS", "frozen,Platt,isotonic,scan").split(",")
for ds in P.dataset.unique():
    d = P[P.dataset == ds]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for m in [m for m in FIG_METHODS if m in set(d.method)]:
        g = d[d.method == m].groupby("context_rows").agg(ll=("test_ll", "mean"), sd=("test_ll", "std"), n=("seed", "count"),
                                                         sec=("tabpfn_predict_seconds", "mean"), fs=("tabpfn_fit_seconds", "mean")).reset_index()
        g["sec"] = g.sec + g.fs; h = stats.t.ppf(0.975, np.maximum(g.n - 1, 1)) * g.sd / np.sqrt(g.n)
        kw = dict(color=COLOR.get(m, "#888"), marker=MARK.get(m, "o"), ms=4, lw=1.6, ls="--" if m == "frozen" else "-", label=m)
        for a, x in ((ax[0], g.context_rows), (ax[1], g.sec)):
            a.plot(x, g.ll, **kw); a.fill_between(x, g.ll - h, g.ll + h, color=kw["color"], alpha=0.12, lw=0)
    ax[0].set_xscale("log"); ax[1].set_xscale("log")
    ax[0].set_xlabel("TabPFN context (labelled rows)"); ax[1].set_xlabel(f"TabPFN seconds per run ({d.tabpfn_device.iloc[0].split(':')[0]})")
    ax[0].set_ylabel("test log-loss")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=7.5); fig.suptitle(DS_NAME.get(ds, ds), fontsize=9); fig.tight_layout()
    for ext_ in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"fig_ladder_{ds}.{ext_}"), dpi=200)
    plt.close(fig)
print("Report written to", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
