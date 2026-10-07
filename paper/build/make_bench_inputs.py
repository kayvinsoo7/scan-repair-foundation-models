"""Paper inputs for the cross-model block: four base models at their best (+ the discovery-size arm).
  python make_bench_inputs.py
Writes paper/latex/{numbers_bench.tex, tab_bench.tex, tab_disc.tex, fig_disc.pdf} and results/disc_summary.csv."""
import os, sys, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import ART, LATEX, RESULTS, LADDER as LADDER_DIR
PAPER = str(LATEX); LADDER = LADDER_DIR / "posthoc.csv"
MODELS = [("xgboost", "XGBoost"), ("catboost", "CatBoost"), ("tabpfn", "TabPFN-2.5"), ("tabicl", "TabICLv2")]
MAC = {"xgboost": "Xgb", "catboost": "Cat", "tabpfn": "Pfn", "tabicl": "Icl"}
DSM = {"synth8": "Pl", "synth1": "Nu", "credit": "Cr"}
def load_main():
    out = []
    for k, n in MODELS:
        if k == "tabpfn": d = pd.read_csv(LADDER); d = d[d.context.astype(str) == "full"]
        else: d = pd.read_csv(ART / "corrections" / k / "posthoc.csv")
        out.append(d.assign(model=n, mkey=k))
    return pd.concat(out, ignore_index=True)
def load_x4():
    return pd.concat([pd.read_csv(ART / "corrections" / "discovery" / k / "posthoc.csv").assign(model=n, mkey=k)
                      for k, n in MODELS], ignore_index=True)
B, X = load_main(), load_x4()
M = {}
def mac(k, v): M[k] = v
def gain(g, m, base="frozen"):
    w = g.pivot(index="seed", columns="method", values="test_ll"); d = 1e3 * (w[base] - w[m]).values
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else 1.0
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), p, int((d < -1e-9).sum()), d.mean() - h, d.mean() + h
def fmt(v, p): return f"{v:+.1f}" + ("*" if p < 0.05 else "")

# ---------- Table: main corrections, four models, three datasets ----------
DSN = [("synth8", "Planted"), ("synth1", "\\begin{tabular}[c]{@{}l@{}}No planted\\\\effect\\end{tabular}"), ("credit", "Credit")]
MAIN = [("Platt", "Platt"), ("beta", "Beta"), ("isotonic", "Iso."), ("scan", "Scan"), ("Platt->scan", "P$\\to$S")]
L = ["\\begin{table}[t]", "\\centering",
     "\\caption{Every base model at its best: test log-loss (LL) and improvement from each correction ($\\times10^{-3}$, "
     "mean of 10 seeds; * paired $t$-test $p<0.05$; bold: best). Trees are tuned by 5-fold cross-validation; the "
     "foundation models see the whole training split as context. P$\\to$S: Platt, then scan.}",
     "\\label{tab:bench5}", "\\footnotesize\\setlength{\\tabcolsep}{3.2pt}",
     "\\begin{tabular}{llr" + "r" * len(MAIN) + "}", "\\toprule",
     "Data & Base model & LL & " + " & ".join(n for _, n in MAIN) + " \\\\", "\\midrule"]
for i, (ds, dn) in enumerate(DSN):
    for j, (k, n) in enumerate(MODELS):
        g = B[(B.dataset == ds) & (B.mkey == k)]; gs = {m: gain(g, m) for m, _ in MAIN}
        best = max(gs, key=lambda m: gs[m][0])
        cells = [("\\textbf{%s}" % fmt(*gs[m][:2])) if (m == best and gs[m][0] > 0) else fmt(*gs[m][:2]) for m, _ in MAIN]
        ll = g[g.method == "frozen"].test_ll.mean()
        L.append((f"\\multirow{{4}}{{*}}{{{dn}}}" if j == 0 else "") + f" & {n} & {ll:.4f} & " + " & ".join(cells) + " \\\\")
        for m, mn in [("Platt", "Platt"), ("scan", "Scan"), ("Platt->scan", "PS"), ("beta", "Beta"), ("isotonic", "Iso")]:
            mac(f"B{MAC[k]}{DSM[ds]}{mn}", f"{gs[m][0]:.1f}" if m in gs else f"{gain(g, m)[0]:.1f}")
        mac(f"B{MAC[k]}{DSM[ds]}LL", f"{ll:.4f}")
        mac(f"B{MAC[k]}{DSM[ds]}AUC", f"{g[g.method == 'frozen'].auroc.mean():.3f}")
        sc = g[g.method == "scan"]; mac(f"B{MAC[k]}{DSM[ds]}Rep", f"{int((sc.terms > 0).sum())}/{len(sc)}")
    L.append("\\midrule" if i < len(DSN) - 1 else "\\bottomrule")
L += ["\\end{tabular}", "\\end{table}"]
open(RESULTS / "tab_bench_main_corrections.tex", "w").write("\n".join(L) + "\n")

# ---------- discovery arm ----------
L = ["\\begin{table}[t]", "\\centering",
     "\\caption{Discovery-power experiment on a separate $n=19{,}200$ draw. Within each model and seed, the frozen "
     "predictor, selection set and test set are fixed; Scan receives 2{,}400 or 9{,}600 discovery rows. Gain: mean test "
     "log-loss improvement ($\\times10^{-3}$); Oracle share: fraction of the best one-rule gain; Jacc.: overlap with the "
     "planted subgroup. Null gain (harmful seeds) is from the matched no-planted-effect control.}",
     "\\label{tab:disc}", "\\footnotesize\\setlength{\\tabcolsep}{2.4pt}",
     "\\begin{tabular}{lcrrrcr}", "\\toprule",
     "Base model & Discovery & \\begin{tabular}[b]{@{}c@{}}With\\\\rule\\end{tabular} & Gain & \\begin{tabular}[b]{@{}c@{}}Oracle\\\\share\\end{tabular} & Jacc. & "
     "\\begin{tabular}[b]{@{}c@{}}Null gain\\\\(harmful seeds)\\end{tabular} \\\\", "\\midrule"]
disc_rows = []
for j, (k, n) in enumerate(MODELS):
    for size, tag in [("2.4k", "d1"), ("9.6k", "d4")]:
        g = X[(X.mkey == k) & (X.dataset == f"synth8x4{tag}")]; g0 = X[(X.mkey == k) & (X.dataset == f"synth1x4{tag}")]
        sg, sp, _, _, _ = gain(g, "scan"); og = gain(g, "oracle subgroup")[0]; ng, _, nh, _, _ = gain(g0, "scan")
        sc = g[g.method == "scan"]; rep = int((sc.terms > 0).sum()); jac = sc.best_term_jaccard.mean()
        bias = {m: g[g.method == m].planted_abs_bias.mean() for m in ["frozen", "Platt", "beta", "isotonic", "scan", "Platt->scan", "oracle subgroup"]}
        disc_rows.append(dict(model=n, key=k, size=size, rep=rep, gain=sg, p=sp, share=100 * sg / og, jac=jac, null=ng, null_harm=nh, **{f"bias_{m}": v for m, v in bias.items()}))
        L.append((f"\\multirow{{2}}{{*}}{{{n}}}" if size == "2.4k" else "") + f" & {size} & {rep}/10 & {fmt(sg, sp)} & "
                 f"{max(100 * sg / og, 0) if abs(100 * sg / og) >= 0.5 else 0:.0f}\\% & {jac:.2f} & {ng:+.1f} ({nh}) \\\\")
        T = "One" if tag == "d1" else "Four"
        mac(f"D{MAC[k]}{T}Rep", f"{rep}/10"); mac(f"D{MAC[k]}{T}Gain", f"{sg:.1f}"); mac(f"D{MAC[k]}{T}Share", f"{100 * sg / og:.0f}")
        mac(f"D{MAC[k]}{T}Jac", f"{jac:.2f}"); mac(f"D{MAC[k]}{T}BiasF", f"{bias['frozen']:.3f}"); mac(f"D{MAC[k]}{T}BiasS", f"{bias['scan']:.3f}")
        mac(f"D{MAC[k]}{T}BiasP", f"{bias['Platt']:.3f}"); mac(f"D{MAC[k]}{T}Oracle", f"{og:.1f}"); mac(f"D{MAC[k]}{T}NullHarm", str(nh))
    if j < len(MODELS) - 1: L.append("\\addlinespace[2pt]")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAPER, "tab_disc.tex"), "w").write("\n".join(L) + "\n")
D = pd.DataFrame(disc_rows); D.to_csv(RESULTS / "disc_summary.csv", index=False)
d4 = D[D["size"] == "9.6k"]; d1 = D[D["size"] == "2.4k"]
mac("DShareLo", f"{d4.share.min():.0f}"); mac("DShareHi", f"{d4.share.max():.0f}")
red = 100 * (1 - d4.bias_scan / d4.bias_frozen); mac("DBiasRedLo", f"{red.min():.0f}"); mac("DBiasRedHi", f"{red.max():.0f}")
mac("DJacLo", f"{d4.jac.min():.2f}"); mac("DJacHi", f"{d4.jac.max():.2f}")
mac("DRepLo", f"{d4.rep.min()}"); mac("DRepOneLo", f"{d1.rep.min()}"); mac("DRepOneHi", f"{d1.rep.max()}")
mac("DBiasFLo", f"{d4.bias_frozen.min():.2f}"); mac("DBiasFHi", f"{d4.bias_frozen.max():.2f}")
mac("DBiasSLo", f"{d4.bias_scan.min():.3f}"); mac("DBiasSHi", f"{d4.bias_scan.max():.3f}")
cal = d4[["bias_Platt", "bias_beta", "bias_isotonic"]].values; mac("DBiasCalLo", f"{cal.min():.2f}"); mac("DBiasCalHi", f"{cal.max():.2f}")
mac("DNullAbsMax", f"{d4.null.abs().max():.1f}"); mac("DNullHarmMax", str(d4.null_harm.max()))
# main benchmark planted residual
pl = B[B.dataset == "synth8"]
bf = pl[pl.method == "frozen"].groupby("mkey").planted_abs_bias.mean(); og = {k: gain(pl[pl.mkey == k], "oracle subgroup")[0] for k, _ in MODELS}
mac("BBiasLo", f"{bf.min():.2f}"); mac("BBiasHi", f"{bf.max():.2f}")
# how much of the planted-subgroup bias global recalibration removes (Sec. IV.A: "reduce subgroup bias by ...%")
_cal = pl.groupby(["mkey", "method"]).planted_abs_bias.mean().unstack()
_red = 100 * (1 - _cal[["Platt", "beta", "isotonic"]].div(_cal["frozen"], axis=0))
mac("BCalRedLo", f"{_red.min().min():.0f}"); mac("BCalRedHi", f"{_red.max().max():.0f}"); mac("BOracleLo", f"{min(og.values()):.1f}"); mac("BOracleHi", f"{max(og.values()):.1f}")

# ---------- robustness: all corrections, four models ----------
ALL = [("temperature", "Temp."), ("Platt", "Platt"), ("beta", "Beta"), ("isotonic", "Iso."), ("histogram binning", "Hist."),
       ("residual tree", "Tree"), ("boosted residual (XGB)", "Boost."), ("scan", "Scan"), ("Platt->scan", "P$\\to$S")]
L = ["\\begin{table*}[t]", "\\centering",
     "\\caption{Frozen-backbone benchmark under the standard correction budget. LL is frozen test log-loss; other "
     "entries are mean improvement ($\\times10^{-3}$, positive is better) over 10 matched seeds. * paired $p<0.05$ "
     "(descriptive, unadjusted); bold: largest mean in the row. Corrections are defined in Sec.~III; P$\\to$S: Platt then Scan.}",
     "\\label{tab:bench}", "\\footnotesize\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabular}{llr" + "r" * len(ALL) + "}", "\\toprule", "Data & Base model & LL & " + " & ".join(n for _, n in ALL) + " \\\\", "\\midrule"]
for i, (ds, dn) in enumerate(DSN):
    for j, (k, n) in enumerate(MODELS):
        g = B[(B.dataset == ds) & (B.mkey == k)]; gs = {m: gain(g, m) for m, _ in ALL}
        bestv = max(round(v[0], 1) for v in gs.values())
        cells = [("\\textbf{%s}" % fmt(*gs[m][:2])) if (round(gs[m][0], 1) == bestv and bestv > 0) else fmt(*gs[m][:2]) for m, _ in ALL]
        ll = g[g.method == "frozen"].test_ll.mean()
        L.append((f"\\multirow{{4}}{{*}}{{{dn}}}" if j == 0 else "") + f" & {n} & {ll:.4f} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule" if i < len(DSN) - 1 else "\\bottomrule")
L += ["\\end{tabular}", "\\end{table*}"]
open(os.path.join(PAPER, "tab_bench.tex"), "w").write("\n".join(L) + "\n")
hb = B[B.method == "histogram binning"]; mac("HistWorse", f"{int(sum(gain(B[(B.dataset == ds) & (B.mkey == k)], 'histogram binning')[0] < 0 for ds, _ in DSN for k, _ in MODELS))}")
mac("XgbBoostLeaves", f"{B[(B.method == 'boosted residual (XGB)')].groupby(['dataset', 'mkey']).n_params.median().median():.0f}")

# ---------- figure: subgroup bias, four models, 2.4k vs 9.6k discovery ----------
plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
fig, ax = plt.subplots(figsize=(3.45, 2.05))
items = [("bias_frozen", "Frozen base model", MUTED, "o", None), ("bias_Platt", "Platt", "#eb6834", "s", None),
         ("bias_scan", "Scan, 2.4k discovery rows", "#b9d4f3", "D", "2.4k"), ("bias_scan", "Scan, 9.6k discovery rows", "#1f5fae", "D", "9.6k")]
ys = np.arange(len(MODELS))[::-1]
for (col, lab, colr, mk, size) in items:
    sub = D[D["size"] == (size or "9.6k")].set_index("key").loc[[k for k, _ in MODELS]]
    big = size == "9.6k"; ax.scatter(sub[col].values, ys, color=colr, marker=mk, s=44 if big else (16 if size else 26), zorder=4 if big else 3, label=lab, edgecolor=INK if big else "white", linewidth=0.7 if big else 0.6)
ora = D[D["size"] == "9.6k"].set_index("key").loc[[k for k, _ in MODELS]]["bias_oracle subgroup"].values
ax.scatter(ora, ys, color=INK, marker="|", s=60, zorder=3, label="True subgroup, best shift (oracle)")
for y, k in zip(ys, [k for k, _ in MODELS]):
    r = D[(D.key == k) & (D["size"] == "9.6k")].iloc[0]; ax.plot([r.bias_scan, r.bias_frozen], [y, y], color=GRID, lw=2, zorder=1)
ax.set_yticks(ys); ax.set_yticklabels([n for _, n in MODELS]); ax.set_xlim(0, 0.17)
ax.set_xlabel("absolute bias on the planted subgroup (test)", color=INK)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, 1.33), ncol=2, handletextpad=0.3, columnspacing=0.8)
fig.savefig(os.path.join(PAPER, "fig_disc.pdf"), bbox_inches="tight", pad_inches=0.02); fig.savefig(os.path.join(PAPER, "fig_disc.png"), dpi=220, bbox_inches="tight")
with open(os.path.join(PAPER, "numbers_bench.tex"), "w") as fh:
    for k, v in sorted(M.items()): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros")
print(D[["model", "size", "rep", "gain", "share", "jac", "null", "null_harm", "bias_frozen", "bias_Platt", "bias_scan"]].round(3).to_string(index=False))
# typographic minus signs in table cells
import re as _re
for _p in [LATEX / "tab_bench.tex", RESULTS / "tab_bench_main_corrections.tex", LATEX / "tab_disc.tex"]:
    if os.path.exists(_p):
        _s = open(_p).read(); _s = _re.sub(r'(?<=[\s(])-(?=\d)', r'$-$', _s); open(_p, "w").write(_s)
