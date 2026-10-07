"""Every number and table of the paper's ladder sections, generated from the ladder outputs.
  python make_paper_ladder_inputs.py <posthoc folder> <paper folder> <timing_cpu.csv> <timing_gpu.csv>
Writes numbers_ladder.tex (macros), tab_ladder_gain.tex, tab_ladder_diag.tex, tab_ladder_cost.tex."""
import os, sys, numpy as np, pandas as pd
from scipy import stats
from paths import LADDER, LATEX, RESULTS, TIMING_CPU, TIMING_GPU
IN, PAPER, TCPU, TGPU = str(LADDER), str(LATEX), str(TIMING_CPU), str(TIMING_GPU)
P = pd.read_csv(os.path.join(IN, "posthoc.csv")); REP = os.path.join(IN, "report")
T8 = pd.read_csv(os.path.join(REP, "T8_attribution_long.csv")); T9 = pd.read_csv(os.path.join(REP, "T9_reprompt_long.csv"))
TT = {"cpu": pd.read_csv(TCPU), "gpu": pd.read_csv(TGPU)}
M = {}
def mac(name, val): M[name] = val
def f1(x): return f"{x:.1f}"
def comma(x): return f"{x:,.0f}".replace(",", "{,}")
def per1000(dev, k):
    t = TT[dev].sort_values("context_rows")
    return float(np.exp(np.interp(np.log(k), np.log(t.context_rows), np.log(t.total_seconds_per_1000))))
W = P.pivot_table(index=["dataset", "context_rows", "seed"], columns="method", values="test_ll")
def gain(ds, k, m, base="frozen"):
    g = W.loc[(ds, k)]; d = 1e3 * (g[base] - g[m]).values; n = len(d)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n); p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else 1.0
    return d.mean(), d.mean() - h, d.mean() + h, p, int((d < -1e-9).sum()), n
FULL = {ds: int(P[P.dataset == ds].context_rows.max()) for ds in P.dataset.unique()}
NAME = {"synth8": "Pl", "credit": "Cr", "synth1": "Nu"}
CTXN = {500: "Five", 1000: "One", 2000: "Two", 4000: "Four"}

# ---------------- macros ----------------
mac("LadSeeds", str(P.seed.nunique() // 1 if False else P[P.dataset == "credit"].seed.nunique()))
mac("LadFiles", str(len(P[P.method == "frozen"])))
for ds, nm in NAME.items():
    mac(f"{nm}Full", comma(FULL[ds]))
    for k, kn in list(CTXN.items()) + [(FULL[ds], "Full")]:
        for m, mn in [("scan", "Scan"), ("Platt", "Platt"), ("Platt->scan", "PS"), ("boosted residual (XGB)", "Xgb"),
                      ("temperature", "Temp"), ("isotonic", "Iso"), ("beta", "Beta")]:
            g = gain(ds, k, m)
            mac(f"{nm}{kn}{mn}", f1(g[0])); mac(f"{nm}{kn}{mn}Lo", f1(g[1])); mac(f"{nm}{kn}{mn}Hi", f1(g[2]))
            mac(f"{nm}{kn}{mn}Harm", str(g[4]))
        if ds == "synth8":
            g = gain(ds, k, "oracle subgroup"); mac(f"{nm}{kn}Oracle", f1(g[0]))
        fz = P[(P.dataset == ds) & (P.context_rows == k)]
        mac(f"{nm}{kn}FrozenLL", f"{fz[fz.method == 'frozen'].test_ll.mean():.3f}")
        sc = fz[fz.method == "scan"]
        mac(f"{nm}{kn}Interv", f"{int((sc.terms > 0).sum())}/{len(sc)}")
        if "planted_abs_bias" in fz and ds == "synth8":
            mac(f"{nm}{kn}ResFrozen", f"{fz[fz.method == 'frozen'].planted_abs_bias.mean():.3f}")
            mac(f"{nm}{kn}ResScan", f"{sc.planted_abs_bias.mean():.3f}")
            mac(f"{nm}{kn}ResPlatt", f"{fz[fz.method == 'Platt'].planted_abs_bias.mean():.3f}")
            mac(f"{nm}{kn}Jac", f"{sc.best_term_jaccard.mean():.2f}")
        if ds == "credit":
            mac(f"{nm}{kn}Audit", f"{int(sc.audited_pass.sum())}/{int(sc.terms.sum())}")
# scan share of oracle at 500-1000
o = [gain("synth8", k, "scan")[0] / gain("synth8", k, "oracle subgroup")[0] for k in (500, 1000)]
mac("PlOracleShareLo", f"{100 * min(o):.0f}"); mac("PlOracleShareHi", f"{100 * max(o):.0f}")
# null harms at >= 2000
nu = [gain("synth1", k, "scan")[4] for k in (2000, 3000, 4000, FULL["synth1"])]
mac("NuHarmLo", str(min(nu))); mac("NuHarmHi", str(max(nu)))
ps = [gain("synth1", k, "Platt->scan")[0] for k in sorted(P[P.dataset == "synth1"].context_rows.unique())]
mac("NuPSMin", f1(min(ps))); mac("NuPSMax", f1(max(ps)))
# equivalent context and cost
for ds, nm in [("credit", "Cr"), ("synth8", "Pl")]:
    for m, mn in [("scan", "Scan"), ("Platt", "Platt"), ("Platt->scan", "PS"), ("boosted residual (XGB)", "Xgb")]:
        r = T9[(T9.dataset == ds) & (T9.method == m) & (T9.context_rows == 1000)].iloc[0]; eq = r.equivalent_context_rows
        mac(f"{nm}Eq{mn}", comma(eq)); mac(f"{nm}Eq{mn}X", f1(eq / 1000))
        mac(f"{nm}Eq{mn}Cpu", f1(per1000("cpu", eq) / per1000("cpu", 1000))); mac(f"{nm}Eq{mn}Gpu", f1(per1000("gpu", eq) / per1000("gpu", 1000)))
        mac(f"{nm}Eq{mn}CpuSec", f"{per1000('cpu', eq):.0f}"); mac(f"{nm}Eq{mn}GpuSec", f"{per1000('gpu', eq):.1f}")
# label-matched context: put the 1k context plus all held-out discovery+selection labels into the context instead
HELD = {"credit": 6000 + 4500, "synth8": 2400 + 1800, "synth1": 2400 + 1800}
for ds, nm in [("credit", "Cr"), ("synth8", "Pl")]:
    fz = P[(P.dataset == ds) & (P.method == "frozen")].groupby("context_rows").test_ll.mean().sort_index()
    km = 1000 + HELD[ds]; llm = float(np.interp(np.log(km), np.log(fz.index.values), fz.values))
    sc = P[(P.dataset == ds) & (P.method == "scan") & (P.context_rows == 1000)].test_ll.mean()
    mac(f"{nm}LabCtx", comma(km)); mac(f"{nm}LabLL", f"{llm:.4f}"); mac(f"{nm}ScanOneLL", f"{sc:.4f}")
    mac(f"{nm}LabGap", f1(1e3 * (sc - llm)))
    mac(f"{nm}LabCpu", f"{per1000('cpu', km) / per1000('cpu', 1000):.0f}"); mac(f"{nm}LabGpu", f"{per1000('gpu', km) / per1000('gpu', 1000):.0f}")
    print(nm, km, round(llm, 4), round(sc, 4), M[f"{nm}LabCpu"], M[f"{nm}LabGpu"])
for dev, dn in [("cpu", "Cpu"), ("gpu", "Gpu")]:
    for k, kn in [(500, "Five"), (1000, "One"), (2000, "Two"), (4000, "Four"), (15000, "Full")]:
        mac(f"{dn}Sec{kn}", comma(per1000(dev, k)) if dev == "cpu" else f"{per1000(dev, k):.1f}")
mac("CpuFullOverOne", f"{per1000('cpu', 15000) / per1000('cpu', 1000):.0f}"); mac("GpuFullOverOne", f"{per1000('gpu', 15000) / per1000('gpu', 1000):.0f}")
sc = P[P.method == "scan"]; mac("ScanSecMed", f1(sc.method_seconds.median())); mac("ScanSecMax", f"{sc.method_seconds.max():.0f}")
mac("ScanSecOneMed", f1(sc[sc.context_rows == 1000].method_seconds.median()))
mac("XgbLeavesCr", f"{P[(P.dataset == 'credit') & (P.method == 'boosted residual (XGB)')].n_params.median():.0f}")
mac("ScanRulesCrOne", f1(sc[(sc.dataset == "credit") & (sc.context_rows == 1000)].terms.mean()))
# attribution
def att(ds, m, col, ks):
    t = T8[(T8.dataset == ds) & (T8.method == m) & (T8.context.astype(int).isin(ks))].groupby("context")[col].mean(); return t.min(), t.max()
lo, hi = att("synth8", "scan", "concentration", [500, 1000, 1500, 2000, 3000, 4000]); mac("PlConcLo", f1(lo)); mac("PlConcHi", f1(hi))
lo, hi = att("credit", "scan", "concentration", [500, 1000, 1500, 2000, 3000, 4000]); mac("CrConcLo", f1(lo)); mac("CrConcHi", f1(hi))
for ds, nm in [("synth8", "Pl"), ("credit", "Cr")]:
    lo, hi = att(ds, "scan", "sign_agreement", [500, 1000]); mac(f"{nm}SignLo", f"{100*lo:.0f}"); mac(f"{nm}SignHi", f"{100*hi:.0f}")
lo, hi = att("synth8", "Platt", "sign_agreement", [500, 1000, 1500, 2000, 3000, 4000]); lo2, hi2 = att("credit", "Platt", "sign_agreement", [500, 1000, 1500, 2000, 3000, 4000])
mac("PlattSignLo", f"{100*min(lo, lo2):.0f}"); mac("PlattSignHi", f"{100*max(hi, hi2):.0f}")
with open(os.path.join(PAPER, "numbers_ladder.tex"), "w") as fh:
    for k, v in sorted(M.items()): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------- Table: gains ----------------
cols = [("temperature", "Temp."), ("Platt", "Platt"), ("beta", "Beta"), ("isotonic", "Iso."), ("histogram binning", "Hist."),
        ("residual tree", "Tree"), ("boosted residual (XGB)", "Boost."), ("scan", "Scan"), ("Platt->scan", "P$\\to$S")]
L = ["\\begin{table*}[t]", "\\centering", "\\caption{Test log-loss improvement over frozen TabPFN-2.5 at the same context "
     "($\\times10^{-3}$, mean of 10 seeds; * paired $t$-test $p<0.05$; bold: best). Every correction uses the same "
     "discovery+selection labels; abbreviations as in Table~\\ref{tab:bench}.}", "\\label{tab:gain}", "\\footnotesize\\setlength{\\tabcolsep}{4.5pt}",
     "\\begin{tabular}{llr" + "r" * len(cols) + "}", "\\toprule",
     "Data & Context & Frozen LL & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
DSN = {"synth8": "Planted subgroup", "credit": "Credit Default", "synth1": "No planted effect"}
for ds in ["synth8", "synth1", "credit"]:
    ks = [500, 1000, 2000, 4000, FULL[ds]]
    for i, k in enumerate(ks):
        gs = {m: gain(ds, k, m) for m, _ in cols}
        best = max((gs[m][0], m) for m, _ in cols)[1]
        cells = []
        for m, _ in cols:
            v, _, _, p, _, _ = gs[m]; s = f"{v:+.1f}" + ("*" if p < 0.05 else "")
            cells.append(f"\\textbf{{{s}}}" if m == best else s)
        lab = (f"{k/1000:g}k" if k >= 1000 else str(k)) + (" (full)" if k == FULL[ds] else "")
        fz = P[(P.dataset == ds) & (P.context_rows == k) & (P.method == "frozen")].test_ll.mean()
        L.append((f"\\multirow{{{len(ks)}}}{{*}}{{{DSN[ds]}}}" if i == 0 else "") + f" & {lab} & {fz:.4f} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule" if ds != "credit" else "\\bottomrule")
L += ["\\end{tabular}", "\\end{table*}"]
open(RESULTS / "tab_ladder_gain.tex", "w").write("\n".join(L) + "\n")
# the complete correction-by-context table (every rung of the ladder), released with the code
rows_ = []
for ds in ["synth8", "synth1", "credit"]:
    for k in sorted(P[P.dataset == ds].context_rows.unique()):
        fz = P[(P.dataset == ds) & (P.context_rows == k) & (P.method == "frozen")].test_ll.mean()
        for m in [c[0] for c in cols] + ["scan a=0.01"] + (["oracle subgroup"] if ds != "credit" else []):
            v, lo, hi, p, worse, n = gain(ds, k, m)
            rows_.append(dict(dataset={"synth8": "planted subgroup", "synth1": "no planted effect", "credit": "Credit Default"}[ds],
                              context_rows=int(k), method=m, frozen_test_ll=round(fz, 6), gain_x1e3=round(v, 3),
                              ci95_lo=round(lo, 3), ci95_hi=round(hi, 3), p_value=round(p, 4), seeds_worse=worse, seeds=n))
pd.DataFrame(rows_).to_csv(RESULTS / "context_ladder_all_corrections.csv", index=False)

# ---------------- Table: diagnostics (planted + credit) ----------------
L = ["\\begin{table}[t]", "\\centering", "\\caption{What remains repairable as TabPFN context grows. Planted: subgroup residual before$\\to$after Scan, "
     "oracle one-rule gain ($\\times10^{-3}$), Scan's share, and Jaccard overlap with the true subgroup. Credit: seeds with "
     "an accepted repair and accepted rules passing the audit.}", "\\label{tab:diag}", "\\footnotesize\\setlength{\\tabcolsep}{3.2pt}",
     "\\begin{tabular}{lcrrrcc}", "\\toprule",
     " & \\multicolumn{4}{c}{Planted subgroup} & \\multicolumn{2}{c}{Credit Default} \\\\",
     "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}", "Context & Bias F$\\to$S & Oracle & Share & Jacc. & With rule & Audited \\\\", "\\midrule"]
for k in [500, 1000, 2000, 4000, "full"]:
    kp = FULL["synth8"] if k == "full" else k; kc = FULL["credit"] if k == "full" else k
    sp = P[(P.dataset == "synth8") & (P.context_rows == kp)]; sc_ = sp[sp.method == "scan"]
    og = gain("synth8", kp, "oracle subgroup")[0]; sg = gain("synth8", kp, "scan")[0]
    cc = P[(P.dataset == "credit") & (P.context_rows == kc) & (P.method == "scan")]
    lab = "full" if k == "full" else (f"{k/1000:g}k" if k >= 1000 else str(k))
    L.append(f"{lab} & {sp[sp.method=='frozen'].planted_abs_bias.mean():.3f}$\\to${sc_.planted_abs_bias.mean():.3f} & {og:.1f} & "
             f"{100*sg/og:.0f}\\% & {sc_.best_term_jaccard.mean():.2f} & {int((cc.terms>0).sum())}/{len(cc)} & "
             f"{int(cc.audited_pass.sum())}/{int(cc.terms.sum())} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAPER, "tab_ladder_diag.tex"), "w").write("\n".join(L) + "\n")

# ---------------- Table: cost ----------------
L = ["\\begin{table}[t]", "\\centering", "\\caption{Recurring context cost and equivalent-context accuracy. Top: seconds to score 1{,}000 rows at "
     "representative TabPFN-2.5 contexts. Bottom: for a correction fitted at 1{,}000 context rows, the frozen context needed "
     "to match its test loss and the corresponding recurring cost multiplier. Scan fitting is one-time "
     f"(median {M['ScanSecOneMed']}\\,s on one CPU core at a 1{{,}}000-row context).}}", "\\label{tab:cost}",
     "\\footnotesize\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{lrrrrr}", "\\toprule",
     "Context & 500 & 1k & 2k & 4k & 15k \\\\", "\\midrule",
     "CPU (s) & " + " & ".join(f"{per1000('cpu', k):,.0f}".replace(",", "{,}") for k in (500, 1000, 2000, 4000, 15000)) + " \\\\",
     "T4 GPU (s) & " + " & ".join(f"{per1000('gpu', k):.1f}" for k in (500, 1000, 2000, 4000, 15000)) + " \\\\",
     "\\bottomrule"]
L += ["\\end{tabular}", "\\\\[4pt]", "\\begin{tabular}{llrrr}", "\\toprule",
      "Data & Fix at 1k & Equiv.\\ context & CPU & T4 \\\\", "\\midrule"]
for ds, dn in [("synth8", "Planted"), ("credit", "Credit")]:
    for i, (m, mn) in enumerate([("Platt", "Platt"), ("scan", "Scan"), ("Platt->scan", "Platt$\\to$scan"), ("boosted residual (XGB)", "Boosted corrector")]):
        r = T9[(T9.dataset == ds) & (T9.method == m) & (T9.context_rows == 1000)].iloc[0]; eq = r.equivalent_context_rows
        L.append((dn if i == 0 else "") + f" & {mn} & {eq:,.0f} & {per1000('cpu', eq)/per1000('cpu', 1000):.1f}$\\times$ & "
                 f"{per1000('gpu', eq)/per1000('gpu', 1000):.1f}$\\times$ \\\\")
    if ds == "synth8": L.append("\\midrule")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(PAPER, "tab_ladder_cost.tex"), "w").write("\n".join(L) + "\n")
print(len(M), "macros; tables written to", PAPER)
# typographic minus signs in table cells
import re as _re
for _p in [RESULTS / "tab_ladder_gain.tex"]:
    if os.path.exists(_p):
        _s = open(_p).read(); _s = _re.sub(r'(?<=[\s(])-(?=\d)', r'$-$', _s); open(_p, "w").write(_s)
