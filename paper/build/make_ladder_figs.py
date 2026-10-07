"""Paper figures from the context ladder (TabPFN-2.5).
  python make_ladder_figs.py <posthoc folder (posthoc.csv, report/)> <out folder> <timing_cpu.csv> <timing_gpu.csv>
fig_gain_ladder   improvement over the frozen model at the same context, per context size (paired 95% CI)
fig_frontier      accuracy vs. prediction cost: the repair buys the accuracy of a larger context
fig_attribution   the scan flags the rows that more context later fixes, and moves them the same way"""
import os, sys, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

from paths import LADDER, LATEX, TIMING_CPU, TIMING_GPU
IN, OUT, TCPU, TGPU = str(LADDER), str(LATEX), str(TIMING_CPU), str(TIMING_GPU)
P = pd.read_csv(os.path.join(IN, "posthoc.csv")); T8 = pd.read_csv(os.path.join(IN, "report", "T8_attribution_long.csv"))
T9 = pd.read_csv(os.path.join(IN, "report", "T9_reprompt_long.csv"))
TT = {"cpu": pd.read_csv(TCPU), "gpu": pd.read_csv(TGPU)}

plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {   # colour follows the method in every figure
    "scan":                   dict(color="#2a78d6", marker="o", lw=2.0, ms=4.5, label="Scan (ours)"),
    "Platt->scan":            dict(color="#1baf7a", marker="D", lw=2.0, ms=4.0, label="Platt → scan (ours)"),
    "Platt":                  dict(color="#eb6834", marker="s", lw=1.6, ms=4.0, label="Platt"),
    "boosted residual (XGB)": dict(color="#9a9893", marker="^", lw=1.2, ms=3.5, ls=(0, (3, 2)), label="Boosted corrector"),
    "oracle subgroup":        dict(color=INK, marker=None, lw=1.0, ls=(0, (1, 1.5)), label="True subgroup, best shift (oracle)"),
    "frozen":                 dict(color=MUTED, marker="o", lw=1.4, ms=3.5, ls=(0, (4, 2)), label="Frozen TabPFN (more context)"),
}
DS = {"synth8": "Synthetic, planted subgroup", "credit": "Credit Default", "synth1": "Synthetic, no planted subgroup effect"}
ROWS = {"synth8": 6000, "synth1": 6000, "credit": 15000}

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
def ctx_axis(ax, ks, full):
    ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator())
    show = [k for k in (500, 1000, 2000, 4000, 8000, full) if k in ks or k == full]
    ax.xaxis.set_major_locator(FixedLocator(show))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v/1000:g}k" if v >= 1000 else f"{v:g}") + ("\nfull" if v == full else "")))
def paired(ds, m):
    d = P[P.dataset == ds]; w = d.pivot_table(index=["context_rows", "seed"], columns="method", values="test_ll")
    out = []
    for k, g in w.groupby(level=0):
        if m not in g: continue
        x = 1e3 * (g["frozen"] - g[m]).values; h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        out.append((k, x.mean(), h))
    return np.array(out)

# ---------------- Figure 1: gain over frozen at the same context ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35))
for ax, ds in zip(axs, ["synth8", "synth1", "credit"]):
    ks = sorted(P[P.dataset == ds].context_rows.unique()); full = max(ks)
    order = ["oracle subgroup", "boosted residual (XGB)", "Platt", "scan", "Platt->scan"]
    off = {m: f for m, f in zip(order, [1.0, 0.96, 0.98, 1.02, 1.04])}
    for m in order:
        if ds != "synth8" and m == "oracle subgroup": continue
        a = paired(ds, m)
        if not len(a): continue
        st = dict(STYLE[m]); x = a[:, 0] * off[m]
        if m == "oracle subgroup":
            ax.plot(x, a[:, 1], color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"]); continue
        ours = "scan" in m
        eb = ax.errorbar(x, a[:, 1], yerr=a[:, 2], color=st["color"], marker=st["marker"], ms=st["ms"], lw=st["lw"], ls=st.get("ls", "-"),
                         elinewidth=0.9 if ours else 0.6, capsize=0, mec="white", mew=0.8, label=st["label"], zorder=3 if ours else 2)
        if not ours: eb[2][0].set_alpha(0.45)
    ax.axhline(0, color=MUTED, lw=0.8)
    ctx_axis(ax, ks, full); clean(ax); ax.set_title(DS[ds], color=INK, pad=4)
    ax.set_xlabel("TabPFN context (labelled rows)", color=INK)
axs[0].set_ylabel("Log-loss improvement over\nfrozen TabPFN (×10$^{-3}$)", color=INK)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.2)
for e in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"fig_gain_ladder.{e}"), dpi=220)
plt.close(fig)

# ---------------- Figure 2: accuracy vs. prediction cost (CPU) ----------------
def per1000(dev, k):
    t = TT[dev].sort_values("context_rows")
    return float(np.exp(np.interp(np.log(k), np.log(t.context_rows), np.log(t.total_seconds_per_1000))))
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.45))
for ax, ds in zip(axs, ["credit", "synth8"]):
    d = P[P.dataset == ds]; full = d.context_rows.max()
    for m in ["frozen", "Platt", "scan", "Platt->scan"]:
        g = d[d.method == m].groupby("context_rows").test_ll.mean()
        x = [per1000("cpu", k) for k in g.index]; st = STYLE[m]
        ax.plot(x, g.values, color=st["color"], marker=st["marker"], ms=st["ms"], lw=st["lw"], ls=st.get("ls", "-"),
                mec="white", mew=0.8, label=st["label"], zorder=3 if "scan" in m else 2)
        if m == "frozen":
            for k, xx, yy in zip(g.index, x, g.values):
                if k in (500, 1000, 2000, 4000, full):
                    ax.annotate((f"{k/1000:g}k" if k >= 1000 else str(k)) + (" (full)" if k == full else ""), (xx, yy),
                                textcoords="offset points", xytext=(4, 3), fontsize=6.3, color=MUTED)
    # the headline: scan at 1,000 rows reaches the accuracy of frozen TabPFN at the equivalent (larger) context
    r = T9[(T9.dataset == ds) & (T9.method == "scan") & (T9.context_rows == 1000)].iloc[0]
    x0, x1 = per1000("cpu", 1000), per1000("cpu", r.equivalent_context_rows)
    ax.plot([x0, x1], [r.fix_ll, r.fix_ll], color=INK, lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
    ax.plot([x1], [r.fix_ll], marker="o", ms=6, mfc="none", mec=INK, mew=1.0, zorder=4)
    ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    clean(ax); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_title(DS[ds], color=INK, pad=4)
    ax.set_xlabel("CPU seconds per 1,000 predictions (2 cores, log scale)", color=INK)
axs[0].set_ylabel("Test log-loss (lower is better)", color=INK)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03), handlelength=2.2)
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
for e in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"fig_frontier.{e}"), dpi=220)
plt.close(fig)

# ---------------- Figure 3: attribution (scan at context k vs. what full context fixes) ----------------
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.0), sharex="col")
for j, ds in enumerate(["synth8", "credit"]):
    t = T8[(T8.dataset == ds)].copy(); t["k"] = t.context.astype(int)
    ks = sorted(t.k.unique())
    for i, (col, ref, ylab) in enumerate([("concentration", 1.0, "Full-context gain in\nflagged rows (× random)"),
                                          ("sign_agreement", 0.5, "Rows moved the same\nway as more context")]):
        ax = axs[i, j]
        for m in (["scan"] if col == "concentration" else ["Platt", "scan"]):
            g = t[t.method == m].groupby("k")[col]; mu = g.mean(); h = g.std() / np.sqrt(g.count()) * stats.t.ppf(0.975, g.count() - 1)
            st = STYLE[m]
            ax.errorbar(mu.index, mu.values, yerr=h.values, color=st["color"], marker=st["marker"], ms=3.5, lw=1.6, elinewidth=0.8,
                        mec="white", mew=0.6, label=st["label"], zorder=3)
        ax.axhline(ref, color=MUTED, lw=0.8, ls=(0, (2, 2)), label="No information (random rows / chance)")
        ctx_axis(ax, ks, -1); clean(ax)
        if j == 0: ax.set_ylabel(ylab, color=INK, fontsize=6.8)
        if i == 0: ax.set_title(DS[ds].replace("Synthetic, planted subgroup", "Planted subgroup"), color=INK, pad=3, fontsize=7.5)
        if i == 1: ax.set_xlabel("Context of the repaired model", color=INK, fontsize=6.8)
h, l = axs[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=6.3, handlelength=1.8, columnspacing=0.8)
fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=0.8)
for e in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"fig_attribution.{e}"), dpi=220)
plt.close(fig)
print("figures ->", OUT)
