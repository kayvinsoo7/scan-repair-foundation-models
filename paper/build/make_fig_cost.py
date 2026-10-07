"""Fig. (repair cost): scaling of one repair fit in three implementations + profiled component times.
Same data as make_paper_inputs.py (cost4_*.csv); labels made consistent and tiny bars annotated."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import LATEX, RESULTS, SYSTEMS
OUT = str(LATEX)
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
c = pd.read_csv(SYSTEMS / "cost4_scaling.csv"); br = pd.read_csv(SYSTEMS / "cost4_breakdown.csv", index_col=0)
s30 = c.pivot_table(index=["n_discovery", "p"], columns="impl", values="seconds").xs(30, level="p")
IMPL = [("original", GRAY, "original", "o"), ("compiled", ORANGE, "compiled scorer", "^"), ("batched", BLUE, "batched + parallel", "s")]
fig, a0 = plt.subplots(figsize=(3.3, 1.75)); fig2, a1 = plt.subplots(figsize=(3.45, 1.55)); ax = [a0, a1]
for col, colr, lab, mk in IMPL:
    ax[0].plot(s30.index / 1000, s30[col], color=colr, lw=1.5, marker=mk, ms=3.3, mec="white", mew=0.6, clip_on=False, label=lab)
ax[0].legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.3, borderaxespad=0, labelspacing=0.2)
ax[0].set_xlabel("discovery rows (thousands)"); ax[0].set_ylabel("fit time (s)"); ax[0].set_ylim(0, s30.original.max() * 1.12)
ax[0].set_xticks([1.2, 2.4, 4.8, 9.6]); ax[0].set_xticklabels(["1.2", "2.4", "4.8", "9.6"]); ax[0].set_xlim(0, 10.6)
for col, colr, lab, mk in IMPL:
    v = s30[col].iloc[-1]; ax[0].text(9.6 + 0.25, v, f"{v:.1f} s", va="center", fontsize=5.6, color=colr, clip_on=False); ax[0].grid(axis="y", color="#e6e5e0", lw=0.5)
comps = [("legacy_fixed_bin", "fixed-bin guard"), ("adaptive_search", "adaptive search"), ("selection_and_refit", "select + refit")]
yb = np.arange(len(comps))[::-1]; hgt = 0.26
for k, (impl, colr, lab, _) in enumerate(IMPL):
    vals = [br.loc[impl, c_] for c_, _ in comps]
    ax[1].barh(yb + (1 - k) * hgt, vals, height=hgt * 0.92, color=colr, label=lab)
    for y_, v in zip(yb + (1 - k) * hgt, vals):          # exact value at the end of every bar
        if v < 0.1:   # invisible bar: show a colour key next to its exact value
            ax[1].plot([0.45], [y_], marker="s", ms=3.2, color=colr, clip_on=False)
            ax[1].text(0.9, y_, f"{v:.3f} s", va="center", fontsize=5.2, color=INK)
        else:
            ax[1].text(v + 0.3, y_, f"{v:.1f}", va="center", fontsize=5.2, color=INK)
ax[1].set_xlim(0, br.loc["original", "legacy_fixed_bin"] * 1.22)
ax[1].legend(frameon=False, fontsize=5.8, loc="lower right", handlelength=1.0, borderaxespad=0, labelspacing=0.2)
ax[1].set_yticks(yb); ax[1].set_yticklabels([n for _, n in comps], fontsize=6.3)
ax[1].set_xlabel("time (s), under profiler"); ax[1].tick_params(axis="y", length=0)
ax[1].grid(axis="x", color="#e6e5e0", lw=0.5); ax[1].set_axisbelow(True)
for f_, nm, OUT in [(fig, "fig_cost", str(LATEX)), (fig2, "fig_cost_breakdown", str(RESULTS))]:
    f_.savefig(f"{OUT}/{nm}.pdf", bbox_inches="tight", pad_inches=0.02); f_.savefig(f"{OUT}/{nm}.png", dpi=220, bbox_inches="tight")
print("ok", br.round(3).to_dict())
