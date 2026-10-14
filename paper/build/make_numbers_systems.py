"""Numbers for the systems section and the reproducibility note -> paper/latex/numbers.tex.
Reads <artistry>/systems/: cost_breakdown.csv (profile of the original implementation), cost4_scaling.csv (three
implementations x six problem sizes), fidelity.json (logit-histogram check), credit_baseline.csv (tree baseline of the
original study), scorer_calls.json and discovery_oracle_gain.csv (in-text numbers of Secs. VI and IV.B)."""
import json, numpy as np, pandas as pd
from paths import LATEX, SYSTEMS
M = {}
f = lambda x, d=1: f"{x:.{d}f}"
# share of one fit spent in the exact scorer (original implementation, profiled)
br = pd.read_csv(SYSTEMS / "cost_breakdown.csv", index_col=0).loc["original"]
M["CostScorerShare"] = f(100 * br.exact_scorer / br.total, 0)
# speed-up of the batched + parallel implementation; every implementation must select identical repairs
c = pd.read_csv(SYSTEMS / "cost4_scaling.csv")
piv = c.pivot_table(index=["n_discovery", "p"], columns="impl", values="seconds")
fp = c.pivot_table(index=["n_discovery", "p"], columns="impl", values="fp", aggfunc="first")
assert (fp.nunique(axis=1) == 1).all(), "repairs differ between implementations"
d0 = piv.loc[(2400, 30)]
M["SpOrigDefault"] = f(d0.original); M["SpBatDefault"] = f(d0.batched); M["SpBatX"] = f(d0.original / d0.batched)
M["SpBatMin"] = f((piv.original / piv.batched).min()); M["SpBatMax"] = f((piv.original / piv.batched).max())
M["SpConfigs"] = str(len(piv))
# logit-histogram fidelity
fj = json.load(open(SYSTEMS / "fidelity.json"))
M["FidHistN"] = f"{fj['hist_n']:,}".replace(",", "{,}"); M["FidHistErr"] = f(fj["hist_max_err"], 3)
# original-study tree baseline on Credit Default
M["CreditXGBbase"] = f(pd.read_csv(SYSTEMS / "credit_baseline.csv").base_ll.mean(), 4)
# in-text numbers (typed in the paper; macros provided for checking)
sc = json.load(open(SYSTEMS / "scorer_calls.json")); M["ScorerCalls"] = f"{sc['scorer_calls']:,}".replace(",", "{,}")
og = pd.read_csv(SYSTEMS / "discovery_oracle_gain.csv").groupby("model").oracle_gain_nats.mean()
M["DiscOracleNatsLo"] = f(og.min(), 0); M["DiscOracleNatsHi"] = f(og.max(), 0)
with open(LATEX / "numbers.tex", "w") as fh:
    for k, v in sorted(M.items()): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(M)
