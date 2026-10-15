"""Summarize the calibrated-acceptance experiment (reads summary.csv written by calibrated_acceptance.py).

Per dataset and context, for the nominal (alpha = 0.05) and the calibrated test:
  null rejection rate of the nominal test (mean over seeds of the share of null replicates with 2*gain > 3.84),
  calibrated alpha (median, range), seeds with at least one rule, mean test log-loss gain (x1e-3) with a paired
  t-test against the frozen model, seeds made worse, mean number of rules, mean best Jaccard with the planted subgroup.
Writes report.csv next to summary.csv and prints it."""
import os, numpy as np, pandas as pd
from scipy.stats import ttest_1samp
from common import ARTISTRY, out_dir

OUT = out_dir(os.environ.get("OUT", ARTISTRY / "systems" / "calibrated_acceptance"))
S = pd.read_csv(OUT / "summary.csv"); S["context"] = S["context"].astype(str)
rows = []
for (ds, ctx, test), g in S.groupby(["dataset", "context", "test"], sort=False):
    g = g.sort_values("seed")
    rows.append(dict(dataset=ds, context=ctx, test=test, seeds=len(g),
                     null_reject_rate_nominal=g["null_reject_rate_nominal"].mean(),
                     null_reject_rate_min=g["null_reject_rate_nominal"].min(), null_reject_rate_max=g["null_reject_rate_nominal"].max(),
                     alpha_median=g["alpha"].median(), alpha_min=g["alpha"].min(), alpha_max=g["alpha"].max(),
                     seeds_with_rule=int((g["terms"] > 0).sum()), mean_rules=g["terms"].mean(),
                     mean_gain_x1e3=g["gain_x1e3"].mean(),
                     gain_p=ttest_1samp(g["gain_x1e3"], 0).pvalue if g["gain_x1e3"].std() > 0 else np.nan,
                     seeds_worse=int((g["gain_x1e3"] < 0).sum()),
                     mean_planted_bias=g["planted_abs_bias"].mean(), frozen_planted_bias=g["frozen_planted_abs_bias"].mean(),
                     mean_jaccard=g["best_term_jaccard"].mean(), null_seconds_per_replicate=g["null_seconds_per_replicate"].mean()))
R = pd.DataFrame(rows); R.to_csv(OUT / "report.csv", index=False)
pd.set_option("display.width", 250); print(R.round(4).to_string(index=False))
