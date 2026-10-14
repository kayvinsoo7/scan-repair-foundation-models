"""Same model, different GPU: TabICLv2 run on a T4 (released results) and on an A100.
Paper, Reproducibility: "Repeating TabICLv2 on an A100 changes frozen test log-loss by at most 7e-5 per run; no
accept/abstain decision changes, and only one of 30 fitted repairs differs in rule count"."""
import pandas as pd
from common import ARTISTRY
t4, a100 = (pd.read_csv(ARTISTRY / "corrections" / m / "posthoc.csv") for m in ("tabicl", "tabicl_a100"))
key = ["dataset", "seed", "method"]
m = t4.merge(a100, on=key, suffixes=("_t4", "_a100"))
fz = m[m.method == "frozen"]; sc = m[m.method == "scan"]
print(f"frozen test log-loss: max |T4 - A100| per run = {(fz.test_ll_t4 - fz.test_ll_a100).abs().max():.1e}")
print(f"scan accept/abstain decisions that differ: {int(((sc.terms_t4 > 0) != (sc.terms_a100 > 0)).sum())} of {len(sc)}")
print(f"scan repairs with a different number of rules: {int((sc.terms_t4 != sc.terms_a100).sum())} of {len(sc)}")
