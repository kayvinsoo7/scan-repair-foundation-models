"""Compare re-run corrections with the released ones.
  python tests/compare_corrections.py <released posthoc.csv> <re-run posthoc.csv>
Rows are matched on (dataset, seed, context, method); only rows present in the re-run are checked, so a partial
re-run (PATTERN=...) can be compared too. Exit code 1 if any test log-loss differs by more than TOL (default 1e-9)
or any scan accepts a different number of rules."""
import os, sys, numpy as np, pandas as pd
ref, new = (pd.read_csv(f) for f in sys.argv[1:3]); TOL = float(os.environ.get("TOL", "1e-9"))
key = ["dataset", "seed", "context", "method"]
for d in (ref, new): d["context"] = d["context"].astype(str)
m = new.merge(ref, on=key, suffixes=("_new", "_ref"), how="left", indicator=True)
missing = int((m._merge != "both").sum())
if missing: print(f"note: {missing} re-run rows have no released counterpart (methods not run in the release):",
                  ", ".join(sorted(m.loc[m._merge != "both", "method"].unique())))
m = m[m._merge == "both"]
dll = (m.test_ll_new - m.test_ll_ref).abs()
dterms = (m.terms_new.fillna(-1) != m.terms_ref.fillna(-1)) if "terms_new" in m else pd.Series(False, index=m.index)
print(f"{len(m)} rows compared ({m[key[:3]].drop_duplicates().shape[0]} logits files); "
      f"max |delta test LL| = {dll.max():.2e}; rule-count mismatches: {int(dterms.sum())}")
bad = m[(dll > TOL) | dterms]
if len(bad):
    print(bad[key + ["test_ll_ref", "test_ll_new"]].to_string(index=False)); sys.exit(1)
print("OK: identical to the released results")
