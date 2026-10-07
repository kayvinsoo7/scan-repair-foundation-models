"""Structured Scan Repair on your own frozen classifier, in about one minute on a CPU.

The repair only needs, for labelled rows the model was NOT trained on:
  * the original features (numeric and categorical, as a pandas DataFrame),
  * the labels (0/1),
  * the model's logits F(x) = log(p / (1 - p)).
Those rows are split into DISCOVERY (the scan proposes subgroup rules) and SELECTION (each rule must survive two
held-out halves and a likelihood-ratio test). The model itself is never changed: the repair adds a few named
subgroup shifts to its logit, or nothing at all.

Two examples (replace `model` and `logits()` with yours: TabPFN, TabICL, CatBoost, a neural network, ...):
  python examples/repair_your_model.py synthetic   planted 5-way subgroup error, 9,600 discovery rows: the scan finds it
  python examples/repair_your_model.py credit      Credit Default: little localized error is left, and the scan abstains"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import numpy as np, xgboost as xgb
from common import DATA, core, ext, ll
import fastexact; fastexact.install()                     # optional: compiled exact scorer (identical results)
import fastscan; fastscan.install()                       # optional: batched, parallel candidate scoring (identical results)

import pandas as pd
WHICH = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
if WHICH == "credit":
    bundle = ext.load_credit_default(DATA, allow_download=False)
    sp = ext.make_extended_split(bundle, seed=42001)          # 15,000 / 6,000 / 4,500 / 4,500 rows
    X, y = bundle.X, np.asarray(bundle.y, int); num, cat = list(bundle.numeric_columns), list(bundle.categorical_columns)
    tr, di, se, te = sp.train, sp.discovery, sp.selection, sp.test
else:
    from ladder_data import load_dataset                       # synthetic data with a planted 5-way subgroup (odds x 8)
    D = load_dataset("synth8x4d4", 1)                          # 6,000 / 9,600 / 1,800 / 1,800 rows
    X, y = pd.DataFrame(D.X, columns=D.feature_names), D.y; num, cat = list(D.feature_names), []
    tr, di, se, te = D.train, D.discovery, D.selection, D.test

# ---- any frozen probabilistic classifier; here a small XGBoost on integer-coded features ----
codes = X.copy()
for c in cat: codes[c] = codes[c].astype("category").cat.codes
model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)
model.fit(codes.iloc[tr], y[tr])
def logits(idx): return model.predict(codes.iloc[idx], output_margin=True)

# ---- Structured Scan Repair ----
prep = core.CompactScanPreprocessor(num, cat); prep.fit(X.iloc[tr])     # scanner features (original names)
Sd, Ss, St = (prep.transform(X.iloc[i]) for i in (di, se, te))
repair = core.make_realdata_scanner(prep.feature_types, discovery_n=len(Sd), selection_n=len(Ss), seed=0)
repair.fit(Sd, y[di], logits(di), Ss, y[se], logits(se))

zt = logits(te); zr = repair.repair_logits(St, zt)
def describe(b):
    """Readable rule. Adaptive-search rules carry literals; fixed-bin guard rules carry bin restrictions per feature."""
    if hasattr(b, "literals"): return core.describe_basis(b, prep)
    parts = []
    for j, bins in sorted(b.restrictions.items()):
        e = b.binner.edges[j]; lo, hi = min(bins), max(bins)
        name = prep.feature_names[j]
        parts.append(f"{name} > {e[lo]:.3f}" if hi == len(e) - 2 else
                     f"{name} <= {e[hi + 1]:.3f}" if lo == 0 else f"{e[lo]:.3f} < {name} <= {e[hi + 1]:.3f}")
    return " AND ".join(parts)
print(f"accepted rules: {len(repair.bases_)}")
for b, d in zip(repair.bases_, np.asarray(repair.coefficients_, float)):
    print(f"  IF {describe(b)}  THEN logit += {d:+.3f}  (odds x {np.exp(d):.2f})")
yt = y[te]
print(f"test log-loss: frozen {ll(yt, core.sigmoid(zt)):.5f} -> repaired {ll(yt, core.sigmoid(zr)):.5f}")
if WHICH != "credit":
    print("planted subgroup: x0..x4 all > 0.397  (odds x 8)")
