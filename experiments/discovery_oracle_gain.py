"""Is the planted subgroup detectable in principle from each base model's discovery rows?
For every base model, the exact likelihood gain (Eq. 2, nats) of the TRUE planted subgroup with its best shift,
computed on the 2,400 discovery rows of the planted benchmark, averaged over the 10 seeds.
Paper (Sec. IV.B): "The true subgroup remains worth a similar discovery-set likelihood gain under all four models
(13-17 nats on average)". Writes <artistry>/systems/discovery_oracle_gain.csv."""
import os, numpy as np, pandas as pd
from common import *
from ladder_data import load_dataset
from mdss_treescan_stage1 import exact_subgroup_logit_correction as exact
LOGITS = {"XGBoost": "xgboost", "CatBoost": "catboost", "TabPFN-2.5": "tabpfn25_ladder", "TabICLv2": "tabicl"}
rows = []
for name, folder in LOGITS.items():
    for seed in range(1, 11):
        D = load_dataset("synth8", seed); z = np.load(ARTISTRY / "logits" / folder / f"synth8_seed{seed}_ctxfull.npz")
        y = D.y[D.discovery]; assert (z["yd"] == y).all(), "logits file does not match the regenerated split"
        r = exact(y, z["zd"], D.truth[D.discovery], l2=0.0, min_size=1, min_events=0, min_nonevents=0)
        rows.append(dict(model=name, seed=seed, oracle_gain_nats=r.gain, delta=r.delta))
d = pd.DataFrame(rows); os.chdir(out_dir(ARTISTRY / "systems")); d.to_csv("discovery_oracle_gain.csv", index=False)
print(d.groupby("model", sort=False).oracle_gain_nats.mean().round(1).to_string())
