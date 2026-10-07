"""Datasets and splits for the context ladder. Identical splits to the paper's earlier TabPFN experiments
(tabpfn_synth.py, tabpfn_credit_v2.py); everything here is deterministic given (dataset, seed)."""
import numpy as np
from types import SimpleNamespace
from common import *

DATASET_SEEDS = {"synth8": list(range(1, 11)), "synth1": list(range(1, 11)), "credit": list(range(42001, 42011)),
                 "synth8x4d1": list(range(1, 11)), "synth8x4d4": list(range(1, 11)),
                 "synth1x4d1": list(range(1, 11)), "synth1x4d4": list(range(1, 11))}
# Discovery-size arm: same generator, n = 19,200; train 6,000 / discovery 9,600 / selection 1,800 / test 1,800.
# "...d4" scans all 9,600 discovery rows, "...d1" only the first 2,400 of them (the main benchmark's size);
# base model, selection and test rows are identical between d1 and d4.
X4 = dict(n=19200, train=6000, disc=9600, sel=1800, test=1800, small=2400)

def load_dataset(name, seed):
    if "x4" in name:
        from mdss_treescan_stage1 import SyntheticConfig, generate_structured_residual_data
        from block_b_correction_benchmarks import BlockBCondition
        odds = float(name[5:name.index("x4")]); n, p, k, s = X4["n"], 30, 5, 0.08
        d = generate_structured_residual_data(SyntheticConfig(n=n, p=p, interaction_order=k, subgroup_prevalence=s,
            subgroup_log_odds=float(np.log(odds)), event_prevalence=0.10, seed=seed))
        perm = np.random.default_rng(seed + 777).permutation(n); a = X4["train"]; b = a + X4["disc"]; c = b + X4["sel"]
        disc = perm[a:b] if name.endswith("d4") else perm[a:a + X4["small"]]
        X = np.asarray(d.X, float); names = [f"x{j}" for j in range(X.shape[1])]
        return SimpleNamespace(name=name, seed=seed, X=X, y=np.asarray(d.y, int), train=perm[:a], discovery=disc,
            selection=perm[b:c], test=perm[c:c + X4["test"]], cat_idx=None, model_seed=seed, kind="synth",
            truth=np.asarray(d.subgroup_mask, bool), true_logits=np.asarray(d.true_logits), odds=odds,
            cond=BlockBCondition(support=s, odds_ratio=odds, seed=seed, n=n, p=p, k=k),
            feature_names=names, numeric_names=names, categorical_names=[], frame=None, bundle=None)
    if name.startswith("synth"):
        from mdss_treescan_stage1 import SyntheticConfig, generate_structured_residual_data
        from crossfit_repair_benchmark_strict import four_way_split
        from block_b_correction_benchmarks import BlockBCondition
        odds = float(name[5:]); n, p, k, s = 12000, 30, 5, 0.08
        d = generate_structured_residual_data(SyntheticConfig(n=n, p=p, interaction_order=k, subgroup_prevalence=s,
            subgroup_log_odds=float(np.log(odds)), event_prevalence=0.10, seed=seed))
        sp = four_way_split(n, seed=seed)
        X = np.asarray(d.X, float)
        names = [f"x{j}" for j in range(X.shape[1])]
        return SimpleNamespace(name=name, seed=seed, X=X, y=np.asarray(d.y, int), train=np.asarray(sp.train), discovery=np.asarray(sp.discovery),
            selection=np.asarray(sp.selection), test=np.asarray(sp.test), cat_idx=None, model_seed=seed, kind="synth",
            truth=np.asarray(d.subgroup_mask, bool), true_logits=np.asarray(d.true_logits), odds=odds,
            cond=BlockBCondition(support=s, odds_ratio=odds, seed=seed, n=n, p=p, k=k),
            feature_names=names, numeric_names=names, categorical_names=[], frame=None, bundle=None)
    if name == "credit":
        bundle = ext.load_credit_default(DATA, allow_download=False)
        Xf, y = bundle.X, np.asarray(bundle.y, int)
        split = ext.make_extended_split(bundle, seed=seed)
        enc = Xf.copy()
        for c in bundle.categorical_columns:
            cats = {v: i for i, v in enumerate(sorted(Xf.iloc[split.train][c].astype(str).unique()))}
            enc[c] = Xf[c].astype(str).map(cats).fillna(-1)
        return SimpleNamespace(name=name, seed=seed, X=enc.to_numpy(float), y=y, train=np.asarray(split.train),
            discovery=np.asarray(split.discovery), selection=np.asarray(split.selection), test=np.asarray(split.test),
            cat_idx=[list(Xf.columns).index(c) for c in bundle.categorical_columns], model_seed=seed % 1000, kind="real",
            truth=None, true_logits=None, odds=None, cond=None, feature_names=list(Xf.columns),
            numeric_names=list(bundle.numeric_columns), categorical_names=list(bundle.categorical_columns), frame=Xf, bundle=bundle)
    raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASET_SEEDS)}")
