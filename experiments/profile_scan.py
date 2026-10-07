"""Default synthetic problem used by the systems experiments (n=12,000, 30 features, planted 5-way subgroup,
fixed-budget XGBoost base) and a helper that fits the frozen scanner on it."""
import sys, time, cProfile, pstats, io
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import common  # noqa: F401  (sets up src/ on the path)
import numpy as np
from mdss_treescan_stage1 import SyntheticConfig, generate_structured_residual_data, sigmoid
from benchmark_treescan import fit_fixed_budget_xgb
from crossfit_repair_benchmark_strict import four_way_split
from block_b_correction_benchmarks import BlockBCondition
from final_scanner_block_b import make_final_scanner
import xgboost as xgb

def setup(n=12000, p=30, k=5, support=0.08, odds=8.0, seed=1):
    cond = BlockBCondition(support=support, odds_ratio=odds, seed=seed, n=n, p=p, k=k)
    cfg = SyntheticConfig(n=n, p=p, interaction_order=k, subgroup_prevalence=support,
                          subgroup_log_odds=float(np.log(odds)), event_prevalence=0.10, seed=seed)
    d = generate_structured_residual_data(cfg)
    sp = four_way_split(n, seed=seed)
    m = fit_fixed_budget_xgb(d.X[sp.train], d.y[sp.train], max_depth=2, rounds=70, seed=seed)
    def lg(idx):
        return m.booster.predict(xgb.DMatrix(d.X[idx]), output_margin=True) if hasattr(m,'booster') else None
    return cond, d, sp, m

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv)>1 else 12000
    p = int(sys.argv[2]) if len(sys.argv)>2 else 30
    cond, d, sp, m = setup(n=n, p=p)
    print(type(m), [a for a in dir(m) if not a.startswith('_')])

def run(n=12000, p=30, prof=False, seed=1):
    cond, d, sp, m = setup(n=n, p=p, seed=seed)
    zd = m.logits(d.X[sp.discovery]); zs = m.logits(d.X[sp.selection])
    learner = make_final_scanner(cond, discovery_n=len(sp.discovery), selection_n=len(sp.selection))
    args = (d.X[sp.discovery], d.y[sp.discovery], zd, d.X[sp.selection], d.y[sp.selection], zs)
    if prof:
        pr = cProfile.Profile(); t=time.perf_counter(); pr.enable(); learner.fit(*args); pr.disable(); el=time.perf_counter()-t
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18); print(s.getvalue()[:6000])
    else:
        t=time.perf_counter(); learner.fit(*args); el=time.perf_counter()-t
    return el, learner, d, sp
