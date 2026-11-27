"""Check that CPU timing does not depend on TabPFN-2.5 weight values (supports Sec. III, Timing).
Builds the random-weight checkpoint (same architecture as the released model, 10.7M parameters), makes copies
with every weight scaled by 0.1x and 3x, and times the same predictions (1,000-row Credit context, 300 rows)
with each. Equal times (within run-to-run variation) mean the compute depends on the architecture only.
  python check_timing_weights.py          # needs tabpfn==9.1.0; writes check_timing_weights.csv
Two runs on a 2-core CPU (scales 1, 0.1, 3): 11.8/12.2/12.6 s and 11.8/12.7/11.8 s; the ordering changes
between runs, so the differences are run-to-run noise (single-model repeats ranged 11.6-13.0 s)."""
import os, sys, subprocess, tempfile, numpy as np, pandas as pd, torch, dataclasses
from tabpfn.architectures import tabpfn_v2_5 as A
if len(sys.argv) > 1 and sys.argv[1] == "--time":        # child process: time one checkpoint
    import time
    from common import *
    import tabpfn_common as TC
    from ladder_data import load_dataset
    D = load_dataset("credit", 42001); perm = np.random.default_rng(0).permutation(D.train); test = D.test[:300]
    TC.FrozenTabPFN(D.X[perm[:200]], D.y[perm[:200]], seed=0, categorical_idx=D.cat_idx).logits(D.X[test[:50]])  # warm-up
    ts = []
    for r in range(3):
        M = TC.FrozenTabPFN(D.X[perm[:1000]], D.y[perm[:1000]], seed=r, categorical_idx=D.cat_idx)
        t = time.perf_counter(); M.logits(D.X[test]); ts.append(time.perf_counter() - t)
    print(float(np.median(ts))); sys.exit(0)
cfg = A.TabPFNV2p5Config(max_num_classes=10); model = A.get_architecture(cfg)
sd = model.state_dict(); n = sum(v.numel() for v in sd.values())
tmp = tempfile.mkdtemp(); rows = []
for scale in (1.0, 0.1, 3.0):
    path = os.path.join(tmp, f"w{scale}.ckpt")
    torch.save({"state_dict": {k: (v * scale if v.is_floating_point() else v) for k, v in sd.items()},
                "config": dataclasses.asdict(cfg), "architecture_name": "tabpfn_v2_5"}, path)
    env = dict(os.environ, TABPFN_MODEL_PATH=path, TABPFN_DEVICE="cpu")
    out = subprocess.run([sys.executable, __file__, "--time"], env=env, capture_output=True, text=True, check=True)
    rows.append(dict(weight_scale=scale, parameters=n, median_seconds_300_rows=float(out.stdout.strip().splitlines()[-1])))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("check_timing_weights.csv", index=False)
