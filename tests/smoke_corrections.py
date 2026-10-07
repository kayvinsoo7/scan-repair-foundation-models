"""Smoke test (about a minute): re-run every correction on one saved TabPFN-2.5 logits file (Credit Default, seed
42001, 1,000-row context) and check the results are identical to the released ones."""
import os, subprocess, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("SSR_ARTIFACTS", REPO.parent / "scan-repair-foundation-models-artifacts"))
out = Path(tempfile.mkdtemp(prefix="ssr_smoke_"))
env = dict(os.environ, PATTERN="credit_seed42001_ctx1000.npz", NUMBA_NUM_THREADS="1")
subprocess.run([sys.executable, str(REPO / "experiments" / "corrections.py"), str(ART / "logits" / "tabpfn25_ladder"), str(out)],
               check=True, env=env)
r = subprocess.run([sys.executable, str(REPO / "tests" / "compare_corrections.py"),
                    str(ART / "corrections" / "tabpfn25_ladder" / "posthoc.csv"), str(out / "posthoc.csv")])
sys.exit(r.returncode)
