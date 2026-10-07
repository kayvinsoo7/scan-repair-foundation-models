"""Paths shared by every script.

REPO       this repository
SRC        src/structured_scan (the Structured Scan Repair implementation) and src/accel (exact speed-ups)
DATA       data/ (Credit Default; the synthetic data are generated in code)
ARTIFACTS  the companion artifacts repository (saved logits, correction results, timings).
           Default: a sibling folder ../scan-repair-foundation-models-artifacts; override with SSR_ARTIFACTS.
"""
import sys, os
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = Path(HERE).resolve().parent
for sub in ("src/structured_scan", "src/accel"):
    p = str(REPO / sub)
    if p not in sys.path: sys.path.insert(0, p)
if HERE not in sys.path: sys.path.insert(0, HERE)
ARTIFACTS = Path(os.environ.get("SSR_ARTIFACTS", REPO.parent / "scan-repair-foundation-models-artifacts")).resolve()
DATA = REPO / "data"
import numpy as np
import structured_scan_realdata_benchmarks as core
import structured_scan_realdata_extended_mechanism as ext

def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

def out_dir(default):
    """Output folder for scripts that write several files: $OUT if set, else `default`; created and returned."""
    d = Path(os.environ.get("OUT", default)); d.mkdir(parents=True, exist_ok=True); return d
