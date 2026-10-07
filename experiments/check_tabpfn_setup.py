"""Preflight check for the modern-TabPFN reruns. Run this first:  python check_tabpfn_setup.py
It checks the GPU, the tabpfn package, the licence/Hugging Face tokens, and then does one tiny
fit + predict, which triggers the one-time licence acceptance and weight download."""
import os, sys, time, importlib.util
from pathlib import Path
os.environ.setdefault("TABPFN_BACKEND", "modern")
ok = True
def line(tag, msg): print(f"[{tag:4s}] {msg}")

try:
    import torch
    line("ok", f"torch {torch.__version__}")
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        line("ok", f"GPU: {g.name}, {g.total_memory/2**30:.1f} GB")
        if g.total_memory < 12 * 2**30: line("warn", "under 12 GB of GPU memory: use TABPFN_CONTEXT=1000 or a smaller --chunk if you hit out-of-memory")
    else:
        line("warn", "no CUDA GPU visible: modern TabPFN will run on CPU, which is very slow for full contexts")
except ImportError:
    line("FAIL", "torch not installed"); ok = False

if importlib.util.find_spec("tabpfn") is None:
    line("FAIL", "tabpfn not installed: pip install tabpfn==9.1.0"); sys.exit(1)
from importlib.metadata import version as _v
v = _v("tabpfn"); line("ok" if v.startswith("9.") else "warn", f"tabpfn {v} (tested with 9.1.0)")

tok = os.environ.get("TABPFN_TOKEN") or next((p.read_text().strip() for p in
       [Path.home()/".cache"/"tabpfn"/"auth_token", Path.home()/".tabpfn"/"token"] if p.is_file()), None)
line("ok" if tok else "info", "Prior Labs licence token found" if tok else
     "no Prior Labs licence token yet: the tiny fit below will open a browser to accept the licence (one time)")
hf = os.environ.get("HF_TOKEN") or (Path.home()/".cache"/"huggingface"/"token").is_file()
line("ok" if hf else "warn", "Hugging Face token found" if hf else
     "no Hugging Face token: run `huggingface-cli login` (or set HF_TOKEN) after accepting the model terms on huggingface.co")

ver = os.environ.get("TABPFN_VERSION", "v2.5")
try:
    import numpy as np
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    rng = np.random.default_rng(0); X = rng.normal(size=(400, 8)); y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
    t = time.perf_counter()
    clf = TabPFNClassifier.create_default_for_version(ModelVersion(ver), device=os.environ.get("TABPFN_DEVICE", "auto"), random_state=0)
    clf.fit(X[:300], y[:300]); p = clf.predict_proba(X[300:])[:, 1]
    acc = ((p > 0.5) == y[300:]).mean()
    line("ok", f"TabPFN {ver}: tiny fit+predict in {time.perf_counter()-t:.1f}s, accuracy {acc:.2f} (expect > 0.8)")
except Exception as e:
    ok = False; line("FAIL", f"TabPFN {ver} could not load: {type(e).__name__}: {str(e)[:400]}")
print("\nREADY" if ok else "\nNOT READY: fix the FAIL lines above (see README_MODERN_TABPFN.md, 'Troubleshooting')")
