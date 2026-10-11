"""Where the paper-build scripts read from and write to.
ART      the artifacts repository (override with SSR_ARTIFACTS); default: sibling folder of this repository
LATEX    paper/latex: generated numbers_*.tex, tab_*.tex and fig_*.pdf go here, next to main.tex
RESULTS  results/: tables and summaries that are released with the code but not printed in the paper"""
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
ART = Path(os.environ.get("SSR_ARTIFACTS", REPO.parent / "scan-repair-foundation-models-artifacts")).resolve()
LATEX = REPO / "paper" / "latex"
RESULTS = REPO / "results"
LADDER = ART / "corrections" / "tabpfn25_ladder"          # posthoc.csv, rows/, report/
TIMING_CPU = ART / "timing" / "timing_cpu_2core.csv"
TIMING_GPU = ART / "timing" / "timing_gpu_t4.csv"
SYSTEMS = ART / "systems"
ICL_LADDER = ART / "corrections" / "tabicl_ladder"       # TabICLv2 context ladder (second foundation model)
TIMING_ICL_CPU = ART / "timing" / "timing_tabicl_cpu_2core.csv"
TIMING_ICL_GPU = ART / "timing" / "timing_tabicl_gpu_t4.csv"
for d in (LATEX, RESULTS): d.mkdir(parents=True, exist_ok=True)
if not ART.exists():
    raise SystemExit(f"Artifacts not found at {ART}. Clone the artifacts repository next to this one, or set SSR_ARTIFACTS.")
