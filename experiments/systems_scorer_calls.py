"""How many times one default repair fit calls the exact scorer (Eq. 2), and how long the fit takes.
Paper: "invoked roughly 65,000 times for a standard 2,400-row, 30-feature discovery set" (we measured 64,813).
Writes <artifacts>/systems/scorer_calls.json."""
import os, sys, json
from common import *
os.chdir(out_dir(ARTIFACTS / "systems"))
import profile_scan as P
import mdss_treescan_stage1 as M
orig = M.exact_subgroup_logit_correction; calls = [0]
def counted(*a, **k):
    calls[0] += 1; return orig(*a, **k)
patched = 0
for mod in list(sys.modules.values()):          # every module that imported the scorer by name
    if mod is not None and getattr(mod, "exact_subgroup_logit_correction", None) is orig:
        mod.exact_subgroup_logit_correction = counted; patched += 1
seconds, L, d, sp = P.run(n=12000, p=30, seed=1)
out = dict(scorer_calls=calls[0], fit_seconds=seconds, discovery_rows=int(len(sp.discovery)), features=30,
           accepted_terms=len(L.bases_), modules_patched=patched)
json.dump(out, open("scorer_calls.json", "w"), indent=1); print(out)
