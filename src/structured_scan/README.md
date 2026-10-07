# `src/structured_scan/` — the Structured Scan Repair implementation

These modules are the implementation of Structured Scan Repair from the author's earlier study, included
**unchanged** so that the published results can be reproduced exactly. Only the modules the paper's experiments
import are included.

Where to look:

| What | Where |
|---|---|
| Exact subgroup score (Eq. 2) and its Newton solve | `mdss_treescan_stage1.py` → `exact_subgroup_logit_correction` |
| Adaptive beam search over conjunctions (proxy screen + exact shortlist) | `adaptive_structured_scanner_v2_1.py`, `adaptive_structured_scanner_v2_2.py` |
| Validation on two selection halves, likelihood-ratio test, repair rounds, joint refit (Algorithm 1) | `adaptive_structured_scanner_v2_3.py` → `StabilitySelectedScanRepairLearner.fit` |
| Scanner used for the synthetic data | `final_scanner_block_b.py` → `make_final_scanner` |
| Scanner used for real data, scanner preprocessing, readable rules | `structured_scan_realdata_benchmarks.py` → `make_realdata_scanner`, `CompactScanPreprocessor`, `describe_basis` |
| Placebo-controlled audit of accepted rules (Credit Default) | `structured_scan_realdata_extended_mechanism.py` → `mechanistic_term_audit` |
| Synthetic generator (planted k-way subgroup) | `mdss_treescan_stage1.py` → `generate_structured_residual_data` |

The exact speed-ups of Sec. VI live in `src/accel/`: `fastexact.py` (compiled scorer), `fastscan.py` (batched,
parallel candidate scoring) and `fastscore.py` (the exploratory logit-histogram score). `install()` swaps them in;
every candidate score is bit-identical to the original.
