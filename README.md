# Scan Repair for Foundation Models

**Fast, interpretable subgroup repair for tabular foundation models (TabPFN, TabICL) and gradient-boosted trees (XGBoost, CatBoost)**

Code for the AccML 2027 paper *Fix What the Model Leaves Behind: Fast Subgroup Repair for Tree Ensembles and Tabular
Foundation Models*. It contains the method, every experiment, and the scripts that rebuild every table, figure and
number in the paper.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<!-- After the first Zenodo release, add the DOI badges here, e.g.
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Artifacts DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.YYYYYYY.svg)](https://doi.org/10.5281/zenodo.YYYYYYY) -->

---

## What it does

A tabular classifier can be accurate on average yet systematically wrong for one **subgroup defined by a combination
of features**. Tuning and global recalibration (temperature, Platt, beta, isotonic) do not remove such errors.
**Structured Scan Repair** is a post-hoc layer for any frozen probabilistic classifier. It:

1. searches interpretable conjunctions on the original features (`age > 52 AND limit <= 30,000 AND ...`) for the
   largest exact Bernoulli log-likelihood gain of one log-odds shift,
   Δ(S, δ) = Σ<sub>i∈S</sub> [ y<sub>i</sub> δ − log(1 − p<sub>i</sub> + e<sup>δ</sup> p<sub>i</sub>) ];
2. keeps a rule only if it improves two held-out halves of a selection set with the same sign and passes a
   likelihood-ratio test;
3. returns F′(x) = F(x) + Σ<sub>t</sub> δ<sub>t</sub>·1[x ∈ S<sub>t</sub>]: a few named rules with one number each, or
   **no change at all** (abstention).

The base model is never refitted, and for an in-context model its context is never changed. The repair therefore
sits on top of a tuned XGBoost model as easily as on TabPFN-2.5.

## Main results

- **One correction problem across model families.** Tuned XGBoost and CatBoost, TabPFN-2.5 and TabICLv2 all leave a
  planted subgroup error (absolute bias 0.15–0.16), and global recalibration removes only 1–22% of it.
- **Given enough data to search, the repair removes it in every model.** With 9,600 discovery rows the scan removes
  67–73% of the subgroup bias and recovers 68–84% of the best possible one-rule gain. When no subgroup effect is
  planted, its mean effect stays within ±0.5×10⁻³.
- **Repair is worth context.** TabPFN-2.5 with a 1,000-row context plus the repair matches the uncorrected model at
  2,152 rows (Credit Default) and 4,175 rows (planted subgroup). That makes every prediction 1.8–4.4× cheaper on a CPU
  than buying the same accuracy with more context. The repair also flags the rows that the larger context later fixes.
- **Exact acceleration.** Compiled, batched and parallel scoring speeds a fit up 3.9–6.2× on a 2-core CPU, with
  bit-identical repairs.
- **Honest limits.** On Credit Default a boosted residual corrector (tens to hundreds of opaque leaves) gains more
  log-loss than the scan. If compute were free, putting all held-out labels into TabPFN's context would be more
  accurate than repairing a small context.

## Quick start

```bash
git clone https://github.com/<GITHUB-USER>/scan-repair-foundation-models
git clone https://github.com/<GITHUB-USER>/scan-repair-foundation-models-artifacts   # saved logits and results (~80 MB)
cd scan-repair-foundation-models
pip install -r requirements.txt           # CPU only; Python 3.11+ (tested on 3.13)

python examples/repair_your_model.py synthetic    # finds the planted rule: x0 > 0.39 AND ... AND x4 > 0.40
python examples/repair_your_model.py credit       # Credit Default: little localized error left, the scan abstains
make paper-inputs                                 # every table, figure and number of the paper (minutes)
```

To repair your own model you need three things for labelled rows the model was not trained on: the original features
(a DataFrame), the labels and the model's logits. `examples/repair_your_model.py` shows the five lines that matter.

## Reproducing the paper

The artifacts repository holds the saved base-model logits, every correction result and the timings. Clone it next to
this repository, or point `SSR_ARTIFACTS` at it. There are four levels, from cheapest to most expensive:

| Level | Reproduces | Command | Needs | Time | Exact? |
|---|---|---|---|---|---|
| **A** | Every table, figure and in-text number from the released results; then the PDF | `make paper-inputs`, `make paper`, `make check` | CPU (+ LaTeX for the PDF) | minutes | Yes: `make check` verifies every generated table, number and figure is unchanged |
| **B** | Every correction (Platt … scan … boosted corrector) re-fitted from the saved logits | `make smoke` (1 file, ~1 min), `make corrections` (all 520 logits files) | CPU | ~5–6 CPU-hours (split with `SHARD=0/2`, `SHARD=1/2`) | Yes: compared row by row with the released results (identical to machine precision) |
| **C** | The base models themselves | `make trees` (XGBoost, CatBoost); `notebooks/foundation_model_logits_colab.ipynb` (TabPFN-2.5, TabICLv2) | CPU hours; a GPU for the foundation models | hours | Trees: identical logits when refitted with the released hyperparameters (`TUNED_DIR=<artifacts>/logits/<model>`; checked). Foundation models: close, not bit-identical, because GPU kernels differ by device (an A100 rerun of TabICLv2 moved test log-loss by ≤ 7×10⁻⁵ per run) |
| **D** | Timings and the speed-up | `make systems`, `make timing-cpu`; GPU timing in the notebook | the matching hardware | ~1 h (systems), hours (CPU timing) | Same pattern; absolute seconds depend on the machine |

### Paper item → code

| Paper | Built by (`paper/build/`) | From (artifacts) | Produced by (`experiments/`) |
|---|---|---|---|
| Table I (benchmark, all corrections) | `make_bench_inputs.py` | `corrections/{xgboost,catboost,tabicl}/`, `corrections/tabpfn25_ladder/` (full context) | `corrections.py` ← `base_trees.py`, `base_tabicl.py`, `base_tabpfn_ladder.py` |
| Table II, Fig. 1 (discovery power, n = 19,200) | `make_bench_inputs.py` | `corrections/discovery/*` | `corrections.py` ← `base_trees.py` (`DATASETS="synth8x4d4 synth1x4d4"`), `base_fm_discovery.py`, `make_discovery_small.py` |
| "13–17 nats" (Sec. IV.B) | `make_numbers_systems.py` | `systems/discovery_oracle_gain.csv` | `discovery_oracle_gain.py` |
| Fig. 2, Table III (context ladder) | `make_ladder_figs.py`, `make_paper_ladder_inputs.py` | `corrections/tabpfn25_ladder/` | `corrections.py` ← `base_tabpfn_ladder.py` |
| Full correction-by-context table | `make_paper_ladder_inputs.py` → `results/context_ladder_all_corrections.csv` | same | same |
| Fig. 3 (attribution) | `make_ladder_figs.py` | `corrections/tabpfn25_ladder/report/T8_*` | `ladder_report.py` (from `rows/`) |
| Fig. 4, Table IV, equivalent contexts, resource-accounting paragraph | `make_ladder_figs.py`, `make_paper_ladder_inputs.py` | `report/T9_*`, `timing/timing_cpu_2core.csv`, `timing/timing_gpu_t4.csv` | `ladder_report.py`, `timing_tabpfn.py` (+ `make_timing_checkpoint.py` for the CPU) |
| Fig. 5, Sec. VI (22.8 s → 4.5 s, 3.9–6.2×, identical repairs) | `make_fig_cost.py`, `make_numbers_systems.py` | `systems/cost4_*.csv` | `systems_speedup.py` |
| "76% of fit time", "~65,000 scorer calls" | `make_numbers_systems.py` | `systems/cost_breakdown.csv`, `systems/scorer_calls.json` | `systems_profile.py`, `systems_scorer_calls.py` |
| B = 64 logit-histogram check | `make_numbers_systems.py` | `systems/fidelity.json` | `systems_fidelity.py` |
| Reproducibility: A100 vs T4, Credit baseline 0.4338 | `make_numbers_systems.py` | `corrections/tabicl_a100/`, `systems/credit_baseline.csv` | `gpu_consistency.py`, `credit_baseline.py` |

Generated LaTeX inputs (`numbers*.tex`, `tab_*.tex`, `fig_*.pdf`) are written to `paper/latex/` next to `main.tex`,
where the paper uses them as macros. The few values typed directly into the text are produced by the scripts above,
for example "13–17 nats", "~65,000 calls" and "7×10⁻⁵".

## Repository layout

```
scan-repair-foundation-models/
├── src/structured_scan/   Structured Scan Repair (scorer, beam search, validation, repair rounds); see its README
├── src/accel/             exact speed-ups: compiled scorer, batched + parallel scoring; logit-histogram score
├── experiments/           base models, corrections, reports, timing, systems experiments (one script each)
├── paper/build/           scripts that turn artifacts into numbers*.tex, tab_*.tex and fig_*.pdf
├── paper/latex/           main.tex and its generated inputs
├── results/               released tables not printed in the paper (e.g. the full context-ladder table)
├── notebooks/             Colab notebook for the GPU runs (TabPFN-2.5, TabICLv2)
├── examples/              repair your own model
├── tests/                 smoke test and result comparison
└── data/credit_default/   UCI Default of Credit Card Clients (CC BY 4.0); synthetic data are generated in code
```

## Data, models and licences

- **Code:** MIT (see `LICENSE`).
- **Credit Default:** Yeh, I.-C. (2009), UCI Machine Learning Repository, CC BY 4.0, https://doi.org/10.24432/C55S3H.
  Split: 15,000 rows for the base model, then 6,000 / 4,500 / 4,500 for discovery / selection / test. The test rows
  are the same in every seed.
- **Synthetic data:** `mdss_treescan_stage1.generate_structured_residual_data`. 30 Uniform(0,1) features and a
  six-feature logistic background signal (x5..x10). The planted 5-way conjunction x0..x4 > 1 − 0.08^{1/5} covers
  8% of rows and multiplies their odds by θ = 8; the "no planted effect" control uses θ = 1. The intercept is solved
  for each generated dataset, after the subgroup term is included, so that the mean event probability is 10%.
- **TabPFN-2.5** (`tabpfn==9.1.0`): the weights are gated. Accept the licence at
  https://huggingface.co/Prior-Labs/tabpfn_2_5 and set `HF_TOKEN` (and `TABPFN_TOKEN`). They are not redistributed
  here. The CPU timings use the same architecture with random weights (`make_timing_checkpoint.py`): identical
  computation, meaningless predictions.
- **TabICLv2** (`tabicl==2.2.0`, checkpoint `tabicl-classifier-v2-20260212.ckpt`): downloaded automatically.
- **Hardware used:** foundation-model logits and GPU timings on a free-tier Colab T4; corrections and CPU timings on a
  2-core x86-64 cloud machine.

## Citing

See `CITATION.cff` (GitHub shows a "Cite this repository" button). Please also cite the archived artifacts (Zenodo DOI
in the artifacts repository).

## Author

<AUTHOR NAME> (<AFFILIATION>).
