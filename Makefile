# Reproduce "Fix What the Model Leaves Behind" (AccML 2027).  See README.md for the four levels.
PY        ?= python3
ART       ?= $(abspath ../scan-repair-foundation-models-artistry)
export SSR_ARTISTRY := $(ART)
export SOURCE_DATE_EPOCH := 0          # deterministic figure files (no creation timestamp)
EXP       := experiments
BUILD     := paper/build
LATEX     := paper/latex

.PHONY: help paper-inputs paper check smoke corrections reports trees systems timing-cpu clean

help:
	@echo "Level A  make paper-inputs   tables, figures and numbers from the released results  (minutes, CPU)"
	@echo "         make paper          ... and compile paper/latex/main.pdf                  (needs pdflatex + bibtex)"
	@echo "         make check          rebuild and verify every generated .tex/.csv is unchanged"
	@echo "Level B  make smoke          re-run all corrections for one saved logits file and compare (about 1 min)"
	@echo "         make corrections    re-run every correction from the saved logits         (5-6 CPU hours)"
	@echo "         make reports        ladder report tables from the corrections"
	@echo "Level C  make trees          re-train XGBoost/CatBoost and save their logits       (CPU hours)"
	@echo "         foundation models:  notebooks/foundation_model_logits_colab.ipynb         (GPU)"
	@echo "Level D  make systems        profile, speed-up, histogram fidelity, scorer calls   (about 1 hour, CPU)"
	@echo "         make timing-cpu     TabPFN-2.5 cost per 1,000 rows on this CPU            (hours)"
	@echo "Artistry are read from ART=$(ART)"

# ---------------- Level A ----------------
paper-inputs:
	cd $(BUILD) && $(PY) make_bench_inputs.py && $(PY) make_paper_ladder_inputs.py && $(PY) make_ladder_figs.py \
	  && $(PY) make_fig_cost.py && $(PY) make_numbers_systems.py

paper: paper-inputs
	cd $(LATEX) && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
	  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null \
	  && echo "-> $(LATEX)/main.pdf"

check: paper-inputs
	git diff --stat --exit-code -- '$(LATEX)/*.tex' '$(LATEX)/fig_*.pdf' 'results/' && echo "Level A: every generated table and number is identical to the committed version."

# ---------------- Level B ----------------
smoke:
	$(PY) tests/smoke_corrections.py

corrections:
	@for s in tabpfn25_ladder xgboost catboost tabicl tabicl_a100 discovery/xgboost discovery/catboost discovery/tabpfn discovery/tabicl; do \
	  case $$s in discovery/*) M="frozen,Platt,beta,isotonic,scan,Platt->scan,oracle subgroup";; *) M="";; esac; \
	  echo "== $$s"; METHODS="$$M" NUMBA_NUM_THREADS=1 $(PY) $(EXP)/corrections.py $(ART)/logits/$$s reproduced/corrections/$$s || exit 1; \
	  $(PY) tests/compare_corrections.py $(ART)/corrections/$$s/posthoc.csv reproduced/corrections/$$s/posthoc.csv || exit 1; done

reports:
	$(PY) $(EXP)/ladder_report.py $(ART)/corrections/tabpfn25_ladder reproduced/report $(ART)/timing/timing_cpu_2core.csv $(ART)/timing/timing_gpu_t4.csv

# ---------------- Level C ----------------
trees:
	cd $(EXP) && $(PY) base_trees.py xgboost $(abspath reproduced/logits) && $(PY) base_trees.py catboost $(abspath reproduced/logits)
	cd $(EXP) && for m in xgboost catboost; do TUNED_DIR=$(abspath reproduced/logits)/$$m DATASETS="synth8x4d4 synth1x4d4" \
	  $(PY) base_trees.py $$m $(abspath reproduced/logits/discovery) && $(PY) make_discovery_small.py $(abspath reproduced/logits/discovery)/$$m; done

# ---------------- Level D ----------------
systems:
	cd $(EXP) && OUT=$(abspath reproduced/systems) $(PY) systems_profile.py && OUT=$(abspath reproduced/systems) $(PY) credit_baseline.py \
	  && OUT=$(abspath reproduced/systems) $(PY) systems_speedup.py && OUT=$(abspath reproduced/systems) $(PY) systems_fidelity.py \
	  && OUT=$(abspath reproduced/systems) $(PY) systems_scorer_calls.py && OUT=$(abspath reproduced/systems) $(PY) discovery_oracle_gain.py

timing-cpu:
	cd $(EXP) && CKPT=$(abspath reproduced/timing_tabpfn_v2_5_random_weights.ckpt) $(PY) make_timing_checkpoint.py \
	  && TABPFN_BACKEND=modern TABPFN_DEVICE=cpu TABPFN_MODEL_PATH=$(abspath reproduced/timing_tabpfn_v2_5_random_weights.ckpt) \
	     OUT=$(abspath reproduced/timing) $(PY) timing_tabpfn.py

clean:
	rm -rf reproduced
	cd $(LATEX) && rm -f *.aux *.bbl *.blg *.log *.out main.pdf
