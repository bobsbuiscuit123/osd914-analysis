PYTHON ?= .venv/bin/python

.PHONY: reproduce baselines permutation-baselines smoke-test baseline-smoke

reproduce:
	$(PYTHON) reproduce.py

baselines:
	$(PYTHON) baselines.py

permutation-baselines:
	$(PYTHON) baselines.py --permutations 1000

smoke-test:
	$(PYTHON) reproduce.py --seeds 101 --epochs 5 --skip-shap --output-dir reproducibility_smoke

baseline-smoke:
	$(PYTHON) baselines.py --permutations 5 --mlp-seeds 101 --epochs 5 --output-dir reproducibility_baseline_smoke
