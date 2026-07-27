PYTHON ?= .venv/bin/python

.PHONY: reproduce baselines permutation-baselines smoke-test baseline-smoke check package clean-smoke

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

check:
	$(PYTHON) -m py_compile reproduce.py baselines.py clean_data.py ai.py figures/generate_nasa_figures.py

package:
	$(PYTHON) -m build --sdist

clean-smoke:
	rm -rf reproducibility_smoke reproducibility_baseline_smoke
