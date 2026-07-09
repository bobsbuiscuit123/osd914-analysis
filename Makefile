PYTHON ?= .venv/bin/python

.PHONY: reproduce smoke-test

reproduce:
	$(PYTHON) reproduce.py

smoke-test:
	$(PYTHON) reproduce.py --seeds 101 --epochs 5 --skip-shap --output-dir reproducibility_smoke
