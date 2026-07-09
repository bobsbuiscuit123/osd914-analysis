# Reproducibility

Run the full reproducible analysis from the raw gzipped inputs with:

```bash
make reproduce
```

or directly:

```bash
.venv/bin/python reproduce.py
```

The canonical pipeline is `reproduce.py`. It rebuilds clean liver/brain matrices from the raw files, uses fold-wise log1p plus z-score normalization fitted only on each LOOCV training fold, trains the MLP under fixed repeated seeds, aggregates SHAP across folds and seeds, and writes all outputs under `reproducibility/`.

The full run was verified on 2026-07-09 by running `make reproduce` twice and checking that `reproducibility/metrics.json` and every CSV in `reproducibility/tables/` were byte-identical across runs.

Current regenerated summary:

- Cohorts: `FL=3`, `HC=3`, `VC=3`
- Clean matrices: liver `9 x 717`, brain `9 x 637`
- Brain source: `GSE294046` brain miRNA columns
- Seeds: `101,202,303,404,505`
- VMSE mean: `1.7026`
- VMSE standard deviation: `0.1397`
- VMSE range: `1.4880-1.8820`
- Top aggregated SHAP feature: `mmu-miR-17-5p`

Primary outputs:

- `reproducibility/metrics.json`
- `reproducibility/tables/metrics_by_seed.csv`
- `reproducibility/tables/fold_metrics.csv`
- `reproducibility/tables/predictions.csv`
- `reproducibility/tables/shap_values.csv`
- `reproducibility/tables/shap_importance_summary.csv`
- `reproducibility/tables/expression_group_summary.csv`
- `reproducibility/figures/*.png`
- `reproducibility/dashboard.html`

The older `ai.py` dashboard is kept for exploratory interactive runs. It is not the canonical reproducibility entry point because it uses global normalization and stochastic single-run outputs.

For a quick wiring check without SHAP, run:

```bash
make smoke-test
```
