# Reproducibility

Run the full reproducible analysis from the raw gzipped inputs with:

```bash
make reproduce
```

or directly:

```bash
.venv/bin/python reproduce.py
```

The canonical pipeline is `reproduce.py`. It rebuilds clean liver/brain matrices from the raw files, selects the top five brain targets by variance inside each outer LOOCV training fold only, uses fold-wise log1p plus z-score normalization fitted only on each LOOCV training fold, trains the MLP under fixed repeated seeds, aggregates SHAP across folds and seeds, and writes all outputs under `reproducibility/`.

The fold-wise target-selection run was regenerated with `make reproduce` and `make baselines`; the 1,000-shuffle permutation null was regenerated with `.venv/bin/python baselines.py --permutations 1000`.

Current regenerated summary:

- Cohorts: `FL=3`, `HC=3`, `VC=3`
- Clean matrices: liver `9 x 717`, brain `9 x 637`
- Brain source: `GSE294046` brain miRNA columns
- Seeds: `101,202,303,404,505`
- Brain target selection: top `5` by variance inside each outer LOOCV training fold, using only the `8` training samples
- Same ordered target set across all folds: `false`
- Same unordered target set across all folds: `false`
- Fold-selected brain target union: `mmu-let-7a-5p`, `mmu-let-7b-5p`, `mmu-let-7c-5p`, `mmu-miR-124-3p`, `mmu-miR-181a-5p`, `mmu-miR-26a-5p`
- Full-cohort reference top five, not used for modeling: `mmu-miR-124-3p`, `mmu-let-7c-5p`, `mmu-miR-26a-5p`, `mmu-let-7a-5p`, `mmu-let-7b-5p`
- VMSE mean: `1.6489`
- VMSE standard deviation: `0.1396`
- VMSE range: `1.4527-1.8567`
- Top aggregated SHAP feature: `mmu-miR-17-5p`

Baseline comparison generated with:

```bash
make baselines
```

Current leakage-free LOOCV baseline results:

| Model | LOOCV VMSE | Relative improvement vs mean baseline |
| --- | ---: | ---: |
| Training-fold mean | `1.9148` | `0.0%` |
| Linear regression | `1.1600` | `39.4%` |
| Nested ridge regression | `1.8809` | `1.8%` |
| MLP, 5-seed mean | `1.6489` | `13.9%` |
| Permuted MLP median | `2.8626` | `-49.5%` |

The baseline script uses the same nine paired mice, six liver inputs, fold-selected brain targets, outer LOOCV folds, and fold-specific log1p/z-score preprocessing as the MLP. Ridge alpha is selected by nested leave-one-out cross-validation inside each outer training fold after that outer fold's brain target set has been selected from its eight training samples.

The current 1,000-shuffle subject-pairing permutation null was generated under the fold-wise target-selection pipeline with:

```bash
.venv/bin/python baselines.py --permutations 1000
```

The regenerated MLP subject-pairing permutation summary is:

- Observed MLP VMSE: `1.6489`
- Permutations: `1000`
- Count permuted VMSE less than or equal to observed VMSE: `74`
- Permutation p-value: `0.07492507492507493`
- Permuted VMSE median: `2.8626`
- Permuted VMSE 2.5-97.5 percentile range: `1.3862-5.1584`

Primary outputs:

- `reproducibility/metrics.json`
- `reproducibility/baseline_metrics.json`
- `reproducibility/tables/metrics_by_seed.csv`
- `reproducibility/tables/fold_metrics.csv`
- `reproducibility/tables/predictions.csv`
- `reproducibility/tables/fold_target_selection.csv`
- `reproducibility/tables/model_comparison.csv`
- `reproducibility/tables/baseline_model_summary.csv`
- `reproducibility/tables/baseline_fold_metrics.csv`
- `reproducibility/tables/baseline_predictions.csv`
- `reproducibility/tables/baseline_fold_target_selection.csv`
- `reproducibility/tables/ridge_alpha_selection.csv`
- `reproducibility/tables/observed_mlp_seed_vmse.csv`
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
