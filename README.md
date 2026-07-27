# OSD914 Analysis: Hepatic-Neuro Cross-Talk AI

Reproducibility package for the paper's liver-brain microRNA co-expression analysis in NASA OSDR Study OSD-914 / GLDS-736. The canonical pipeline rebuilds clean matrices from the raw gzipped inputs, runs leakage-free leave-one-out cross-validation (LOOCV), summarizes SHAP feature importance, compares deterministic baselines, and writes all paper-facing outputs under `reproducibility/`.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `reproduce.py` | Canonical MLP reproduction pipeline from raw data to metrics, tables, figures, and dashboard. |
| `baselines.py` | Mean, linear regression, nested ridge, and optional subject-pairing permutation comparisons. |
| `Makefile` | Short commands for the common reproduction workflows. |
| `pyproject.toml` | Python package metadata, dependencies, and console entry points. |
| `GSE294046_miRNA_complete_quantification_raw.tsv.gz` | Raw miRNA quantification input used for liver and matched brain miRNA matrices. |
| `GSE295428_series_matrix.txt.gz` | Raw GEO series matrix checked for predefined brain target genes. |
| `reproducibility/` | Current regenerated paper outputs. |
| `REPRODUCIBILITY.md` | Snapshot-style notes for the latest regenerated result set. |
| `ai.py` | Legacy exploratory dashboard. Do not use this as the paper reproduction entry point. |

## Fresh Setup

Use Python 3.9 or newer. The checked output snapshot was generated with Python 3.9.6 and the package versions recorded in `reproducibility/metrics.json`.

```bash
cd "NASA Research"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The source distribution includes the raw gzipped inputs listed above. If rebuilding from a stripped source tree, place both files in the repository root before running:

```text
GSE294046_miRNA_complete_quantification_raw.tsv.gz
GSE295428_series_matrix.txt.gz
```

Current raw-file SHA256 checksums:

| File | SHA256 |
| --- | --- |
| `GSE294046_miRNA_complete_quantification_raw.tsv.gz` | `053342fd376cde1ef7673f0581253c50fd8fe5524fcd8aa5a4935f1ebdd9c7fb` |
| `GSE295428_series_matrix.txt.gz` | `1344e2157b025afcb050b353621383e7cea7a89d2df770ce13d14aae84ef401d` |

To create a distributable source archive for review or release:

```bash
make package
```

This writes `dist/osd914_analysis-0.1.0.tar.gz` with the code, raw inputs, current `reproducibility/` outputs, manuscript artifacts, and figures listed in `MANIFEST.in`.

## Reproduce the Paper Outputs

Run a quick wiring check first. This skips SHAP and writes to a disposable smoke directory.

```bash
make smoke-test
```

Run the canonical MLP reproduction pipeline:

```bash
make reproduce
```

Equivalent console entry point after `pip install -e .`:

```bash
osd914-reproduce
```

Run deterministic baseline comparisons:

```bash
make baselines
```

Run the 1,000-shuffle subject-pairing permutation null. This is the slowest step.

```bash
make permutation-baselines
```

Equivalent baseline CLI:

```bash
osd914-baselines --permutations 1000
```

To rebuild only the clean liver and brain matrices used by exploratory scripts:

```bash
osd914-clean-data
```

## Expected Current Results

The pipeline rebuilds clean matrices with 9 matched mice, 717 filtered liver miRNAs, and 637 filtered brain miRNAs. The modeled liver inputs are:

```text
mmu-miR-122-5p
mmu-miR-17-5p
mmu-miR-18a-5p
mmu-miR-19a-5p
mmu-miR-20a-5p
mmu-miR-92a-1-5p
```

Brain targets are selected inside each outer LOOCV training fold only. The current fold-selected target union is:

```text
mmu-let-7a-5p
mmu-let-7b-5p
mmu-let-7c-5p
mmu-miR-124-3p
mmu-miR-181a-5p
mmu-miR-26a-5p
```

Headline MLP result across seeds `101,202,303,404,505`:

| Metric | Value |
| --- | ---: |
| VMSE mean | `1.6489` |
| VMSE standard deviation | `0.1396` |
| VMSE range | `1.4527-1.8567` |
| Top aggregated SHAP feature | `mmu-miR-17-5p` |

Baseline comparison:

| Model | LOOCV VMSE | Relative improvement vs mean baseline |
| --- | ---: | ---: |
| Training-fold mean | `1.9148` | `0.0%` |
| Linear regression | `1.1600` | `39.4%` |
| Nested ridge regression | `1.8809` | `1.8%` |
| MLP, 5-seed mean | `1.6489` | `13.9%` |
| Permuted MLP median | `2.8626` | `-49.5%` |

Permutation summary for the current 1,000-shuffle null:

| Metric | Value |
| --- | ---: |
| Observed MLP VMSE | `1.6489` |
| Permutations | `1000` |
| Permuted VMSE <= observed | `74` |
| Permutation p-value | `0.07492507492507493` |
| Permuted VMSE median | `2.8626` |
| 2.5-97.5 percentile range | `1.3862-5.1584` |

## Output Map

Primary regenerated outputs:

```text
reproducibility/metrics.json
reproducibility/baseline_metrics.json
reproducibility/tables/metrics_by_seed.csv
reproducibility/tables/fold_metrics.csv
reproducibility/tables/predictions.csv
reproducibility/tables/fold_target_selection.csv
reproducibility/tables/model_comparison.csv
reproducibility/tables/baseline_model_summary.csv
reproducibility/tables/baseline_fold_metrics.csv
reproducibility/tables/baseline_predictions.csv
reproducibility/tables/ridge_alpha_selection.csv
reproducibility/tables/shap_values.csv
reproducibility/tables/shap_importance_summary.csv
reproducibility/figures/*.png
reproducibility/figures/*.svg
reproducibility/dashboard.html
```

`reproduce.py` and `baselines.py` accept `--output-dir` if you want to write a separate run without overwriting the checked `reproducibility/` snapshot.

## Reproducibility Design

The canonical pipeline avoids leakage by fitting target selection and normalization inside each outer LOOCV training fold. For each held-out mouse, only the remaining 8 training samples are used to choose the top 5 brain targets by variance and to fit log1p plus z-score scalers. The MLP is retrained from scratch in every fold and seed.

The older `ai.py` dashboard is retained for exploratory inspection only. It uses global normalization and stochastic single-run outputs, so it should not be cited as the reproducibility pipeline.

## Archive and Citation

Zenodo archives are updated from tagged GitHub releases for this repository. The concept DOI is https://doi.org/10.5281/zenodo.21283917. Cite release-specific DOI records when exact frozen artifacts are required.

Project team:

- Principal Investigator: Pratheek Mukkavilli
- Technical Advisor: Xavier-Lewis Palmer, Ph.D.
- Biological Validation: Rosa Prahl, Michelle Medeiros
