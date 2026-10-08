# Reproducibility Guide

## Environment

Install the packages listed in `requirements.txt`. The revision-audit environment used Python 3.8.20 with pandas, numpy, scipy, statsmodels, scikit-learn, scanpy, anndata, matplotlib, and seaborn. Newer versions may work but can change plotting defaults and numerical details.

## Folder initialization

```bash
python scripts/00_project_setup.py
```

This creates standard folders such as `data_raw/`, `data_processed/`, `results/`, `figures/`, `metadata/`, and `logs/`.

## Core analysis order

The main analysis scripts are numbered approximately in execution order:

1. Dataset inventory and public metadata: `01` to `08`.
2. GSE206785 discovery and GSE163558 validation: `09` to `13`.
3. Spatial and bulk transfer analyses: `14` to `20`.
4. GSE183904 atlas and mechanism/peritoneal analyses: `26`, `27`, `31`, `34`.
5. Bulk signature benchmarks and dual-axis ecology: `28`, `29`, `42`.
6. Primary and external spatial/CosMx analyses: `30`, `35`, `37` to `41`.
7. Manuscript evidence summaries and figures: `36`, `50`.

Scripts download or read public datasets into local `data_raw/` and `data_processed/` folders. Large matrices are not included in this repository.

## Round-2 reviewer additions

The following scripts regenerate the added machine-readable supplementary tables from the included compact derived summaries:

```bash
python revision_scripts/71_round2_sample_level_peritoneal_tables.py
python revision_scripts/72_round2_build_supplementary_tables.py
```

Outputs are written to `04_supplementary_tables/` by default.

## Biological replicate policy

For reviewer-facing inferential summaries, patient- or sample-level summaries are the primary biological replicate layer. Cell-level and spot-level displays are descriptive unless explicitly aggregated or summarized at the patient/sample level.

## Known limitations

- The repository does not include raw public expression matrices or large processed objects.
- Some public datasets provide processed matrices from original authors rather than raw-data reconstruction.
- Results that depend on exact package versions, nearest-neighbor graph construction, or plotting libraries may show small numerical or visual differences.
- No script in this repository establishes causal inference, perturbation evidence, histology-confirmed TLS, or independent clinical prognostic utility.
