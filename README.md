# CAF-associated myeloid suppressive-like ecology in gastric cancer

This repository contains analysis scripts and machine-readable reproducibility materials for the manuscript:

**A Multi-Cohort Computational Analysis Links Cancer-Associated Fibroblast-Related Signals to a Putative Myeloid Suppressive-Like Ecology in Gastric Cancer**

The code supports a computational, observational, and hypothesis-generating multi-cohort analysis of gastric cancer single-cell RNA-seq, spatial transcriptomic, CosMx spatial molecular imaging, and bulk transcriptomic cohorts.

## Repository contents

- `scripts/`: core analysis scripts used for dataset inventory, single-cell analysis, spatial analysis, bulk transfer, crosstalk summaries, and manuscript figure generation.
- `revision_scripts/`: round-2 reviewer-response reproducibility scripts for sample-level peritoneal summaries and supplementary table generation.
- `supplementary_tables/`: machine-readable supplementary tables added during peer review.
- `metadata/`: cohort manifests and analysis-planning metadata.
- `results/`: small derived summary files required to regenerate the round-2 supplementary tables. Raw expression matrices and large processed objects are not included.
- `results_summary/`: compact manifest summaries.
- `requirements.txt`: Python package requirements used by the analysis environment.

## Data availability

The repository does not redistribute raw expression matrices. Public data should be downloaded from the original sources:

- GEO: `GSE206785`
- GEO: `GSE163558`
- GEO: `GSE183904`
- GEO: `GSE270678`
- GEO: `GSE251950`
- GEO: `GSE308624`
- GEO: `GSE84437`
- TCGA/GDC: `TCGA-STAD`

See `DATA_AVAILABILITY.md` for details.

## Quick start

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Initialize the folder structure:

```bash
python scripts/00_project_setup.py
```

Regenerate the round-2 supplementary tables from the included small derived summaries:

```bash
python revision_scripts/71_round2_sample_level_peritoneal_tables.py
python revision_scripts/72_round2_build_supplementary_tables.py
```

## Reproducibility notes

The complete end-to-end analysis requires downloading public GEO/TCGA input files and may require substantial memory for single-cell and spatial objects. The included `supplementary_tables/` and `results/` folders provide the small derived files needed to audit the second-round reviewer additions without redistributing large public matrices.

The scripts are written for transparent auditability rather than as a packaged Python library. Most scripts read and write relative to the repository root through `scripts/project_setup.py`.

## Citation

If using this code before publication, cite the manuscript title and the public data sources above. A formal citation can be added after publication.

## License

No open-source license has been selected yet. See `LICENSE_PENDING_AUTHOR_CONFIRMATION.txt`. The authors should choose and add a license before final public release if reuse permissions are intended.
