# Data Availability

This repository contains code and small derived summary files only. Raw public expression matrices, large processed single-cell objects, spatial objects, and TCGA/GDC files are not redistributed.

## Public datasets

| Accession | Source | Role in analysis |
| --- | --- | --- |
| `GSE206785` | GEO | Primary single-cell discovery cohort for myeloid state axis, CAF association, and curated ligand-receptor hypotheses |
| `GSE163558` | GEO | External single-cell validation across primary and metastatic gastric cancer sites |
| `GSE183904` | GEO | External single-cell atlas and exploratory sample-level peritoneal analysis |
| `GSE270678` | GEO | Primary spatial transcriptomic niche analysis |
| `GSE251950` | GEO | External spatial support cohort |
| `GSE308624` | GEO | CosMx spatial molecular imaging support cohort |
| `TCGA-STAD` | GDC/TCGA | Bulk RNA-seq translational context and clinical-covariate-adjusted survival analysis |
| `GSE84437` | GEO | External bulk microarray translational context |

## Included derived files

The `results/` directory contains compact derived CSV/JSON summaries needed to regenerate the peer-review supplementary tables. These files are derived from public datasets and are included for auditability.

## Excluded files

The following are intentionally excluded:

- Raw FASTQ files, count matrices, H5/H5AD files, RDS objects, and Visium/CosMx large matrices.
- Manuscript submission documents and reviewer correspondence.
- Local workstation paths, credentials, tokens, and journal-portal records.

Users should download public source data from GEO, GDC/TCGA, or the original publications before running the full pipeline.
