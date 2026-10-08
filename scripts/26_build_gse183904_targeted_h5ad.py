from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from project_setup import PROJECT_PATHS, main as setup_main


SUPP_DIR = PROJECT_PATHS["data_raw"] / "GSE183904" / "supplementary"
MANIFEST_PATH = PROJECT_PATHS["metadata"] / "sample_manifests" / "GSE183904_sample_manifest.csv"

MARKER_MODULES = {
    "myeloid": ["LST1", "TYROBP", "FCER1G", "AIF1", "CTSS", "SAT1", "C1QA", "C1QB", "FCN1", "VCAN", "S100A8", "S100A9"],
    "tcell": ["CD3D", "CD3E", "TRBC1", "IL7R", "LTB", "NKG7", "TRAC"],
    "bcell": ["MS4A1", "CD79A", "CD74", "HLA-DRA", "CD79B"],
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "KRT17"],
    "stromal": ["COL1A1", "COL1A2", "DCN", "LUM", "TAGLN", "RGS5", "ACTA2"],
}

CURATED_AXIS_GENES = [
    "INHBA",
    "PLA2G7",
    "APOE",
    "APOC1",
    "C1QA",
    "C1QB",
    "C1QC",
    "CD68",
    "GPNMB",
    "TREM2",
    "LGALS3",
    "LGALS1",
    "MS4A7",
    "FABP5",
    "GRN",
    "CTSB",
    "CTSD",
    "CTSL",
    "PSAP",
    "NPC2",
    "LIPA",
    "S100A8",
    "S100A9",
    "FCN1",
    "VCAN",
    "IL1B",
    "THBS1",
    "EREG",
    "NAMPT",
    "TIMP1",
]

RECEPTOR_SWITCH_GENES = [
    "CD74",
    "CSF1R",
    "TGFBR1",
    "TGFBR2",
    "CCR2",
    "CXCR4",
    "IL1R1",
    "HAVCR2",
    "CD44",
]


def load_gene_panel() -> list[str]:
    genes: set[str] = set()
    for gene_list in MARKER_MODULES.values():
        genes.update(gene_list)
    genes.update(CURATED_AXIS_GENES)
    genes.update(RECEPTOR_SWITCH_GENES)

    program_genes = pd.read_csv(
        PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"
    )["gene"].dropna().astype(str)
    genes.update(program_genes.tolist())

    trajectory_summary = json.loads(
        (PROJECT_PATHS["results"] / "mechanism_trajectory" / "GSE206785_trajectory_summary.json").read_text(
            encoding="utf-8"
        )
    )
    genes.update(trajectory_summary["top_late_up_genes"][:30])
    genes.update(trajectory_summary["top_early_up_genes"][:20])
    genes.update([row["gene"] for row in trajectory_summary["top_candidate_regulators"][:20]])
    return sorted(gene for gene in genes if gene)


def parse_sample_label(raw: str) -> tuple[str, str]:
    raw = str(raw).lower()
    if "peritonium" in raw or "peritoneum" in raw:
        site = "peritoneum"
    else:
        site = "stomach"

    if "normal" in raw:
        status = "normal"
    elif "tumor" in raw:
        status = "tumor"
    else:
        status = "other"
    return site, status


def load_one_sample(path: Path, manifest_row: pd.Series, target_genes: list[str]) -> ad.AnnData:
    target_set = set(target_genes)
    selected_genes: list[str] = []
    selected_rows: list[np.ndarray] = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        cell_barcodes = [cell.strip('"') for cell in header[1:]]
        for row in reader:
            if not row:
                continue
            gene = row[0].strip('"')
            if gene not in target_set:
                continue
            values = np.asarray(row[1:], dtype=np.float32)
            selected_genes.append(gene)
            selected_rows.append(values)

    matrix = np.vstack(selected_rows).T if selected_rows else np.zeros((len(cell_barcodes), 0), dtype=np.float32)
    obs = pd.DataFrame(index=[f"{manifest_row['geo_accession']}:{barcode}" for barcode in cell_barcodes])
    obs["cell_barcode"] = cell_barcodes
    obs["geo_accession"] = str(manifest_row["geo_accession"])
    obs["title"] = str(manifest_row["title"])
    obs["characteristics_ch1"] = str(manifest_row["characteristics_ch1"])
    obs["source_name_ch1"] = str(manifest_row["source_name_ch1"])
    obs["sample_id"] = str(manifest_row["title"]).split(":", 1)[0]
    site, status = parse_sample_label(manifest_row["characteristics_ch1"])
    obs["site_group"] = site
    obs["disease_group"] = status
    obs["tissue_group"] = f"{site}_{status}"

    var = pd.DataFrame(index=pd.Index(selected_genes, name="gene_symbol"))
    adata = ad.AnnData(X=sparse.csr_matrix(matrix), obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def main() -> None:
    setup_main()
    manifest = pd.read_csv(MANIFEST_PATH).sort_values("geo_accession").reset_index(drop=True)
    target_genes = load_gene_panel()
    sample_adatas = []
    sample_rows = []

    for _, row in manifest.iterrows():
        path = Path(row["supplementary_file_1"]).name
        file_path = SUPP_DIR / path
        adata = load_one_sample(file_path, row, target_genes)
        sample_adatas.append(adata)
        sample_rows.append(
            {
                "geo_accession": str(row["geo_accession"]),
                "file_name": path,
                "sample_id": str(row["title"]).split(":", 1)[0],
                "n_cells": int(adata.n_obs),
                "n_target_genes_detected": int(adata.n_vars),
                "tissue_group": str(adata.obs["tissue_group"].iloc[0]),
            }
        )

    combined = ad.concat(sample_adatas, join="outer", merge="same", fill_value=0)
    combined.X = sparse.csr_matrix(combined.X)
    combined.uns["dataset_id"] = "GSE183904"
    combined.uns["target_gene_panel"] = target_genes

    outdir = PROJECT_PATHS["data_processed"] / "GSE183904"
    outdir.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(outdir / "GSE183904_targeted.h5ad")

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(outdir / "GSE183904_targeted_sample_summary.csv", index=False)

    summary = {
        "dataset_id": "GSE183904",
        "n_samples": int(len(sample_rows)),
        "shape": [int(combined.n_obs), int(combined.n_vars)],
        "target_gene_count": int(len(target_genes)),
        "tissue_group_counts": sample_df["tissue_group"].value_counts().to_dict(),
    }
    (outdir / "GSE183904_targeted_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
