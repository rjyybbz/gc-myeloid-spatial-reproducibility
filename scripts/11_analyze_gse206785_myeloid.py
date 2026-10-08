from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from project_setup import PROJECT_PATHS, main as setup_main


CANDIDATE_ANNOTATIONS = ["M08_Mono_INHBA", "M04_Macro_C3", "M07_Macro_APOE", "M11_Macro_PLA2G7"]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def extract_rank_genes(adata: ad.AnnData, groupby: str, n_top: int = 30) -> pd.DataFrame:
    rows = []
    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    for group in groups:
        for i in range(min(n_top, len(result["names"][group]))):
            rows.append(
                {
                    "group": group,
                    "rank": i + 1,
                    "gene": result["names"][group][i],
                    "logfoldchange": float(result["logfoldchanges"][group][i]),
                    "pvals_adj": float(result["pvals_adj"][group][i]),
                    "score": float(result["scores"][group][i]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    setup_main()
    input_path = PROJECT_PATHS["data_processed"] / "GSE206785" / "GSE206785_myeloid_logexpr.h5ad"
    adata = sc.read_h5ad(input_path)

    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack")
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=0.5)

    sc.tl.rank_genes_groups(adata, groupby="Annotation", method="wilcoxon")
    markers = extract_rank_genes(adata, groupby="Annotation", n_top=40)

    annotation_summary = pd.crosstab(adata.obs["Annotation"], adata.obs["Tissue"])
    annotation_summary["total"] = annotation_summary.sum(axis=1)
    annotation_summary["tumor_fraction"] = annotation_summary.get("Tumor", 0) / annotation_summary["total"]
    annotation_summary = annotation_summary.reset_index().sort_values("tumor_fraction", ascending=False)

    candidate_markers = markers[markers["group"].isin(CANDIDATE_ANNOTATIONS)].copy()
    candidate_markers = candidate_markers[(candidate_markers["logfoldchange"] > 0.5) & (candidate_markers["pvals_adj"] < 1e-5)]
    program_counts = candidate_markers.groupby("gene")["group"].nunique().sort_values(ascending=False)
    program_genes = program_counts[program_counts >= 2].index.tolist()

    program_table = pd.DataFrame(
        {
            "gene": program_genes,
            "supporting_candidate_states": [int(program_counts[g]) for g in program_genes],
        }
    )

    if program_genes:
        avg_expr = pd.DataFrame(
            adata[:, program_genes].to_df().groupby(adata.obs["Annotation"]).mean().T.reset_index()
        ).rename(columns={"index": "gene"})
    else:
        avg_expr = pd.DataFrame(columns=["gene"])

    embedding = pd.DataFrame(
        {
            "cell_id": adata_hvg.obs_names,
            "umap_1": adata_hvg.obsm["X_umap"][:, 0],
            "umap_2": adata_hvg.obsm["X_umap"][:, 1],
            "Annotation": adata_hvg.obs["Annotation"].values,
            "Tissue": adata_hvg.obs["Tissue"].values,
            "Patient": adata_hvg.obs["Patient"].values,
            "leiden": adata_hvg.obs["leiden"].values,
        }
    )

    outdir = PROJECT_PATHS["results"] / "myeloid_discovery"
    write_csv(outdir / "GSE206785_annotation_tissue_summary.csv", annotation_summary)
    write_csv(outdir / "GSE206785_annotation_markers_top40.csv", markers)
    write_csv(outdir / "GSE206785_candidate_program_genes.csv", program_table)
    write_csv(outdir / "GSE206785_candidate_program_average_expression.csv", avg_expr)
    write_csv(outdir / "GSE206785_myeloid_umap.csv", embedding)

    summary = {
        "dataset_id": "GSE206785",
        "myeloid_cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "candidate_annotations": CANDIDATE_ANNOTATIONS,
        "program_gene_count": len(program_genes),
        "top_program_genes": program_genes[:30],
    }
    (outdir / "GSE206785_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
