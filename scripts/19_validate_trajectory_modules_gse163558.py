from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import scanpy as sc

from project_setup import PROJECT_PATHS, main as setup_main


MARKER_MODULES = {
    "myeloid_score": ["LST1", "TYROBP", "FCER1G", "AIF1", "CTSS", "SAT1", "C1QA", "C1QB"],
    "tcell_score": ["CD3D", "CD3E", "TRBC1", "IL7R", "LTB"],
    "bcell_score": ["MS4A1", "CD79A", "CD74", "HLA-DRA"],
    "epithelial_score": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1"],
    "stromal_score": ["COL1A1", "COL1A2", "DCN", "LUM", "TAGLN"],
}


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def map_tissue_label(raw: str) -> str:
    raw = str(raw).lower()
    if "primary gastric cancer" in raw:
        return "primary_tumor"
    if "adjacent non-tumoral" in raw:
        return "adjacent_normal"
    if "lymph node" in raw:
        return "lymph_node_met"
    if "ovary" in raw:
        return "ovary_met"
    if "peritoneum" in raw:
        return "peritoneal_met"
    if "liver" in raw:
        return "liver_met"
    return "other"


def main() -> None:
    setup_main()

    dynamic_genes = pd.read_csv(PROJECT_PATHS["results"] / "mechanism_trajectory" / "GSE206785_late_vs_early_dynamic_genes.csv")
    program_genes = pd.read_csv(PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv")["gene"].tolist()
    late_axis_genes = dynamic_genes.loc[dynamic_genes["direction"] == "late_up", "gene"].head(50).tolist()
    early_axis_genes = dynamic_genes.loc[dynamic_genes["direction"] == "early_up", "gene"].head(30).tolist()

    adata = sc.read_h5ad(PROJECT_PATHS["data_processed"] / "GSE163558" / "GSE163558_full.h5ad")
    adata.var_names_make_unique()
    adata.obs["tissue_group"] = adata.obs["source_name_ch1"].map(map_tissue_label)

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)

    module_defs = {
        **MARKER_MODULES,
        "late_axis_score": [g for g in late_axis_genes if g in adata.var_names],
        "early_axis_score": [g for g in early_axis_genes if g in adata.var_names],
        "suppressive_program_score": [g for g in program_genes if g in adata.var_names],
    }
    for score_name, genes in module_defs.items():
        sc.tl.score_genes(adata, gene_list=genes, score_name=score_name)

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack")
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=0.6)

    score_cols = list(module_defs.keys())
    for col in score_cols:
        adata_hvg.obs[col] = adata.obs.loc[adata_hvg.obs_names, col].values

    cluster_scores = adata_hvg.obs.groupby("leiden")[list(MARKER_MODULES.keys())].mean().reset_index()
    cluster_scores["assigned_lineage"] = cluster_scores[list(MARKER_MODULES.keys())].idxmax(axis=1).str.replace("_score", "", regex=False)
    myeloid_clusters = cluster_scores.loc[cluster_scores["assigned_lineage"] == "myeloid", "leiden"].tolist()
    adata_hvg.obs["is_myeloid_cluster"] = adata_hvg.obs["leiden"].isin(myeloid_clusters)

    myeloid_cells = adata_hvg.obs.loc[adata_hvg.obs["is_myeloid_cluster"]].copy()
    myeloid_cells["late_minus_early"] = myeloid_cells["late_axis_score"] - myeloid_cells["early_axis_score"]

    tissue_summary = (
        myeloid_cells.groupby("tissue_group")[["late_axis_score", "early_axis_score", "late_minus_early", "suppressive_program_score"]]
        .agg(["count", "mean", "median"])
    )
    tissue_summary.columns = ["_".join(col).strip("_") for col in tissue_summary.columns]
    tissue_summary = tissue_summary.reset_index().sort_values("late_axis_score_mean", ascending=False)

    cluster_tissue = (
        adata_hvg.obs.groupby(["leiden", "tissue_group"])[["late_axis_score", "early_axis_score", "suppressive_program_score"]]
        .mean()
        .reset_index()
        .sort_values(["leiden", "late_axis_score"], ascending=[True, False])
    )

    embedding = pd.DataFrame(
        {
            "cell_id": adata_hvg.obs_names,
            "umap_1": adata_hvg.obsm["X_umap"][:, 0],
            "umap_2": adata_hvg.obsm["X_umap"][:, 1],
            "leiden": adata_hvg.obs["leiden"].values,
            "tissue_group": adata_hvg.obs["tissue_group"].values,
            "late_axis_score": adata_hvg.obs["late_axis_score"].values,
            "early_axis_score": adata_hvg.obs["early_axis_score"].values,
            "suppressive_program_score": adata_hvg.obs["suppressive_program_score"].values,
            "is_myeloid_cluster": adata_hvg.obs["is_myeloid_cluster"].values,
        }
    )

    outdir = PROJECT_PATHS["results"] / "mechanism_trajectory"
    write_csv(outdir / "GSE163558_trajectory_module_tissue_summary.csv", tissue_summary)
    write_csv(outdir / "GSE163558_trajectory_module_cluster_tissue.csv", cluster_tissue)
    write_csv(outdir / "GSE163558_trajectory_module_umap.csv", embedding)

    summary = {
        "dataset_id": "GSE163558",
        "myeloid_clusters": myeloid_clusters,
        "late_axis_genes_available": len(module_defs["late_axis_score"]),
        "early_axis_genes_available": len(module_defs["early_axis_score"]),
        "suppressive_program_genes_available": len(module_defs["suppressive_program_score"]),
        "top_tissue_groups_by_late_axis_mean": tissue_summary[["tissue_group", "late_axis_score_mean", "suppressive_program_score_mean"]].to_dict(orient="records"),
    }
    (outdir / "GSE163558_trajectory_module_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
