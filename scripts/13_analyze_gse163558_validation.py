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
    discovery_program = pd.read_csv(PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv")
    program_genes = discovery_program["gene"].tolist()

    adata = sc.read_h5ad(PROJECT_PATHS["data_processed"] / "GSE163558" / "GSE163558_full.h5ad")
    adata.var_names_make_unique()
    adata.obs["tissue_group"] = adata.obs["source_name_ch1"].map(map_tissue_label)

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)
    for score_name, genes in MARKER_MODULES.items():
        valid = [g for g in genes if g in adata.var_names]
        sc.tl.score_genes(adata, gene_list=valid, score_name=score_name)
    valid_program = [g for g in program_genes if g in adata.var_names]
    sc.tl.score_genes(adata, gene_list=valid_program, score_name="suppressive_program_score")

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack")
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=0.6)

    score_cols = list(MARKER_MODULES.keys()) + ["suppressive_program_score"]
    cluster_scores = adata_hvg.obs.groupby("leiden")[score_cols].mean().reset_index()
    cluster_scores["assigned_lineage"] = cluster_scores[[c for c in MARKER_MODULES.keys()]].idxmax(axis=1).str.replace("_score", "", regex=False)
    myeloid_clusters = cluster_scores.loc[cluster_scores["assigned_lineage"] == "myeloid", "leiden"].tolist()
    adata_hvg.obs["is_myeloid_cluster"] = adata_hvg.obs["leiden"].isin(myeloid_clusters)

    myeloid_cells = adata_hvg.obs.loc[adata_hvg.obs["is_myeloid_cluster"]].copy()
    tissue_summary = (
        myeloid_cells.groupby("tissue_group")["suppressive_program_score"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    cluster_tissue = (
        adata_hvg.obs.groupby(["leiden", "tissue_group"])["suppressive_program_score"]
        .agg(["count", "mean"])
        .reset_index()
        .sort_values(["leiden", "mean"], ascending=[True, False])
    )
    embedding = pd.DataFrame(
        {
            "cell_id": adata_hvg.obs_names,
            "umap_1": adata_hvg.obsm["X_umap"][:, 0],
            "umap_2": adata_hvg.obsm["X_umap"][:, 1],
            "leiden": adata_hvg.obs["leiden"].values,
            "tissue_group": adata_hvg.obs["tissue_group"].values,
            "myeloid_score": adata_hvg.obs["myeloid_score"].values,
            "suppressive_program_score": adata_hvg.obs["suppressive_program_score"].values,
            "is_myeloid_cluster": adata_hvg.obs["is_myeloid_cluster"].values,
        }
    )

    outdir = PROJECT_PATHS["results"] / "myeloid_validation"
    write_csv(outdir / "GSE163558_cluster_scores.csv", cluster_scores)
    write_csv(outdir / "GSE163558_myeloid_tissue_summary.csv", tissue_summary)
    write_csv(outdir / "GSE163558_cluster_tissue_program.csv", cluster_tissue)
    write_csv(outdir / "GSE163558_umap_scores.csv", embedding)

    summary = {
        "dataset_id": "GSE163558",
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "program_gene_count_available": len(valid_program),
        "myeloid_clusters": myeloid_clusters,
        "top_tissue_groups_by_program_mean": tissue_summary.head(10).to_dict(orient="records"),
    }
    (outdir / "GSE163558_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
