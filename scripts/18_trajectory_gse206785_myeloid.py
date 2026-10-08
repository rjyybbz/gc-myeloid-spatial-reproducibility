from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

from project_setup import PROJECT_PATHS, main as setup_main


LINEAGE_ANNOTATIONS = [
    "M02_Mono_CD14",
    "M05_Mono_VCAN",
    "M08_Mono_INHBA",
    "M13_Mono_VCAN",
    "M04_Macro_C3",
    "M06_Macro_SLC40A1",
    "M07_Macro_APOE",
    "M11_Macro_PLA2G7",
]

ROOT_ANNOTATIONS = [
    "M02_Mono_CD14",
    "M05_Mono_VCAN",
    "M13_Mono_VCAN",
]

TERMINAL_SUPPRESSIVE_ANNOTATIONS = [
    "M04_Macro_C3",
    "M07_Macro_APOE",
    "M11_Macro_PLA2G7",
]

CURATED_REGULATORS = [
    "SPI1",
    "CEBPA",
    "CEBPB",
    "CEBPD",
    "IRF1",
    "IRF2",
    "IRF5",
    "IRF7",
    "IRF8",
    "STAT1",
    "STAT2",
    "STAT3",
    "STAT5A",
    "STAT5B",
    "STAT6",
    "NFKB1",
    "NFKB2",
    "RELA",
    "RELB",
    "REL",
    "JUN",
    "JUND",
    "FOS",
    "FOSB",
    "ATF3",
    "BHLHE40",
    "HIF1A",
    "EPAS1",
    "MAF",
    "MAFB",
    "KLF4",
    "KLF6",
    "PPARG",
    "NR1H3",
    "XBP1",
    "SREBF1",
    "SREBF2",
    "EGR1",
    "EGR2",
    "NR4A1",
    "NR4A2",
    "NR4A3",
    "BACH1",
    "BACH2",
    "TFEB",
    "NFE2L2",
    "RUNX1",
    "RUNX3",
    "BATF",
    "BATF3",
]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_program_genes() -> list[str]:
    path = PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"
    return pd.read_csv(path)["gene"].dropna().astype(str).tolist()


def extract_rank_genes(adata: ad.AnnData, key: str, group: str, n_top: int = 200) -> pd.DataFrame:
    rows = []
    result = adata.uns[key]
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


def choose_root_index(adata: ad.AnnData) -> int:
    root_mask = adata.obs["Annotation"].isin(ROOT_ANNOTATIONS) & adata.obs["Tissue"].eq("Normal")
    if root_mask.sum() == 0:
        root_mask = adata.obs["Annotation"].isin(ROOT_ANNOTATIONS)
    root_scores = adata.obs.loc[root_mask, "suppressive_program_score"]
    root_cell_name = root_scores.idxmin()
    return int(np.where(adata.obs_names == root_cell_name)[0][0])


def main() -> None:
    setup_main()
    input_path = PROJECT_PATHS["data_processed"] / "GSE206785" / "GSE206785_myeloid_logexpr.h5ad"
    adata = sc.read_h5ad(input_path)
    program_genes = [g for g in read_program_genes() if g in adata.var_names]

    adata = adata[adata.obs["Annotation"].isin(LINEAGE_ANNOTATIONS)].copy()
    sc.tl.score_genes(adata, gene_list=program_genes, score_name="suppressive_program_score")

    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2500)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver="arpack")
    sc.pp.neighbors(adata_hvg, n_neighbors=20, n_pcs=30)
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=0.4)
    sc.tl.diffmap(adata_hvg)
    adata_hvg.obs["suppressive_program_score"] = adata.obs.loc[adata_hvg.obs_names, "suppressive_program_score"].values
    adata_hvg.uns["iroot"] = choose_root_index(adata_hvg)
    sc.tl.dpt(adata_hvg, n_dcs=15)
    sc.tl.paga(adata_hvg, groups="Annotation")

    # Directed projection from monocyte-like start states to suppressive TAM-like terminal states.
    pca = adata_hvg.obsm["X_pca"][:, :20]
    root_mask = adata_hvg.obs["Annotation"].isin(ROOT_ANNOTATIONS).values
    terminal_mask = adata_hvg.obs["Annotation"].isin(TERMINAL_SUPPRESSIVE_ANNOTATIONS).values
    root_centroid = pca[root_mask].mean(axis=0)
    terminal_centroid = pca[terminal_mask].mean(axis=0)
    axis_vector = terminal_centroid - root_centroid
    axis_vector = axis_vector / np.linalg.norm(axis_vector)
    suppressive_axis = (pca - root_centroid) @ axis_vector
    suppressive_axis = (suppressive_axis - suppressive_axis.min()) / (suppressive_axis.max() - suppressive_axis.min())
    adata_hvg.obs["suppressive_axis_score"] = suppressive_axis

    adata.obs = adata.obs.loc[adata_hvg.obs_names].copy()
    adata.obs["dpt_pseudotime"] = adata_hvg.obs["dpt_pseudotime"].values
    adata.obs["leiden"] = adata_hvg.obs["leiden"].values
    adata.obs["suppressive_axis_score"] = adata_hvg.obs["suppressive_axis_score"].values
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"].copy()
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"].copy()
    adata.obsm["X_diffmap"] = adata_hvg.obsm["X_diffmap"].copy()
    adata.uns["trajectory_root_index"] = int(adata_hvg.uns["iroot"])
    adata.uns["trajectory_root_cell"] = str(adata.obs_names[adata_hvg.uns["iroot"]])
    adata.uns["paga"] = adata_hvg.uns["paga"]
    adata.obs["trajectory_state"] = pd.qcut(
        adata.obs["suppressive_axis_score"],
        q=3,
        labels=["early", "intermediate", "late"],
        duplicates="drop",
    ).astype(str)
    adata.obs["trajectory_decile"] = pd.qcut(
        adata.obs["suppressive_axis_score"],
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
        duplicates="drop",
    ).astype(str)

    sc.tl.rank_genes_groups(
        adata,
        groupby="trajectory_state",
        groups=["late"],
        reference="early",
        method="wilcoxon",
        key_added="rank_genes_trajectory_late_vs_early",
    )
    sc.tl.rank_genes_groups(
        adata,
        groupby="trajectory_state",
        groups=["early"],
        reference="late",
        method="wilcoxon",
        key_added="rank_genes_trajectory_early_vs_late",
    )
    dynamic_genes = extract_rank_genes(adata, "rank_genes_trajectory_late_vs_early", "late", n_top=300)
    early_genes = extract_rank_genes(adata, "rank_genes_trajectory_early_vs_late", "early", n_top=150)
    early_genes["direction"] = "early_up"
    dynamic_genes["direction"] = "late_up"
    dynamic_genes = pd.concat([dynamic_genes, early_genes], ignore_index=True)

    annotation_summary = (
        adata.obs.groupby("Annotation")
        .agg(
            n_cells=("Annotation", "size"),
            n_patients=("Patient", "nunique"),
            tumor_fraction=("Tissue", lambda s: float((s == "Tumor").mean())),
            mean_axis_score=("suppressive_axis_score", "mean"),
            median_axis_score=("suppressive_axis_score", "median"),
            mean_dpt_pseudotime=("dpt_pseudotime", "mean"),
            median_dpt_pseudotime=("dpt_pseudotime", "median"),
            mean_program_score=("suppressive_program_score", "mean"),
            median_program_score=("suppressive_program_score", "median"),
        )
        .reset_index()
        .sort_values("median_axis_score")
    )

    trajectory_state_summary = (
        adata.obs.groupby("trajectory_state")
        .agg(
            n_cells=("trajectory_state", "size"),
            tumor_fraction=("Tissue", lambda s: float((s == "Tumor").mean())),
            mean_program_score=("suppressive_program_score", "mean"),
            median_program_score=("suppressive_program_score", "median"),
            mean_axis_score=("suppressive_axis_score", "mean"),
            mean_dpt_pseudotime=("dpt_pseudotime", "mean"),
        )
        .reset_index()
    )

    decile_summary = (
        adata.obs.groupby("trajectory_decile")
        .agg(
            n_cells=("trajectory_decile", "size"),
            tumor_fraction=("Tissue", lambda s: float((s == "Tumor").mean())),
            mean_program_score=("suppressive_program_score", "mean"),
            median_axis_score=("suppressive_axis_score", "median"),
            median_dpt_pseudotime=("dpt_pseudotime", "median"),
        )
        .reset_index()
    )

    paga_conn = adata.uns["paga"]["connectivities"].toarray()
    paga_names = adata.obs["Annotation"].astype("category").cat.categories.tolist()
    paga_rows = []
    for i, source in enumerate(paga_names):
        for j, target in enumerate(paga_names):
            if j <= i:
                continue
            paga_rows.append(
                {
                    "source": source,
                    "target": target,
                    "connectivity": float(paga_conn[i, j]),
                }
            )
    paga_df = pd.DataFrame(paga_rows).sort_values("connectivity", ascending=False)

    available_regulators = [g for g in CURATED_REGULATORS if g in adata.var_names]
    regulator_rows = []
    if available_regulators:
        reg_expr = adata[:, available_regulators].to_df()
        early_mask = adata.obs["trajectory_state"].eq("early").values
        late_mask = adata.obs["trajectory_state"].eq("late").values
        for gene in available_regulators:
            rho_axis, p_axis = spearmanr(adata.obs["suppressive_axis_score"], reg_expr[gene], nan_policy="omit")
            rho_score, p_score = spearmanr(adata.obs["suppressive_program_score"], reg_expr[gene], nan_policy="omit")
            early_mean = float(reg_expr.loc[early_mask, gene].mean())
            late_mean = float(reg_expr.loc[late_mask, gene].mean())
            regulator_rows.append(
                {
                    "gene": gene,
                    "rho_with_axis_score": float(rho_axis),
                    "axis_score_p_value": float(p_axis),
                    "rho_with_program_score": float(rho_score),
                    "program_score_p_value": float(p_score),
                    "early_mean_expr": early_mean,
                    "late_mean_expr": late_mean,
                    "late_minus_early": late_mean - early_mean,
                }
            )
    regulators_df = pd.DataFrame(regulator_rows).sort_values(
        ["rho_with_axis_score", "late_minus_early"], ascending=False
    )

    embedding = pd.DataFrame(
        {
            "cell_id": adata.obs_names,
            "umap_1": adata.obsm["X_umap"][:, 0],
            "umap_2": adata.obsm["X_umap"][:, 1],
            "Annotation": adata.obs["Annotation"].values,
            "Tissue": adata.obs["Tissue"].values,
            "Patient": adata.obs["Patient"].values,
            "leiden": adata.obs["leiden"].values,
            "suppressive_program_score": adata.obs["suppressive_program_score"].values,
            "suppressive_axis_score": adata.obs["suppressive_axis_score"].values,
            "dpt_pseudotime": adata.obs["dpt_pseudotime"].values,
            "trajectory_state": adata.obs["trajectory_state"].values,
            "trajectory_decile": adata.obs["trajectory_decile"].values,
        }
    )

    trajectory_h5ad_path = PROJECT_PATHS["data_processed"] / "GSE206785" / "GSE206785_myeloid_trajectory.h5ad"
    adata.write_h5ad(trajectory_h5ad_path)

    outdir = PROJECT_PATHS["results"] / "mechanism_trajectory"
    write_csv(outdir / "GSE206785_trajectory_embedding.csv", embedding)
    write_csv(outdir / "GSE206785_annotation_trajectory_summary.csv", annotation_summary)
    write_csv(outdir / "GSE206785_trajectory_state_summary.csv", trajectory_state_summary)
    write_csv(outdir / "GSE206785_trajectory_decile_summary.csv", decile_summary)
    write_csv(outdir / "GSE206785_paga_annotation_connectivity.csv", paga_df)
    write_csv(outdir / "GSE206785_late_vs_early_dynamic_genes.csv", dynamic_genes)
    write_csv(outdir / "GSE206785_curated_regulator_dynamics.csv", regulators_df)

    summary = {
        "dataset_id": "GSE206785",
        "lineage_cell_count": int(adata.n_obs),
        "lineage_annotations": LINEAGE_ANNOTATIONS,
        "root_annotations": ROOT_ANNOTATIONS,
        "terminal_suppressive_annotations": TERMINAL_SUPPRESSIVE_ANNOTATIONS,
        "root_cell": adata.uns["trajectory_root_cell"],
        "program_score_vs_axis_spearman": float(
            spearmanr(adata.obs["suppressive_axis_score"], adata.obs["suppressive_program_score"], nan_policy="omit").statistic
        ),
        "program_score_vs_dpt_spearman": float(
            spearmanr(adata.obs["dpt_pseudotime"], adata.obs["suppressive_program_score"], nan_policy="omit").statistic
        ),
        "annotation_order_by_median_axis_score": annotation_summary[
            ["Annotation", "median_axis_score", "median_dpt_pseudotime", "median_program_score", "tumor_fraction"]
        ].to_dict(orient="records"),
        "top_late_up_genes": dynamic_genes.loc[dynamic_genes["direction"] == "late_up", "gene"].head(30).tolist(),
        "top_early_up_genes": dynamic_genes.loc[dynamic_genes["direction"] == "early_up", "gene"].head(20).tolist(),
        "top_candidate_regulators": regulators_df.head(20).to_dict(orient="records"),
        "trajectory_h5ad": str(trajectory_h5ad_path),
    }
    (outdir / "GSE206785_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
