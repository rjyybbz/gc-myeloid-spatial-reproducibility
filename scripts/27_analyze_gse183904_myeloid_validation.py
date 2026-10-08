from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu

from project_setup import PROJECT_PATHS, main as setup_main


MARKER_MODULES = {
    "myeloid_score": ["LST1", "TYROBP", "FCER1G", "AIF1", "CTSS", "SAT1", "C1QA", "C1QB", "FCN1", "VCAN", "S100A8", "S100A9"],
    "tcell_score": ["CD3D", "CD3E", "TRBC1", "IL7R", "LTB", "NKG7", "TRAC"],
    "bcell_score": ["MS4A1", "CD79A", "CD74", "HLA-DRA", "CD79B"],
    "epithelial_score": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "KRT17"],
    "stromal_score": ["COL1A1", "COL1A2", "DCN", "LUM", "TAGLN", "RGS5", "ACTA2"],
}

RECEPTOR_POSITIVE = ["CD74", "CSF1R", "TGFBR1", "TGFBR2", "CXCR4", "IL1R1"]
RECEPTOR_NEGATIVE = ["CCR2"]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def score_mean_expr(adata: sc.AnnData, genes: list[str], score_name: str) -> None:
    valid = [gene for gene in genes if gene in adata.var_names]
    if not valid:
        adata.obs[score_name] = np.nan
        return
    sc.tl.score_genes(adata, gene_list=valid, score_name=score_name)


def safe_mwu(df: pd.DataFrame, metric: str, group_col: str) -> dict:
    sub = df[[group_col, metric]].dropna()
    labels = sorted(sub[group_col].astype(str).unique().tolist())
    if len(labels) != 2:
        return {
            "metric": metric,
            "group_col": group_col,
            "group_a": None,
            "group_b": None,
            "n_group_a": 0,
            "n_group_b": 0,
            "median_group_a": np.nan,
            "median_group_b": np.nan,
            "u_statistic": np.nan,
            "pvalue": np.nan,
        }
    group_a, group_b = labels
    vals_a = sub.loc[sub[group_col].eq(group_a), metric].astype(float)
    vals_b = sub.loc[sub[group_col].eq(group_b), metric].astype(float)
    stat = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {
        "metric": metric,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "n_group_a": int(len(vals_a)),
        "n_group_b": int(len(vals_b)),
        "median_group_a": float(vals_a.median()),
        "median_group_b": float(vals_b.median()),
        "u_statistic": float(stat.statistic),
        "pvalue": float(stat.pvalue),
    }


def main() -> None:
    setup_main()
    adata = sc.read_h5ad(PROJECT_PATHS["data_processed"] / "GSE183904" / "GSE183904_targeted.h5ad")
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    discovery_program = pd.read_csv(
        PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"
    )["gene"].dropna().astype(str).tolist()
    trajectory_summary = json.loads(
        (PROJECT_PATHS["results"] / "mechanism_trajectory" / "GSE206785_trajectory_summary.json").read_text(
            encoding="utf-8"
        )
    )
    early_genes = [gene for gene in trajectory_summary["top_early_up_genes"][:20] if gene in adata.var_names]
    late_genes = [gene for gene in trajectory_summary["top_late_up_genes"][:30] if gene in adata.var_names]
    regulator_genes = [row["gene"] for row in trajectory_summary["top_candidate_regulators"][:20] if row["gene"] in adata.var_names]

    for score_name, genes in MARKER_MODULES.items():
        score_mean_expr(adata, genes, score_name)
    score_mean_expr(adata, discovery_program, "suppressive_program_score")
    score_mean_expr(adata, early_genes, "early_module_score")
    score_mean_expr(adata, late_genes, "late_module_score")
    score_mean_expr(adata, regulator_genes, "late_regulator_score")
    score_mean_expr(adata, RECEPTOR_POSITIVE, "receptor_positive_score")
    score_mean_expr(adata, RECEPTOR_NEGATIVE, "receptor_negative_score")

    adata.obs["receptor_switch_score"] = adata.obs["receptor_positive_score"] - adata.obs["receptor_negative_score"].fillna(0.0)
    adata.obs["suppressive_axis_score"] = adata.obs["late_module_score"] - adata.obs["early_module_score"]
    lineage_score_cols = list(MARKER_MODULES.keys())
    adata.obs["dominant_lineage"] = (
        adata.obs[lineage_score_cols].idxmax(axis=1).str.replace("_score", "", regex=False)
    )
    adata.obs["is_myeloid_like"] = adata.obs["dominant_lineage"].eq("myeloid")

    myeloid = adata[adata.obs["is_myeloid_like"]].copy()
    sc.pp.scale(myeloid, max_value=10)
    sc.tl.pca(myeloid, svd_solver="arpack")
    sc.pp.neighbors(myeloid, n_neighbors=20, n_pcs=min(20, myeloid.obsm["X_pca"].shape[1]))
    sc.tl.umap(myeloid)
    sc.tl.leiden(myeloid, resolution=0.5)

    cluster_scores = (
        myeloid.obs.groupby("leiden", observed=True)[
            [
                "suppressive_program_score",
                "early_module_score",
                "late_module_score",
                "suppressive_axis_score",
                "late_regulator_score",
                "receptor_switch_score",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("suppressive_axis_score", ascending=False)
    )
    suppressive_clusters = cluster_scores.head(min(2, len(cluster_scores)))["leiden"].tolist()
    early_cluster = str(cluster_scores.sort_values("suppressive_axis_score", ascending=True).iloc[0]["leiden"])
    myeloid.obs["is_suppressive_cluster"] = myeloid.obs["leiden"].isin(suppressive_clusters)
    myeloid.obs["is_early_cluster"] = myeloid.obs["leiden"].astype(str).eq(early_cluster)

    tissue_summary = (
        myeloid.obs.groupby("tissue_group", observed=True)[
            [
                "suppressive_program_score",
                "early_module_score",
                "late_module_score",
                "suppressive_axis_score",
                "late_regulator_score",
                "receptor_switch_score",
            ]
        ]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    tissue_summary.columns = ["_".join(col).strip("_") for col in tissue_summary.columns]
    tissue_summary = tissue_summary.sort_values("suppressive_axis_score_mean", ascending=False)

    cluster_tissue = (
        myeloid.obs.groupby(["leiden", "tissue_group"], observed=True)
        .agg(
            n_cells=("leiden", "size"),
            mean_axis_score=("suppressive_axis_score", "mean"),
            mean_program_score=("suppressive_program_score", "mean"),
        )
        .reset_index()
        .sort_values(["leiden", "n_cells"], ascending=[True, False])
    )

    sample_summary = (
        adata.obs.groupby(["sample_id", "tissue_group"], observed=True)
        .agg(
            n_cells=("sample_id", "size"),
            myeloid_fraction=("is_myeloid_like", "mean"),
            mean_myeloid_program=("suppressive_program_score", lambda s: float(s[adata.obs.loc[s.index, "is_myeloid_like"]].mean())),
            mean_myeloid_axis=("suppressive_axis_score", lambda s: float(s[adata.obs.loc[s.index, "is_myeloid_like"]].mean())),
            mean_myeloid_late_module=("late_module_score", lambda s: float(s[adata.obs.loc[s.index, "is_myeloid_like"]].mean())),
        )
        .reset_index()
    )

    suppressive_fraction = (
        myeloid.obs.groupby(["sample_id", "tissue_group"], observed=True)
        .agg(
            n_myeloid=("sample_id", "size"),
            early_cluster_fraction=("is_early_cluster", "mean"),
            suppressive_cluster_fraction=("is_suppressive_cluster", "mean"),
            mean_axis_score=("suppressive_axis_score", "mean"),
            mean_program_score=("suppressive_program_score", "mean"),
        )
        .reset_index()
    )

    sample_summary = sample_summary.merge(
        suppressive_fraction[["sample_id", "tissue_group", "early_cluster_fraction", "suppressive_cluster_fraction"]],
        on=["sample_id", "tissue_group"],
        how="left",
    )
    sample_summary["disease_group"] = sample_summary["tissue_group"].str.split("_").str[-1]
    primary_samples = sample_summary[sample_summary["tissue_group"].isin(["stomach_normal", "stomach_tumor"])].copy()

    tissue_burden = (
        myeloid.obs.groupby("tissue_group", observed=True)
        .agg(
            n_myeloid=("tissue_group", "size"),
            early_cluster_fraction=("is_early_cluster", "mean"),
            suppressive_cluster_fraction=("is_suppressive_cluster", "mean"),
            median_axis_score=("suppressive_axis_score", "median"),
            median_program_score=("suppressive_program_score", "median"),
        )
        .reset_index()
        .sort_values("suppressive_cluster_fraction", ascending=False)
    )

    tests = pd.DataFrame(
        [
            safe_mwu(primary_samples, "myeloid_fraction", "disease_group"),
            safe_mwu(primary_samples, "mean_myeloid_program", "disease_group"),
            safe_mwu(primary_samples, "mean_myeloid_axis", "disease_group"),
            safe_mwu(primary_samples, "mean_myeloid_late_module", "disease_group"),
            safe_mwu(primary_samples, "early_cluster_fraction", "disease_group"),
            safe_mwu(primary_samples, "suppressive_cluster_fraction", "disease_group"),
        ]
    ).sort_values("pvalue", na_position="last")

    embedding = pd.DataFrame(
        {
            "cell_id": myeloid.obs_names,
            "sample_id": myeloid.obs["sample_id"].values,
            "tissue_group": myeloid.obs["tissue_group"].values,
            "leiden": myeloid.obs["leiden"].values,
            "umap_1": myeloid.obsm["X_umap"][:, 0],
            "umap_2": myeloid.obsm["X_umap"][:, 1],
            "suppressive_program_score": myeloid.obs["suppressive_program_score"].values,
            "early_module_score": myeloid.obs["early_module_score"].values,
            "late_module_score": myeloid.obs["late_module_score"].values,
            "suppressive_axis_score": myeloid.obs["suppressive_axis_score"].values,
            "late_regulator_score": myeloid.obs["late_regulator_score"].values,
            "receptor_switch_score": myeloid.obs["receptor_switch_score"].values,
            "is_suppressive_cluster": myeloid.obs["is_suppressive_cluster"].values,
        }
    )

    outdir = PROJECT_PATHS["results"] / "myeloid_external_atlas"
    write_csv(outdir / "GSE183904_myeloid_cluster_scores.csv", cluster_scores)
    write_csv(outdir / "GSE183904_myeloid_tissue_summary.csv", tissue_summary)
    write_csv(outdir / "GSE183904_myeloid_cluster_tissue.csv", cluster_tissue)
    write_csv(outdir / "GSE183904_myeloid_sample_summary.csv", sample_summary)
    write_csv(outdir / "GSE183904_myeloid_tissue_cluster_burden.csv", tissue_burden)
    write_csv(outdir / "GSE183904_myeloid_primary_tumor_vs_normal_tests.csv", tests)
    write_csv(outdir / "GSE183904_myeloid_umap_scores.csv", embedding)

    summary = {
        "dataset_id": "GSE183904",
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "myeloid_like_cells": int(myeloid.n_obs),
        "early_cluster": early_cluster,
        "suppressive_clusters": suppressive_clusters,
        "top_cluster_scores": cluster_scores.head(5).to_dict(orient="records"),
        "primary_tumor_vs_normal_tests": tests.to_dict(orient="records"),
        "tissue_cluster_burden": tissue_burden.to_dict(orient="records"),
        "top_tissue_groups_by_axis_mean": tissue_summary.head(6).to_dict(orient="records"),
    }
    (outdir / "GSE183904_myeloid_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
