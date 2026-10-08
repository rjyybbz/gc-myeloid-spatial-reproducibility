from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, spearmanr

from project_setup import PROJECT_PATHS, main as setup_main


MARKER_MODULES = {
    "myeloid_score": ["LST1", "TYROBP", "FCER1G", "AIF1", "CTSS", "SAT1", "C1QA", "C1QB", "FCN1", "VCAN", "S100A8", "S100A9"],
    "tcell_score": ["CD3D", "CD3E", "TRBC1", "IL7R", "LTB", "NKG7", "TRAC"],
    "bcell_score": ["MS4A1", "CD79A", "CD74", "HLA-DRA", "CD79B"],
    "epithelial_score": ["EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "KRT17"],
    "stromal_score": ["COL1A1", "COL1A2", "DCN", "LUM", "TAGLN", "RGS5", "ACTA2"],
}

SUPPRESSIVE_RECEPTOR_GENES = ["CD74", "CSF1R", "TGFBR1", "TGFBR2", "CXCR4", "IL10RA", "IL10RB"]
INFLAMMATORY_RECEPTOR_GENES = ["CCR2", "IL1R1"]
FORMATION_INPUT_GENES = ["CCL2", "CXCL12", "CSF1", "IL34", "MIF", "TGFB1", "INHBA"]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def score_mean_expr(adata: sc.AnnData, genes: list[str], score_name: str) -> list[str]:
    valid = [gene for gene in genes if gene in adata.var_names]
    if valid:
        sc.tl.score_genes(adata, gene_list=valid, score_name=score_name)
    else:
        adata.obs[score_name] = np.nan
    return valid


def safe_mwu(df: pd.DataFrame, metric: str, group_col: str, group_a: str, group_b: str) -> dict:
    vals_a = df.loc[df[group_col].eq(group_a), metric].dropna().astype(float)
    vals_b = df.loc[df[group_col].eq(group_b), metric].dropna().astype(float)
    if len(vals_a) < 10 or len(vals_b) < 10:
        return {
            "metric": metric,
            "group_col": group_col,
            "group_a": group_a,
            "group_b": group_b,
            "n_group_a": int(len(vals_a)),
            "n_group_b": int(len(vals_b)),
            "median_group_a": np.nan if len(vals_a) == 0 else float(vals_a.median()),
            "median_group_b": np.nan if len(vals_b) == 0 else float(vals_b.median()),
            "pvalue": np.nan,
        }
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
        "pvalue": float(stat.pvalue),
    }


def safe_spearman(df: pd.DataFrame, x: str, y: str) -> dict:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 50:
        return {"metric_x": x, "metric_y": y, "n_cells": int(len(sub)), "rho": np.nan, "pvalue": np.nan}
    rho, pvalue = spearmanr(sub[x], sub[y], nan_policy="omit")
    return {"metric_x": x, "metric_y": y, "n_cells": int(len(sub)), "rho": float(rho), "pvalue": float(pvalue)}


def main() -> None:
    setup_main()
    adata = sc.read_h5ad(PROJECT_PATHS["data_processed"] / "GSE183904" / "GSE183904_targeted.h5ad")
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    trajectory_summary = json.loads(
        (PROJECT_PATHS["results"] / "mechanism_trajectory" / "GSE206785_trajectory_summary.json").read_text(
            encoding="utf-8"
        )
    )
    discovery_program = pd.read_csv(
        PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"
    )["gene"].dropna().astype(str).tolist()
    early_genes = trajectory_summary["top_early_up_genes"][:20]
    late_genes = trajectory_summary["top_late_up_genes"][:30]
    regulator_genes = [row["gene"] for row in trajectory_summary["top_candidate_regulators"][:20]]

    for score_name, genes in MARKER_MODULES.items():
        score_mean_expr(adata, genes, score_name)
    score_mean_expr(adata, discovery_program, "suppressive_program_score")
    score_mean_expr(adata, early_genes, "early_module_score")
    score_mean_expr(adata, late_genes, "late_module_score")
    score_mean_expr(adata, regulator_genes, "late_regulator_score")
    score_mean_expr(adata, SUPPRESSIVE_RECEPTOR_GENES, "suppressive_receptor_score")
    score_mean_expr(adata, INFLAMMATORY_RECEPTOR_GENES, "inflammatory_receptor_score")
    score_mean_expr(adata, FORMATION_INPUT_GENES, "formation_input_score")

    adata.obs["trajectory_axis_score"] = adata.obs["late_module_score"] - adata.obs["early_module_score"]
    adata.obs["receptor_switch_score"] = adata.obs["suppressive_receptor_score"] - adata.obs["inflammatory_receptor_score"].fillna(0.0)
    lineage_score_cols = list(MARKER_MODULES.keys())
    adata.obs["dominant_lineage"] = adata.obs[lineage_score_cols].idxmax(axis=1).str.replace("_score", "", regex=False)
    adata.obs["is_myeloid_like"] = adata.obs["dominant_lineage"].eq("myeloid")

    myeloid = adata[adata.obs["is_myeloid_like"]].copy()
    sc.pp.scale(myeloid, max_value=10)
    sc.tl.pca(myeloid, svd_solver="arpack")
    sc.pp.neighbors(myeloid, n_neighbors=20, n_pcs=min(20, myeloid.obsm["X_pca"].shape[1]))
    sc.tl.umap(myeloid)
    sc.tl.leiden(myeloid, resolution=0.5)

    cluster_summary = (
        myeloid.obs.groupby("leiden", observed=True)[
            [
                "suppressive_program_score",
                "early_module_score",
                "late_module_score",
                "trajectory_axis_score",
                "late_regulator_score",
                "receptor_switch_score",
                "formation_input_score",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values("trajectory_axis_score", ascending=False)
    )

    early_cluster = str(cluster_summary.sort_values("trajectory_axis_score", ascending=True).iloc[0]["leiden"])
    suppressive_clusters = cluster_summary.head(min(2, len(cluster_summary)))["leiden"].astype(str).tolist()
    myeloid.obs["mechanism_state"] = np.where(
        myeloid.obs["leiden"].astype(str).isin(suppressive_clusters),
        "suppressive_like",
        np.where(myeloid.obs["leiden"].astype(str).eq(early_cluster), "early_like", "intermediate_like"),
    )

    tissue_summary = (
        myeloid.obs.groupby("tissue_group", observed=True)[
            [
                "suppressive_program_score",
                "trajectory_axis_score",
                "late_regulator_score",
                "receptor_switch_score",
                "formation_input_score",
            ]
        ]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    tissue_summary.columns = ["_".join(col).strip("_") for col in tissue_summary.columns]
    tissue_summary = tissue_summary.sort_values("trajectory_axis_score_mean", ascending=False)

    state_summary = (
        myeloid.obs.groupby("mechanism_state", observed=True)[
            ["trajectory_axis_score", "late_regulator_score", "receptor_switch_score", "formation_input_score"]
        ]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    state_summary.columns = ["_".join(col).strip("_") for col in state_summary.columns]

    tests = pd.DataFrame(
        [
            safe_mwu(myeloid.obs, "trajectory_axis_score", "mechanism_state", "suppressive_like", "early_like"),
            safe_mwu(myeloid.obs, "late_regulator_score", "mechanism_state", "suppressive_like", "early_like"),
            safe_mwu(myeloid.obs, "receptor_switch_score", "mechanism_state", "suppressive_like", "early_like"),
            safe_mwu(myeloid.obs, "formation_input_score", "mechanism_state", "suppressive_like", "early_like"),
            safe_mwu(myeloid.obs, "trajectory_axis_score", "tissue_group", "peritoneum_tumor", "stomach_tumor"),
            safe_mwu(myeloid.obs, "late_regulator_score", "tissue_group", "peritoneum_tumor", "stomach_tumor"),
            safe_mwu(myeloid.obs, "receptor_switch_score", "tissue_group", "peritoneum_tumor", "stomach_tumor"),
            safe_mwu(myeloid.obs, "formation_input_score", "tissue_group", "peritoneum_tumor", "stomach_tumor"),
        ]
    ).sort_values("pvalue", na_position="last")

    correlations = pd.DataFrame(
        [
            safe_spearman(myeloid.obs, "trajectory_axis_score", "late_regulator_score"),
            safe_spearman(myeloid.obs, "trajectory_axis_score", "receptor_switch_score"),
            safe_spearman(myeloid.obs, "trajectory_axis_score", "formation_input_score"),
            safe_spearman(myeloid.obs, "late_regulator_score", "receptor_switch_score"),
        ]
    ).sort_values("pvalue", na_position="last")

    embedding = pd.DataFrame(
        {
            "cell_id": myeloid.obs_names,
            "sample_id": myeloid.obs["sample_id"].values,
            "tissue_group": myeloid.obs["tissue_group"].values,
            "leiden": myeloid.obs["leiden"].values,
            "mechanism_state": myeloid.obs["mechanism_state"].values,
            "umap_1": myeloid.obsm["X_umap"][:, 0],
            "umap_2": myeloid.obsm["X_umap"][:, 1],
            "suppressive_program_score": myeloid.obs["suppressive_program_score"].values,
            "trajectory_axis_score": myeloid.obs["trajectory_axis_score"].values,
            "late_regulator_score": myeloid.obs["late_regulator_score"].values,
            "receptor_switch_score": myeloid.obs["receptor_switch_score"].values,
            "formation_input_score": myeloid.obs["formation_input_score"].values,
        }
    )

    outdir = PROJECT_PATHS["results"] / "myeloid_external_mechanism"
    write_csv(outdir / "GSE183904_mechanism_cluster_summary.csv", cluster_summary)
    write_csv(outdir / "GSE183904_mechanism_tissue_summary.csv", tissue_summary)
    write_csv(outdir / "GSE183904_mechanism_state_summary.csv", state_summary)
    write_csv(outdir / "GSE183904_mechanism_tests.csv", tests)
    write_csv(outdir / "GSE183904_mechanism_correlations.csv", correlations)
    write_csv(outdir / "GSE183904_mechanism_umap_scores.csv", embedding)

    summary = {
        "dataset_id": "GSE183904",
        "myeloid_like_cells": int(myeloid.n_obs),
        "early_cluster": early_cluster,
        "suppressive_clusters": suppressive_clusters,
        "cluster_summary_top": cluster_summary.head(5).to_dict(orient="records"),
        "tissue_summary_top": tissue_summary.head(6).to_dict(orient="records"),
        "state_tests": tests.to_dict(orient="records"),
        "correlations": correlations.to_dict(orient="records"),
    }
    (outdir / "GSE183904_mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
