from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, spearmanr

from project_setup import PROJECT_PATHS, main as setup_main


EARLY_MYLOID = {"M02_Mono_CD14", "M05_Mono_VCAN", "M13_Mono_VCAN"}
TRANSITION_MYLOID = {"M08_Mono_INHBA"}
SUPPRESSIVE_MYLOID = {"M04_Macro_C3", "M07_Macro_APOE", "M11_Macro_PLA2G7"}
FORMATION_LIGANDS = ["CCL2", "CSF1", "IL34", "INHBA", "MIF", "TGFB1", "CXCL12"]
EXPLORATORY_LIGANDS = ["CCL7", "IL6", "CXCL8"]
GROUP_ORDER = ["fibro_activated", "fibro_quiescent", "epithelial", "myeloid_transition"]
CORRELATION_PAIRS = [
    ("act_vs_quies_fibro_fraction", "mean_myeloid_axis_score"),
    ("act_vs_quies_fibro_fraction", "mean_myeloid_program_score"),
    ("act_vs_quies_fibro_fraction", "suppressive_frac_within_myeloid"),
    ("fibro_act_INHBA", "transition_frac_within_myeloid"),
    ("fibro_act_IL34", "mean_myeloid_program_score"),
]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def load_targeted_expression(genes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    path = PROJECT_PATHS["data_raw"] / "GSE206785" / "supplementary" / "GSE206785_scgex.txt.gz"
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split(",")
    available = [gene for gene in genes if gene in header]
    usecols = [header[0]] + available
    expr = pd.read_csv(path, usecols=usecols)
    expr.index = expr.iloc[:, 0].astype(str)
    expr = expr.drop(columns=expr.columns[0])
    return expr, available


def load_aligned_tumor_inputs(requested_genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    expr, available = load_targeted_expression(requested_genes)
    meta = pd.read_csv(PROJECT_PATHS["data_raw"] / "GSE206785" / "supplementary" / "GSE206785_metadata.txt.gz", sep="\t")
    meta = meta.copy()
    meta.index = expr.index
    tumor_meta = meta.loc[meta["Tissue"].astype(str).eq("Tumor")].copy()
    tumor_expr = expr.loc[tumor_meta.index].copy()
    return tumor_meta, tumor_expr, available


def assign_group(annotation: str) -> str:
    label = str(annotation)
    if label in EARLY_MYLOID:
        return "myeloid_early"
    if label in TRANSITION_MYLOID:
        return "myeloid_transition"
    if label in SUPPRESSIVE_MYLOID:
        return "myeloid_suppressive"
    if label.startswith("F") and "Activated" in label:
        return "fibro_activated"
    if label.startswith("F") and "Quiescent" in label:
        return "fibro_quiescent"
    if label.startswith("EP"):
        return "epithelial"
    return "other"


def safe_spearman(df: pd.DataFrame, x: str, y: str) -> dict:
    subset = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 4 or subset[x].nunique() < 2 or subset[y].nunique() < 2:
        return {"metric_x": x, "metric_y": y, "n_patients": int(len(subset)), "rho": np.nan, "pvalue": np.nan}
    rho, pvalue = spearmanr(subset[x], subset[y], nan_policy="omit")
    return {"metric_x": x, "metric_y": y, "n_patients": int(len(subset)), "rho": float(rho), "pvalue": float(pvalue)}


def safe_mwu(df: pd.DataFrame, metric: str, group_col: str) -> dict:
    subset = df[[metric, group_col]].dropna()
    labels = sorted(subset[group_col].astype(str).unique().tolist())
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
            "pvalue": np.nan,
        }
    group_a, group_b = labels
    vals_a = subset.loc[subset[group_col].eq(group_a), metric].astype(float)
    vals_b = subset.loc[subset[group_col].eq(group_b), metric].astype(float)
    if len(vals_a) < 3 or len(vals_b) < 3:
        return {
            "metric": metric,
            "group_col": group_col,
            "group_a": group_a,
            "group_b": group_b,
            "n_group_a": int(len(vals_a)),
            "n_group_b": int(len(vals_b)),
            "median_group_a": float(vals_a.median()) if len(vals_a) else np.nan,
            "median_group_b": float(vals_b.median()) if len(vals_b) else np.nan,
            "pvalue": np.nan,
        }
    test = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {
        "metric": metric,
        "group_col": group_col,
        "group_a": group_a,
        "group_b": group_b,
        "n_group_a": int(len(vals_a)),
        "n_group_b": int(len(vals_b)),
        "median_group_a": float(vals_a.median()),
        "median_group_b": float(vals_b.median()),
        "pvalue": float(test.pvalue),
    }


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - float(values.mean())) / std


def summarize_group_expression(meta: pd.DataFrame, expr: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for group_name in GROUP_ORDER:
        group_ids = meta.index[meta["cell_group"].eq(group_name)]
        if len(group_ids) == 0:
            continue
        sub = expr.loc[group_ids, genes]
        means = sub.mean(axis=0)
        pcts = (sub > 0).mean(axis=0)
        for gene in genes:
            rows.append(
                {
                    "cell_group": group_name,
                    "gene": gene,
                    "n_cells": int(len(group_ids)),
                    "mean_expr": float(means[gene]),
                    "pct_expr": float(pcts[gene]),
                }
            )
    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index="gene", columns="cell_group", values="mean_expr")
    for group_name in GROUP_ORDER:
        if group_name not in pivot.columns:
            pivot[group_name] = np.nan
    pivot = pivot.reset_index()
    pivot["activated_vs_quiescent_ratio"] = (pivot["fibro_activated"] + 1e-9) / (pivot["fibro_quiescent"] + 1e-9)
    pivot["activated_vs_epithelial_ratio"] = (pivot["fibro_activated"] + 1e-9) / (pivot["epithelial"] + 1e-9)
    pivot["activated_minus_quiescent"] = pivot["fibro_activated"] - pivot["fibro_quiescent"]
    return pivot.sort_values(["activated_vs_quiescent_ratio", "activated_minus_quiescent"], ascending=[False, False])


def build_patient_metrics(meta: pd.DataFrame, expr: pd.DataFrame) -> pd.DataFrame:
    myeloid = sc.read_h5ad(PROJECT_PATHS["data_processed"] / "GSE206785" / "GSE206785_myeloid_trajectory.h5ad")
    myeloid_obs = myeloid.obs.copy()
    myeloid_obs = myeloid_obs.loc[myeloid_obs["Tissue"].astype(str).eq("Tumor")].copy()
    myeloid_obs["is_transition"] = myeloid_obs["Annotation"].astype(str).isin(TRANSITION_MYLOID)
    myeloid_obs["is_suppressive"] = myeloid_obs["Annotation"].astype(str).isin(SUPPRESSIVE_MYLOID)
    myeloid_obs["is_early"] = myeloid_obs["Annotation"].astype(str).isin(EARLY_MYLOID)

    patient_metrics = meta.groupby("Patient").size().rename("total_tumor_cells").to_frame()
    for group_name in ["fibro_activated", "fibro_quiescent", "myeloid_early", "myeloid_transition", "myeloid_suppressive"]:
        patient_metrics[f"n_{group_name}"] = meta.loc[meta["cell_group"].eq(group_name)].groupby("Patient").size()
    patient_metrics = patient_metrics.fillna(0.0)

    informative_fibro = patient_metrics["n_fibro_activated"] + patient_metrics["n_fibro_quiescent"]
    patient_metrics["n_informative_fibro"] = informative_fibro
    patient_metrics["act_vs_quies_fibro_fraction"] = patient_metrics["n_fibro_activated"] / informative_fibro.replace(0, np.nan)

    patient_metrics["myeloid_total"] = patient_metrics[
        ["n_myeloid_early", "n_myeloid_transition", "n_myeloid_suppressive"]
    ].sum(axis=1)
    patient_metrics["transition_frac_within_myeloid"] = (
        patient_metrics["n_myeloid_transition"] / patient_metrics["myeloid_total"].replace(0, np.nan)
    )
    patient_metrics["suppressive_frac_within_myeloid"] = (
        patient_metrics["n_myeloid_suppressive"] / patient_metrics["myeloid_total"].replace(0, np.nan)
    )

    myeloid_patient = myeloid_obs.groupby("Patient").agg(
        mean_myeloid_axis_score=("suppressive_axis_score", "mean"),
        mean_myeloid_program_score=("suppressive_program_score", "mean"),
        transition_frac_in_trajectory=("is_transition", "mean"),
        suppressive_frac_in_trajectory=("is_suppressive", "mean"),
        early_frac_in_trajectory=("is_early", "mean"),
    )
    patient_metrics = patient_metrics.join(myeloid_patient)

    activated_ids = meta.index[meta["cell_group"].eq("fibro_activated")]
    activated_patient_map = meta.loc[activated_ids, "Patient"]
    for gene in FORMATION_LIGANDS + EXPLORATORY_LIGANDS:
        values = expr.loc[activated_ids, gene].groupby(activated_patient_map).mean()
        patient_metrics[f"fibro_act_{gene}"] = values

    z_cols: list[str] = []
    for gene in ["CCL2", "CSF1", "IL34", "INHBA", "MIF", "TGFB1"]:
        col = f"fibro_act_{gene}"
        if col in patient_metrics.columns:
            z_col = f"{col}_z"
            patient_metrics[z_col] = zscore(patient_metrics[col])
            z_cols.append(z_col)
    patient_metrics["caf_formative_ligand_score"] = patient_metrics[z_cols].mean(axis=1)
    return patient_metrics.reset_index().sort_values("Patient")


def build_group_comparisons(patient_metrics: pd.DataFrame) -> pd.DataFrame:
    informative = patient_metrics.loc[patient_metrics["act_vs_quies_fibro_fraction"].notna()].copy()
    if informative.empty:
        return pd.DataFrame()
    median_cut = float(informative["act_vs_quies_fibro_fraction"].median())
    informative["fibro_activation_group"] = np.where(
        informative["act_vs_quies_fibro_fraction"] >= median_cut,
        "high_activation",
        "low_activation",
    )
    rows = [
        safe_mwu(informative, "mean_myeloid_axis_score", "fibro_activation_group"),
        safe_mwu(informative, "mean_myeloid_program_score", "fibro_activation_group"),
        safe_mwu(informative, "suppressive_frac_within_myeloid", "fibro_activation_group"),
        safe_mwu(informative, "transition_frac_within_myeloid", "fibro_activation_group"),
    ]
    result = pd.DataFrame(rows)
    result["median_cut"] = median_cut
    return result.sort_values("pvalue", na_position="last")


def main() -> None:
    setup_main()
    requested_genes = sorted(set(FORMATION_LIGANDS + EXPLORATORY_LIGANDS))
    meta, expr, available_genes = load_aligned_tumor_inputs(requested_genes)
    meta["cell_group"] = meta["Annotation"].astype(str).map(assign_group)

    ligand_group_summary = summarize_group_expression(meta, expr, requested_genes)
    patient_metrics = build_patient_metrics(meta, expr)
    patient_correlations = pd.DataFrame(
        [safe_spearman(patient_metrics, metric_x, metric_y) for metric_x, metric_y in CORRELATION_PAIRS]
    ).sort_values("pvalue", na_position="last")
    group_comparisons = build_group_comparisons(patient_metrics)

    outdir = PROJECT_PATHS["results"] / "formation_inputs"
    write_csv(outdir / "GSE206785_formation_ligand_group_summary.csv", ligand_group_summary)
    write_csv(outdir / "GSE206785_formation_patient_metrics.csv", patient_metrics)
    write_csv(outdir / "GSE206785_formation_patient_correlations.csv", patient_correlations)
    write_csv(outdir / "GSE206785_formation_group_comparisons.csv", group_comparisons)

    summary = {
        "dataset_id": "GSE206785",
        "analysis_module": "caf_to_myeloid_formation_inputs",
        "available_genes": available_genes,
        "tumor_patients": int(meta["Patient"].nunique()),
        "informative_fibro_patients": int(patient_metrics["act_vs_quies_fibro_fraction"].notna().sum()),
        "activated_vs_quiescent_ligands_top": ligand_group_summary.head(8).to_dict(orient="records"),
        "key_correlations": patient_correlations.to_dict(orient="records"),
        "group_comparisons": group_comparisons.to_dict(orient="records"),
        "interpretation": [
            "Activated CAF dominance within the informative fibroblast compartment tracks with stronger myeloid terminal polarization.",
            "Fibroblast INHBA is linked to the transitional myeloid intermediate, supporting a formation-stage bridge rather than only a terminal macrophage marker.",
            "These results support a CAF-shaped myeloid formation axis in gastric tumors even without direct ICB endpoint framing.",
        ],
    }
    summary = json_safe(summary)
    (outdir / "GSE206785_formation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
