from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from project_setup import PROJECT_PATHS, main as setup_main


EARLY_MYLOID = {"M02_Mono_CD14", "M05_Mono_VCAN", "M13_Mono_VCAN"}
TRANSITION_MYLOID = {"M08_Mono_INHBA"}
SUPPRESSIVE_MYLOID = {"M04_Macro_C3", "M07_Macro_APOE", "M11_Macro_PLA2G7"}
ACTIVATED_FIBROBLAST = {"F08_Activated_VSTM2A", "F13_Activated_CTHRC1", "F14_Activated_ADAM28"}
TREG = {"T01_CD4_Reg-ICOS", "T11_CD4_Reg-FOXP3"}
CD8_EXHAUSTED = {"T25_CD8_Ex-HAVCR2"}
CD8_LAG3 = {"T20_CD8_RM-LAG3"}
CD8_CXCL13 = {"T24_CD8_TLS-CXCL13"}

EXHAUSTION_GENES = ["HAVCR2", "LAG3", "PDCD1", "TIGIT", "CTLA4", "TOX", "CXCL13"]
CYTOTOXIC_GENES = ["IFNG", "GZMB", "PRF1"]

FORMATION_PAIRS = [
    {
        "context": "formation",
        "sender_group": "fibroblast_activated_tumor",
        "receiver_group": "myeloid_early_tumor",
        "ligand": "CCL2",
        "receptor": "CCR2",
        "axis_label": "CAF CCL2 -> early myeloid CCR2",
    },
    {
        "context": "formation",
        "sender_group": "fibroblast_activated_tumor",
        "receiver_group": "myeloid_early_tumor",
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "axis_label": "CAF CXCL12 -> early myeloid CXCR4",
    },
    {
        "context": "formation",
        "sender_group": "fibroblast_activated_tumor",
        "receiver_group": "myeloid_suppressive_tumor",
        "ligand": "CSF1",
        "receptor": "CSF1R",
        "axis_label": "CAF CSF1 -> suppressive myeloid CSF1R",
    },
    {
        "context": "formation",
        "sender_group": "fibroblast_activated_tumor",
        "receiver_group": "myeloid_suppressive_tumor",
        "ligand": "IL34",
        "receptor": "CSF1R",
        "axis_label": "CAF IL34 -> suppressive myeloid CSF1R",
    },
    {
        "context": "formation",
        "sender_group": "fibroblast_activated_tumor",
        "receiver_group": "myeloid_suppressive_tumor",
        "ligand": "CXCL12",
        "receptor": "CXCR4",
        "axis_label": "CAF CXCL12 -> suppressive myeloid CXCR4",
    },
    {
        "context": "formation",
        "sender_group": "fibroblast_activated_tumor",
        "receiver_group": "myeloid_suppressive_tumor",
        "ligand": "MIF",
        "receptor": "CD74",
        "axis_label": "CAF MIF -> suppressive myeloid CD74",
    },
    {
        "context": "formation",
        "sender_group": "epithelial_tumor",
        "receiver_group": "myeloid_suppressive_tumor",
        "ligand": "MIF",
        "receptor": "CD74",
        "axis_label": "Tumor MIF -> suppressive myeloid CD74",
    },
    {
        "context": "formation",
        "sender_group": "myeloid_transition_inhba_tumor",
        "receiver_group": "myeloid_transition_inhba_tumor",
        "ligand": "IL1B",
        "receptor": "IL1R1",
        "axis_label": "Transitional myeloid IL1B -> IL1R1",
    },
    {
        "context": "formation",
        "sender_group": "myeloid_transition_inhba_tumor",
        "receiver_group": "myeloid_suppressive_tumor",
        "ligand": "TNF",
        "receptor": "TNFRSF1A",
        "axis_label": "Transitional myeloid TNF -> suppressive TNFRSF1A",
    },
]

OUTPUT_PAIRS = [
    {
        "context": "output",
        "sender_group": "myeloid_suppressive_tumor",
        "receiver_group": "cd8_exhausted_tumor",
        "ligand": "LGALS9",
        "receptor": "HAVCR2",
        "axis_label": "Suppressive myeloid LGALS9 -> exhausted CD8 HAVCR2",
    },
    {
        "context": "output",
        "sender_group": "myeloid_suppressive_tumor",
        "receiver_group": "cd8_exhausted_tumor",
        "ligand": "SPP1",
        "receptor": "CD44",
        "axis_label": "Suppressive myeloid SPP1 -> exhausted CD8 CD44",
    },
    {
        "context": "output",
        "sender_group": "myeloid_suppressive_tumor",
        "receiver_group": "cd8_lag3_tumor",
        "ligand": "SPP1",
        "receptor": "CD44",
        "axis_label": "Suppressive myeloid SPP1 -> LAG3+ CD8 CD44",
    },
    {
        "context": "output",
        "sender_group": "myeloid_suppressive_tumor",
        "receiver_group": "treg_tumor",
        "ligand": "SPP1",
        "receptor": "CD44",
        "axis_label": "Suppressive myeloid SPP1 -> Treg CD44",
    },
    {
        "context": "output",
        "sender_group": "myeloid_suppressive_tumor",
        "receiver_group": "cd8_exhausted_tumor",
        "ligand": "TGFB1",
        "receptor": "TGFBR2",
        "axis_label": "Suppressive myeloid TGFB1 -> exhausted CD8 TGFBR2",
    },
    {
        "context": "output",
        "sender_group": "myeloid_suppressive_tumor",
        "receiver_group": "treg_tumor",
        "ligand": "TGFB1",
        "receptor": "TGFBR2",
        "axis_label": "Suppressive myeloid TGFB1 -> Treg TGFBR2",
    },
    {
        "context": "output",
        "sender_group": "myeloid_suppressive_tumor",
        "receiver_group": "cd8_exhausted_tumor",
        "ligand": "IL10",
        "receptor": "IL10RA",
        "axis_label": "Suppressive myeloid IL10 -> exhausted CD8 IL10RA",
    },
]

PATIENT_CORRELATION_PAIRS = [
    ("activated_fibro_fraction", "suppressive_myeloid_fraction"),
    ("suppressive_myeloid_fraction", "all_tcell_exhaustion_score"),
    ("suppressive_myeloid_fraction", "cd8_exhaustion_score"),
    ("suppressive_myeloid_fraction", "cd8_cytotoxic_score"),
    ("fibro_CSF1_x_suppressive_CSF1R", "cd8_exhaustion_score"),
    ("fibro_IL34_x_suppressive_CSF1R", "cd8_exhaustion_score"),
    ("fibro_MIF_x_suppressive_CD74", "cd8_exhaustion_score"),
    ("suppressive_LGALS9_x_cd8_HAVCR2", "suppressive_myeloid_fraction"),
    ("suppressive_LGALS9_x_cd8_HAVCR2", "cd8_exhaustion_score"),
    ("suppressive_SPP1_x_cd8_CD44", "suppressive_myeloid_fraction"),
    ("suppressive_SPP1_x_cd8_CD44", "cd8_exhaustion_score"),
    ("suppressive_TGFB1_x_cd8_TGFBR2", "cd8_exhaustion_score"),
]

MYELOID_RECEPTOR_PANEL = [
    "CCR2",
    "CXCR4",
    "CSF1R",
    "CD74",
    "IL1R1",
    "IL1RAP",
    "TNFRSF1A",
    "TNFRSF1B",
    "TGFBR1",
    "TGFBR2",
    "CD44",
    "ITGAV",
    "ITGB1",
    "IL10RA",
    "IL10RB",
]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_metadata() -> pd.DataFrame:
    path = PROJECT_PATHS["data_raw"] / "GSE206785" / "supplementary" / "GSE206785_metadata.txt.gz"
    return pd.read_csv(path, sep="\t")


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


def assign_focus_group(row: pd.Series) -> str | float:
    annotation = str(row["Annotation"])
    tissue = str(row["Tissue"])
    if tissue != "Tumor":
        return np.nan
    if annotation in EARLY_MYLOID:
        return "myeloid_early_tumor"
    if annotation in TRANSITION_MYLOID:
        return "myeloid_transition_inhba_tumor"
    if annotation in SUPPRESSIVE_MYLOID:
        return "myeloid_suppressive_tumor"
    if annotation in ACTIVATED_FIBROBLAST:
        return "fibroblast_activated_tumor"
    if annotation.startswith("EP"):
        return "epithelial_tumor"
    if annotation in CD8_EXHAUSTED:
        return "cd8_exhausted_tumor"
    if annotation in CD8_LAG3:
        return "cd8_lag3_tumor"
    if annotation in CD8_CXCL13:
        return "cd8_cxcl13_tumor"
    if annotation in TREG:
        return "treg_tumor"
    return np.nan


def summarize_group_expression(expr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for group_name, group_meta in meta.groupby("focus_group"):
        cell_ids = group_meta.index
        sub = expr.loc[cell_ids]
        means = sub.mean(axis=0)
        pcts = (sub > 0).mean(axis=0)
        for gene in expr.columns:
            rows.append(
                {
                    "focus_group": group_name,
                    "gene": gene,
                    "n_cells": int(len(cell_ids)),
                    "mean_expr": float(means[gene]),
                    "pct_expr": float(pcts[gene]),
                }
            )
    return pd.DataFrame(rows)


def build_lr_scores(group_summary: pd.DataFrame, pair_defs: list[dict]) -> pd.DataFrame:
    lookup = group_summary.set_index(["focus_group", "gene"])
    rows: list[dict] = []
    for pair in pair_defs:
        sender = lookup.loc[(pair["sender_group"], pair["ligand"])]
        receiver = lookup.loc[(pair["receiver_group"], pair["receptor"])]
        raw_score = float(sender["mean_expr"] * receiver["mean_expr"] * sender["pct_expr"] * receiver["pct_expr"])
        stabilized_score = float(
            np.log1p(sender["mean_expr"])
            * np.log1p(receiver["mean_expr"])
            * sender["pct_expr"]
            * receiver["pct_expr"]
        )
        rows.append(
            {
                **pair,
                "sender_n_cells": int(sender["n_cells"]),
                "receiver_n_cells": int(receiver["n_cells"]),
                "ligand_mean_expr": float(sender["mean_expr"]),
                "ligand_pct_expr": float(sender["pct_expr"]),
                "receptor_mean_expr": float(receiver["mean_expr"]),
                "receptor_pct_expr": float(receiver["pct_expr"]),
                "raw_product_score": raw_score,
                "stabilized_score": stabilized_score,
            }
        )
    return pd.DataFrame(rows).sort_values(["context", "stabilized_score"], ascending=[True, False]).reset_index(drop=True)


def build_myeloid_receptor_switch(group_summary: pd.DataFrame) -> pd.DataFrame:
    focus = group_summary[
        group_summary["focus_group"].isin(
            ["myeloid_early_tumor", "myeloid_transition_inhba_tumor", "myeloid_suppressive_tumor"]
        )
        & group_summary["gene"].isin(MYELOID_RECEPTOR_PANEL)
    ].copy()
    pivot = focus.pivot(index="gene", columns="focus_group", values="mean_expr").reset_index()
    for col in ["myeloid_early_tumor", "myeloid_transition_inhba_tumor", "myeloid_suppressive_tumor"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["suppressive_vs_early_ratio"] = (
        pivot["myeloid_suppressive_tumor"] + 1e-9
    ) / (pivot["myeloid_early_tumor"] + 1e-9)
    pivot["transition_vs_early_ratio"] = (
        pivot["myeloid_transition_inhba_tumor"] + 1e-9
    ) / (pivot["myeloid_early_tumor"] + 1e-9)
    return pivot.sort_values("suppressive_vs_early_ratio", ascending=False)


def mean_over(expr: pd.DataFrame, cell_ids: pd.Index, gene: str) -> float:
    if len(cell_ids) == 0 or gene not in expr.columns:
        return np.nan
    return float(expr.loc[cell_ids, gene].mean())


def score_over(expr: pd.DataFrame, cell_ids: pd.Index, genes: list[str]) -> float:
    genes = [gene for gene in genes if gene in expr.columns]
    if len(cell_ids) == 0 or not genes:
        return np.nan
    return float(expr.loc[cell_ids, genes].mean(axis=1).mean())


def fraction_in_group(obs: pd.DataFrame, labels: set[str]) -> float:
    if len(obs) == 0:
        return np.nan
    return float(obs["Annotation"].isin(labels).mean())


def build_patient_metrics(meta: pd.DataFrame, expr: pd.DataFrame) -> pd.DataFrame:
    tumor_meta = meta[meta["Tissue"].eq("Tumor")].copy()
    rows: list[dict] = []
    for patient, patient_meta in tumor_meta.groupby("Patient"):
        patient_meta = patient_meta.copy()
        ann = patient_meta["Annotation"].astype(str)
        tumor_myeloid = patient_meta[ann.str.startswith("M")]
        tumor_t = patient_meta[ann.str.startswith("T")]
        tumor_cd8 = patient_meta[ann.str.startswith("T") & ann.str.contains("CD8")]
        tumor_f = patient_meta[ann.str.startswith("F")]
        activated_f = patient_meta[patient_meta["Annotation"].isin(ACTIVATED_FIBROBLAST)]
        suppressive = patient_meta[patient_meta["Annotation"].isin(SUPPRESSIVE_MYLOID)]
        early = patient_meta[patient_meta["Annotation"].isin(EARLY_MYLOID)]

        row = {
            "Patient": patient,
            "tumor_total_cells": int(len(patient_meta)),
            "tumor_myeloid_cells": int(len(tumor_myeloid)),
            "tumor_t_cells": int(len(tumor_t)),
            "tumor_cd8_cells": int(len(tumor_cd8)),
            "tumor_fibroblast_cells": int(len(tumor_f)),
            "suppressive_myeloid_fraction": fraction_in_group(tumor_myeloid, SUPPRESSIVE_MYLOID),
            "transition_myeloid_fraction": fraction_in_group(tumor_myeloid, TRANSITION_MYLOID),
            "early_myeloid_fraction": fraction_in_group(tumor_myeloid, EARLY_MYLOID),
            "activated_fibro_fraction": fraction_in_group(tumor_f, ACTIVATED_FIBROBLAST),
            "exhausted_cd8_fraction": fraction_in_group(tumor_t, CD8_EXHAUSTED),
            "lag3_cd8_fraction": fraction_in_group(tumor_t, CD8_LAG3),
            "cxcl13_cd8_fraction": fraction_in_group(tumor_t, CD8_CXCL13),
            "treg_fraction": fraction_in_group(tumor_t, TREG),
            "all_tcell_exhaustion_score": score_over(expr, tumor_t.index, EXHAUSTION_GENES),
            "all_tcell_cytotoxic_score": score_over(expr, tumor_t.index, CYTOTOXIC_GENES),
            "cd8_exhaustion_score": score_over(expr, tumor_cd8.index, EXHAUSTION_GENES),
            "cd8_cytotoxic_score": score_over(expr, tumor_cd8.index, CYTOTOXIC_GENES),
            "fibro_CSF1_x_suppressive_CSF1R": mean_over(expr, activated_f.index, "CSF1")
            * mean_over(expr, suppressive.index, "CSF1R"),
            "fibro_IL34_x_suppressive_CSF1R": mean_over(expr, activated_f.index, "IL34")
            * mean_over(expr, suppressive.index, "CSF1R"),
            "fibro_CXCL12_x_suppressive_CXCR4": mean_over(expr, activated_f.index, "CXCL12")
            * mean_over(expr, suppressive.index, "CXCR4"),
            "fibro_MIF_x_suppressive_CD74": mean_over(expr, activated_f.index, "MIF")
            * mean_over(expr, suppressive.index, "CD74"),
            "fibro_CCL2_x_early_CCR2": mean_over(expr, activated_f.index, "CCL2")
            * mean_over(expr, early.index, "CCR2"),
            "suppressive_LGALS9_x_cd8_HAVCR2": mean_over(expr, suppressive.index, "LGALS9")
            * mean_over(expr, tumor_cd8.index, "HAVCR2"),
            "suppressive_SPP1_x_cd8_CD44": mean_over(expr, suppressive.index, "SPP1")
            * mean_over(expr, tumor_cd8.index, "CD44"),
            "suppressive_TGFB1_x_cd8_TGFBR2": mean_over(expr, suppressive.index, "TGFB1")
            * mean_over(expr, tumor_cd8.index, "TGFBR2"),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Patient").reset_index(drop=True)


def safe_spearman(df: pd.DataFrame, x: str, y: str) -> dict:
    subset = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3 or subset[x].nunique() < 2 or subset[y].nunique() < 2:
        return {"metric_x": x, "metric_y": y, "n_patients": int(len(subset)), "spearman_rho": np.nan, "pvalue": np.nan}
    rho, pvalue = spearmanr(subset[x], subset[y])
    return {
        "metric_x": x,
        "metric_y": y,
        "n_patients": int(len(subset)),
        "spearman_rho": float(rho),
        "pvalue": float(pvalue),
    }


def to_top_axis_records(df: pd.DataFrame, n_top: int = 5) -> list[dict]:
    cols = [
        "axis_label",
        "sender_group",
        "receiver_group",
        "ligand",
        "receptor",
        "ligand_mean_expr",
        "receptor_mean_expr",
        "raw_product_score",
        "stabilized_score",
    ]
    return df.loc[:, cols].head(n_top).to_dict(orient="records")


def main() -> None:
    setup_main()

    all_pair_defs = FORMATION_PAIRS + OUTPUT_PAIRS
    requested_genes = sorted(
        set(EXHAUSTION_GENES + CYTOTOXIC_GENES + MYELOID_RECEPTOR_PANEL)
        | {pair["ligand"] for pair in all_pair_defs}
        | {pair["receptor"] for pair in all_pair_defs}
    )

    meta = load_metadata()
    expr, available_genes = load_targeted_expression(requested_genes)
    meta = meta.copy()
    meta.index = expr.index
    meta["focus_group"] = meta.apply(assign_focus_group, axis=1)
    focus_meta = meta[meta["focus_group"].notna()].copy()

    group_counts = (
        focus_meta.groupby("focus_group")
        .agg(n_cells=("focus_group", "size"), n_patients=("Patient", "nunique"))
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )
    group_expression = summarize_group_expression(expr, focus_meta)
    lr_scores = build_lr_scores(group_expression, all_pair_defs)
    formation_scores = lr_scores[lr_scores["context"].eq("formation")].copy()
    output_scores = lr_scores[lr_scores["context"].eq("output")].copy()
    receptor_switch = build_myeloid_receptor_switch(group_expression)
    patient_metrics = build_patient_metrics(meta, expr)
    patient_correlations = pd.DataFrame(
        [safe_spearman(patient_metrics, metric_x, metric_y) for metric_x, metric_y in PATIENT_CORRELATION_PAIRS]
    ).sort_values(["pvalue", "spearman_rho"], ascending=[True, False], na_position="last")

    outdir = PROJECT_PATHS["results"] / "cell_crosstalk"
    write_csv(outdir / "GSE206785_focus_group_counts.csv", group_counts)
    write_csv(outdir / "GSE206785_focus_group_gene_summary.csv", group_expression)
    write_csv(outdir / "GSE206785_curated_lr_scores.csv", lr_scores)
    write_csv(outdir / "GSE206785_top_formative_lr_scores.csv", formation_scores)
    write_csv(outdir / "GSE206785_top_output_lr_scores.csv", output_scores)
    write_csv(outdir / "GSE206785_myeloid_receptor_switch.csv", receptor_switch)
    write_csv(outdir / "GSE206785_tumor_patient_state_metrics.csv", patient_metrics)
    write_csv(outdir / "GSE206785_tumor_patient_correlations.csv", patient_correlations)

    summary = {
        "dataset_id": "GSE206785",
        "analysis_module": "cell_crosstalk_and_tcell_coupling",
        "input_scale": "public_log_normalized_gse206785_targeted_gene_panel",
        "genes_requested": requested_genes,
        "genes_available": available_genes,
        "tumor_focus_cells": int(len(focus_meta)),
        "focus_group_counts": group_counts.to_dict(orient="records"),
        "top_formation_axes": to_top_axis_records(formation_scores, n_top=5),
        "top_output_axes": to_top_axis_records(output_scores, n_top=5),
        "key_correlations": patient_correlations.head(8).to_dict(orient="records"),
        "score_definition": {
            "raw_product_score": "ligand_mean_expr * receptor_mean_expr * ligand_pct_expr * receptor_pct_expr",
            "stabilized_score": "log1p(ligand_mean_expr) * log1p(receptor_mean_expr) * ligand_pct_expr * receptor_pct_expr",
        },
    }
    write_json(outdir / "GSE206785_crosstalk_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
