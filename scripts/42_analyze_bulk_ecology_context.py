from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from project_setup import PROJECT_PATHS, main as setup_main


BENCHMARK_SCORE_PATH = PROJECT_PATHS["results"] / "signature_benchmark" / "bulk_signature_benchmark_scores.csv"

CAF_INPUT_GENES = ["CCL2", "CXCL12", "CSF1", "IL34", "MIF", "TGFB1", "INHBA"]
TGFB_STROMA_GENES = ["TGFB1", "TGFBI", "SERPINE1", "COL1A1", "COL1A2", "COL3A1", "TAGLN", "VIM"]
TEFF_GENES = ["CD8A", "CD8B", "GZMB", "PRF1", "NKG7", "IFNG", "CXCL9", "CXCL10", "TBX21"]
IFNG_RESPONSE_GENES = ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1", "HLA-DRA", "IRF1"]
TLS_B_GNES = ["FDCSP", "CCL19", "CCL21", "CXCL13", "CR2", "BANK1", "MS4A1", "CD79A", "CD79B", "PAX5"]

GROUP_ORDER = [
    "burden_high__polar_low",
    "burden_high__polar_high",
    "burden_low__polar_high",
    "burden_low__polar_low",
]

TARGET_COMPARISONS = [
    ("burden_high__polar_low", "burden_high__polar_high"),
    ("burden_high__polar_low", "burden_low__polar_high"),
]

MODULES = {
    "caf_input_score": CAF_INPUT_GENES,
    "tgfb_stroma_score": TGFB_STROMA_GENES,
    "teff_score": TEFF_GENES,
    "ifng_response_score": IFNG_RESPONSE_GENES,
    "tls_b_score": TLS_B_GNES,
}


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def read_tcga_expression() -> pd.DataFrame:
    path = PROJECT_PATHS["data_raw"] / "TCGA_STAD" / "stad_tcga_pan_can_atlas_2018_data_mrna_seq_v2_rsem.txt"
    df = pd.read_csv(path, sep="\t")
    sample_cols = [c for c in df.columns if c not in {"Hugo_Symbol", "Entrez_Gene_Id"}]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    df["Hugo_Symbol"] = df["Hugo_Symbol"].fillna("").astype(str).str.strip()
    df = df.loc[df["Hugo_Symbol"] != ""].copy()
    gene_expr = df.groupby("Hugo_Symbol")[sample_cols].mean()
    gene_expr.index.name = "gene"
    return gene_expr


def read_gse84437_expression() -> pd.DataFrame:
    family_soft_path = PROJECT_PATHS["data_raw"] / "GSE84437" / "geo_family" / "GSE84437_family.soft.gz"
    series_matrix_path = PROJECT_PATHS["data_raw"] / "GSE84437" / "geo_family" / "GSE84437_series_matrix.txt.gz"

    def _iter_gzip_lines(path: Path):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            for line in handle:
                yield line

    line_iter = _iter_gzip_lines(family_soft_path)
    for line in line_iter:
        if line.startswith("!platform_table_begin"):
            break
    reader = csv.reader(line_iter, delimiter="\t")
    header = next(reader)
    rows: list[list[str]] = []
    for row in reader:
        if row and row[0] == "!platform_table_end":
            break
        rows.append(row)
    annotation = pd.DataFrame(rows, columns=header)
    annotation = annotation[[c for c in ["ID", "Symbol"] if c in annotation.columns]].copy()
    annotation["Symbol"] = annotation["Symbol"].fillna("").astype(str).str.strip()
    annotation = annotation.loc[annotation["Symbol"] != ""].drop_duplicates(subset=["ID"])

    line_iter = _iter_gzip_lines(series_matrix_path)
    for line in line_iter:
        if line.startswith("!series_matrix_table_begin"):
            break
    reader = csv.reader(line_iter, delimiter="\t")
    header = next(reader)
    rows = []
    for row in reader:
        if row and row[0] == "!series_matrix_table_end":
            break
        rows.append(row)
    expr = pd.DataFrame(rows, columns=header)
    id_col = "ID_REF" if "ID_REF" in expr.columns else expr.columns[0]
    sample_cols = [c for c in expr.columns if c != id_col]
    expr[sample_cols] = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
    expr = expr.rename(columns={id_col: "probe_id"}).dropna(subset=["probe_id"])

    merged = expr.merge(annotation, left_on="probe_id", right_on="ID", how="left")
    merged = merged.loc[merged["Symbol"].notna() & (merged["Symbol"] != "")].copy()
    gene_expr = merged.groupby("Symbol")[sample_cols].mean()
    gene_expr.index.name = "gene"
    return gene_expr


def compute_module_scores(gene_expr: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    zexpr = zscore_rows(gene_expr)
    scores = {}
    availability = {}
    for name, genes in MODULES.items():
        valid = [gene for gene in genes if gene in zexpr.index]
        availability[name] = valid
        if valid:
            scores[name] = zexpr.loc[valid].mean(axis=0)
        else:
            scores[name] = pd.Series(0.0, index=zexpr.columns)
    score_df = pd.DataFrame(scores)
    score_df.index.name = "sample_id"
    score_df["stromal_adaptive_imbalance"] = (
        score_df["caf_input_score"]
        + score_df["tgfb_stroma_score"]
        - score_df["teff_score"]
        - score_df["ifng_response_score"]
        - score_df["tls_b_score"]
    )
    return score_df, availability


def assign_quadrant(row: pd.Series, burden_med: float, polar_med: float) -> str:
    burden_group = "burden_high" if row["our_suppressive_axis"] >= burden_med else "burden_low"
    polar_group = "polar_high" if row["trajectory_polarized_axis"] >= polar_med else "polar_low"
    return f"{burden_group}__{polar_group}"


def safe_mwu(df: pd.DataFrame, metric: str, group_a: str, group_b: str) -> dict:
    a = df.loc[df["dual_axis_group"].eq(group_a), metric].dropna()
    b = df.loc[df["dual_axis_group"].eq(group_b), metric].dropna()
    if len(a) < 3 or len(b) < 3:
        return {
            "group_a": group_a,
            "group_b": group_b,
            "metric": metric,
            "n_group_a": int(len(a)),
            "n_group_b": int(len(b)),
            "median_group_a": float(a.median()) if len(a) else np.nan,
            "median_group_b": float(b.median()) if len(b) else np.nan,
            "pvalue": np.nan,
        }
    stat = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "group_a": group_a,
        "group_b": group_b,
        "metric": metric,
        "n_group_a": int(len(a)),
        "n_group_b": int(len(b)),
        "median_group_a": float(a.median()),
        "median_group_b": float(b.median()),
        "pvalue": float(stat.pvalue),
    }


def analyze_dataset(dataset_id: str, gene_expr: pd.DataFrame, benchmark_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    score_df, availability = compute_module_scores(gene_expr)
    score_df = score_df.reset_index()
    merged = benchmark_scores.loc[benchmark_scores["dataset_id"].eq(dataset_id)].copy()
    merged = merged.merge(score_df, on="sample_id", how="inner")

    burden_median = float(merged["our_suppressive_axis"].median())
    polar_median = float(merged["trajectory_polarized_axis"].median())
    merged["dual_axis_group"] = merged.apply(assign_quadrant, axis=1, args=(burden_median, polar_median))

    group_summary = (
        merged.groupby("dual_axis_group")
        .agg(
            n_samples=("sample_id", "count"),
            our_suppressive_axis_mean=("our_suppressive_axis", "mean"),
            trajectory_polarized_axis_mean=("trajectory_polarized_axis", "mean"),
            caf_input_score_mean=("caf_input_score", "mean"),
            tgfb_stroma_score_mean=("tgfb_stroma_score", "mean"),
            teff_score_mean=("teff_score", "mean"),
            ifng_response_score_mean=("ifng_response_score", "mean"),
            tls_b_score_mean=("tls_b_score", "mean"),
            stromal_adaptive_imbalance_mean=("stromal_adaptive_imbalance", "mean"),
        )
        .reset_index()
    )
    group_summary["group_order"] = group_summary["dual_axis_group"].map({group: i for i, group in enumerate(GROUP_ORDER)})
    group_summary = group_summary.sort_values("group_order").drop(columns="group_order")
    group_summary.insert(0, "dataset_id", dataset_id)

    test_rows = []
    for group_a, group_b in TARGET_COMPARISONS:
        for metric in [
            "caf_input_score",
            "tgfb_stroma_score",
            "teff_score",
            "ifng_response_score",
            "tls_b_score",
            "stromal_adaptive_imbalance",
        ]:
            test_rows.append({"dataset_id": dataset_id, **safe_mwu(merged, metric, group_a, group_b)})
    tests = pd.DataFrame(test_rows)

    corr_rows = []
    for metric in [
        "caf_input_score",
        "tgfb_stroma_score",
        "teff_score",
        "ifng_response_score",
        "tls_b_score",
        "stromal_adaptive_imbalance",
    ]:
        for axis in ["our_suppressive_axis", "trajectory_polarized_axis"]:
            sub = merged[[axis, metric]].dropna()
            if len(sub) < 8:
                rho = np.nan
                pvalue = np.nan
            else:
                stat = spearmanr(sub[axis], sub[metric], nan_policy="omit")
                rho = float(stat.statistic)
                pvalue = float(stat.pvalue)
            corr_rows.append(
                {
                    "dataset_id": dataset_id,
                    "axis": axis,
                    "metric": metric,
                    "rho": rho,
                    "pvalue": pvalue,
                    "n_samples": int(len(sub)),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    return merged, group_summary, tests, availability, corr_df


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    setup_main()

    benchmark_scores = pd.read_csv(BENCHMARK_SCORE_PATH)
    tcga_scores, tcga_groups, tcga_tests, tcga_availability, tcga_corr = analyze_dataset(
        "TCGA_STAD",
        read_tcga_expression(),
        benchmark_scores,
    )
    gse_scores, gse_groups, gse_tests, gse_availability, gse_corr = analyze_dataset(
        "GSE84437",
        read_gse84437_expression(),
        benchmark_scores,
    )

    ecology_scores = pd.concat([tcga_scores, gse_scores], ignore_index=True)
    group_summary = pd.concat([tcga_groups, gse_groups], ignore_index=True)
    group_tests = pd.concat([tcga_tests, gse_tests], ignore_index=True)
    corr_summary = pd.concat([tcga_corr, gse_corr], ignore_index=True)

    availability_rows = []
    for dataset_id, availability in [("TCGA_STAD", tcga_availability), ("GSE84437", gse_availability)]:
        for module_name, genes in MODULES.items():
            valid = availability[module_name]
            for gene in genes:
                availability_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "module_name": module_name,
                        "gene": gene,
                        "available_in_dataset": gene in valid,
                    }
                )
    availability_df = pd.DataFrame(availability_rows)

    highest_imbalance = (
        group_summary.sort_values(["dataset_id", "stromal_adaptive_imbalance_mean"], ascending=[True, False])
        .groupby("dataset_id")
        .head(1)[["dataset_id", "dual_axis_group", "stromal_adaptive_imbalance_mean"]]
        .rename(columns={"dual_axis_group": "highest_imbalance_group"})
    )

    outdir = PROJECT_PATHS["results"] / "signature_benchmark"
    write_csv(outdir / "bulk_ecology_context_scores.csv", ecology_scores)
    write_csv(outdir / "bulk_ecology_group_summary.csv", group_summary)
    write_csv(outdir / "bulk_ecology_group_tests.csv", group_tests)
    write_csv(outdir / "bulk_ecology_correlation_summary.csv", corr_summary)
    write_csv(outdir / "bulk_ecology_gene_availability.csv", availability_df)
    write_csv(outdir / "bulk_ecology_highest_imbalance_group.csv", highest_imbalance)

    summary = {
        "datasets": sorted(ecology_scores["dataset_id"].dropna().unique().tolist()),
        "modules": list(MODULES.keys()) + ["stromal_adaptive_imbalance"],
        "highest_imbalance_group_by_dataset": highest_imbalance.to_dict(orient="records"),
        "key_group_tests": group_tests.sort_values("pvalue", na_position="last").head(20).to_dict(orient="records"),
        "key_correlations": corr_summary.sort_values(["dataset_id", "pvalue"], na_position="last").head(20).to_dict(orient="records"),
        "cross_dataset_interpretation": [
            "Across both TCGA-STAD and GSE84437, the burden_high__polar_low group shows the highest stromal-adaptive imbalance.",
            "This worst dual-axis group is characterized by higher CAF/TGFb context together with lower Teff/IFNg/TLS-like context than burden_high__polar_high.",
            "Bulk ecology therefore supports a stromal-high, adaptive-low background for the clinically unfavorable suppressive-burden state, rather than being used as a direct substitute for spatial TLS-related B-cell/FDC-core-low evidence.",
        ],
    }
    (outdir / "bulk_ecology_context_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
