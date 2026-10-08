from __future__ import annotations

import csv
import gzip
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from project_setup import PROJECT_PATHS, main as setup_main


DATASET_ID = "GSE84437"


def read_program_genes() -> list[str]:
    path = PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"
    return pd.read_csv(path)["gene"].dropna().astype(str).tolist()


def _iter_gzip_lines(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            yield line


def read_platform_annotation(family_soft_path: Path) -> pd.DataFrame:
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
    df = pd.DataFrame(rows, columns=header)
    keep = [c for c in ["ID", "Symbol", "RefSeq_ID", "Entrez_Gene_ID"] if c in df.columns]
    df = df[keep].copy()
    df["Symbol"] = df["Symbol"].fillna("").astype(str).str.strip()
    df = df.loc[df["Symbol"] != ""].drop_duplicates(subset=["ID"])
    return df


def read_series_matrix(series_matrix_path: Path) -> pd.DataFrame:
    line_iter = _iter_gzip_lines(series_matrix_path)
    for line in line_iter:
        if line.startswith("!series_matrix_table_begin"):
            break
    reader = csv.reader(line_iter, delimiter="\t")
    header = next(reader)
    rows: list[list[str]] = []
    for row in reader:
        if row and row[0] == "!series_matrix_table_end":
            break
        rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    id_col = "ID_REF" if "ID_REF" in df.columns else df.columns[0]
    sample_cols = [c for c in df.columns if c != id_col]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={id_col: "probe_id"}).dropna(subset=["probe_id"])
    return df


def parse_characteristics(cell: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in str(cell).split(" | "):
        if ": " not in part:
            continue
        key, value = part.split(": ", 1)
        key = key.strip().lower().replace(" ", "_")
        result[key] = value.strip()
    return result


def read_clinical_manifest() -> pd.DataFrame:
    path = PROJECT_PATHS["metadata"] / "sample_manifests" / f"{DATASET_ID}_sample_manifest.csv"
    df = pd.read_csv(path)
    parsed = df["characteristics_ch1"].apply(parse_characteristics)
    clinical = pd.DataFrame(parsed.tolist())
    clinical.insert(0, "gsm_id", df["gsm_id"].astype(str))
    clinical.insert(1, "title", df["title"].astype(str))
    if "age" in clinical.columns:
        clinical["age"] = pd.to_numeric(clinical["age"], errors="coerce")
    if "death" in clinical.columns:
        clinical["os_event"] = pd.to_numeric(clinical["death"], errors="coerce")
    if "duration_overall_survival" in clinical.columns:
        clinical["os_months"] = pd.to_numeric(clinical["duration_overall_survival"], errors="coerce")
    return clinical


def build_gene_expression(expr: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    merged = expr.merge(annotation, left_on="probe_id", right_on="ID", how="left")
    merged = merged.loc[merged["Symbol"].notna() & (merged["Symbol"] != "")].copy()
    sample_cols = [c for c in expr.columns if c != "probe_id"]
    gene_expr = merged.groupby("Symbol")[sample_cols].mean()
    gene_expr.index.name = "gene"
    return gene_expr


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def surv_prob_at(sf: SurvfuncRight, timepoint: float) -> float | None:
    if len(sf.surv_times) == 0:
        return None
    mask = sf.surv_times <= timepoint
    if not np.any(mask):
        return 1.0
    return float(sf.surv_prob[np.where(mask)[0][-1]])


def fit_cox(formula: str, data: pd.DataFrame) -> list[dict]:
    if data.empty:
        return []
    model = PHReg.from_formula(formula, status="os_event", data=data)
    result = model.fit(disp=0)
    rows = []
    for name, coef, pval, se in zip(model.exog_names, result.params, result.pvalues, result.bse):
        rows.append(
            {
                "term": name,
                "coef": float(coef),
                "se": float(se),
                "hazard_ratio": float(np.exp(coef)),
                "ci95_lower": float(np.exp(coef - 1.96 * se)),
                "ci95_upper": float(np.exp(coef + 1.96 * se)),
                "p_value": float(pval),
            }
        )
    return rows


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    setup_main()

    family_soft_path = PROJECT_PATHS["data_raw"] / DATASET_ID / "geo_family" / f"{DATASET_ID}_family.soft.gz"
    series_matrix_path = PROJECT_PATHS["data_raw"] / DATASET_ID / "geo_family" / f"{DATASET_ID}_series_matrix.txt.gz"

    program_genes = read_program_genes()
    annotation = read_platform_annotation(family_soft_path)
    expr = read_series_matrix(series_matrix_path)
    gene_expr = build_gene_expression(expr, annotation)
    clinical = read_clinical_manifest()

    available_program_genes = [g for g in program_genes if g in gene_expr.index]
    gene_expr_program = gene_expr.loc[available_program_genes].copy()
    gene_expr_program_z = zscore_rows(gene_expr_program)
    signature_score = gene_expr_program_z.mean(axis=0).rename("suppressive_myeloid_score")

    score_df = signature_score.reset_index().rename(columns={"index": "gsm_id"})
    score_df["signature_gene_count"] = len(available_program_genes)
    score_df["signature_gene_fraction"] = len(available_program_genes) / max(len(program_genes), 1)

    merged = clinical.merge(score_df, on="gsm_id", how="inner")
    merged["score_group"] = np.where(
        merged["suppressive_myeloid_score"] >= merged["suppressive_myeloid_score"].median(),
        "high",
        "low",
    )
    merged["signature_score_std"] = (
        merged["suppressive_myeloid_score"] - merged["suppressive_myeloid_score"].mean()
    ) / merged["suppressive_myeloid_score"].std(ddof=0)

    survival_df = merged.dropna(subset=["os_months", "os_event", "signature_score_std"]).copy()
    univ_rows = fit_cox("os_months ~ signature_score_std", survival_df)

    adjusted_df = survival_df.dropna(subset=["age", "sex", "ptstage", "pnstage"]).copy()
    adjusted_rows: list[dict] = []
    if not adjusted_df.empty:
        adjusted_rows = fit_cox(
            "os_months ~ signature_score_std + age + C(sex) + C(ptstage) + C(pnstage)",
            adjusted_df,
        )

    high_df = survival_df.loc[survival_df["score_group"] == "high"].copy()
    low_df = survival_df.loc[survival_df["score_group"] == "low"].copy()
    logrank_stat, logrank_p = survdiff(
        survival_df["os_months"].to_numpy(),
        survival_df["os_event"].to_numpy(),
        (survival_df["score_group"] == "high").astype(int).to_numpy(),
    )

    high_sf = SurvfuncRight(high_df["os_months"].to_numpy(), high_df["os_event"].to_numpy())
    low_sf = SurvfuncRight(low_df["os_months"].to_numpy(), low_df["os_event"].to_numpy())
    death_mwu = mannwhitneyu(
        merged.loc[merged["os_event"] == 1, "suppressive_myeloid_score"],
        merged.loc[merged["os_event"] == 0, "suppressive_myeloid_score"],
        alternative="two-sided",
    )

    outdir = PROJECT_PATHS["results"] / "bulk_transfer"
    outdir.mkdir(parents=True, exist_ok=True)

    availability_df = pd.DataFrame(
        {
            "gene": program_genes,
            "available_in_bulk": [g in available_program_genes for g in program_genes],
        }
    )
    write_csv(outdir / f"{DATASET_ID}_signature_gene_availability.csv", availability_df)

    score_export_cols = [
        "gsm_id",
        "title",
        "age",
        "sex",
        "ptstage",
        "pnstage",
        "os_event",
        "os_months",
        "suppressive_myeloid_score",
        "signature_score_std",
        "score_group",
    ]
    write_csv(outdir / f"{DATASET_ID}_signature_scores.csv", merged[score_export_cols].sort_values("suppressive_myeloid_score", ascending=False))

    group_summary = merged.groupby("score_group").agg(
        n_samples=("gsm_id", "count"),
        mean_score=("suppressive_myeloid_score", "mean"),
        median_score=("suppressive_myeloid_score", "median"),
        death_rate=("os_event", "mean"),
        median_followup_months=("os_months", "median"),
    ).reset_index()
    write_csv(outdir / f"{DATASET_ID}_score_group_summary.csv", group_summary)

    cox_rows = []
    for row in univ_rows:
        row = {"model": "univ_continuous", **row}
        cox_rows.append(row)
    for row in adjusted_rows:
        row = {"model": "adjusted", **row}
        cox_rows.append(row)
    write_csv(outdir / f"{DATASET_ID}_cox_results.csv", pd.DataFrame(cox_rows))

    gene_expr_program.to_csv(outdir / f"{DATASET_ID}_signature_gene_expression_matrix.csv")

    summary = {
        "dataset_id": DATASET_ID,
        "n_samples_total": int(merged.shape[0]),
        "n_samples_survival": int(survival_df.shape[0]),
        "n_samples_adjusted_model": int(adjusted_df.shape[0]),
        "n_signature_genes": len(program_genes),
        "n_signature_genes_available": len(available_program_genes),
        "available_signature_genes": available_program_genes,
        "score_mean": float(merged["suppressive_myeloid_score"].mean()),
        "score_std": float(merged["suppressive_myeloid_score"].std(ddof=0)),
        "score_median": float(merged["suppressive_myeloid_score"].median()),
        "death_vs_alive_mannwhitney_p": float(death_mwu.pvalue),
        "death_group_mean_scores": {
            "alive": float(merged.loc[merged["os_event"] == 0, "suppressive_myeloid_score"].mean()),
            "dead": float(merged.loc[merged["os_event"] == 1, "suppressive_myeloid_score"].mean()),
        },
        "high_low_logrank": {
            "statistic": float(logrank_stat),
            "p_value": float(logrank_p),
            "high_group_n": int(high_df.shape[0]),
            "low_group_n": int(low_df.shape[0]),
            "high_group_60m_survival": surv_prob_at(high_sf, 60),
            "low_group_60m_survival": surv_prob_at(low_sf, 60),
        },
        "cox_univariate": univ_rows,
        "cox_adjusted": adjusted_rows,
    }
    (outdir / f"{DATASET_ID}_survival_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
