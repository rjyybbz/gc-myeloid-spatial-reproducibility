from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from project_setup import PROJECT_PATHS, main as setup_main


DATASET_ID = "TCGA_STAD"


def read_program_genes() -> list[str]:
    path = PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"
    return pd.read_csv(path)["gene"].dropna().astype(str).tolist()


def read_expression() -> pd.DataFrame:
    path = PROJECT_PATHS["data_raw"] / DATASET_ID / "stad_tcga_pan_can_atlas_2018_data_mrna_seq_v2_rsem.txt"
    df = pd.read_csv(path, sep="\t")
    sample_cols = [c for c in df.columns if c not in {"Hugo_Symbol", "Entrez_Gene_Id"}]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    df["Hugo_Symbol"] = df["Hugo_Symbol"].fillna("").astype(str).str.strip()
    df = df.loc[df["Hugo_Symbol"] != ""].copy()
    gene_expr = df.groupby("Hugo_Symbol")[sample_cols].mean()
    gene_expr.index.name = "gene"
    return gene_expr


def read_clinical() -> pd.DataFrame:
    base = PROJECT_PATHS["data_raw"] / DATASET_ID
    patient = pd.read_csv(
        base / "stad_tcga_pan_can_atlas_2018_data_clinical_patient.txt",
        sep="\t",
        comment="#",
        skiprows=[1, 2, 3],
    )
    sample = pd.read_csv(
        base / "stad_tcga_pan_can_atlas_2018_data_clinical_sample.txt",
        sep="\t",
        comment="#",
        skiprows=[1, 2, 3],
    )
    merged = sample.merge(patient, on="PATIENT_ID", how="left", suffixes=("_sample", "_patient"))
    merged["os_event"] = merged["OS_STATUS"].astype(str).str.startswith("1:").astype(float)
    merged["os_months"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
    merged["age"] = pd.to_numeric(merged["AGE"], errors="coerce")
    merged["msi_sensor_score"] = pd.to_numeric(merged["MSI_SENSOR_SCORE"], errors="coerce")
    merged["tmb_nonsynonymous"] = pd.to_numeric(merged["TMB_NONSYNONYMOUS"], errors="coerce")
    stage_text = merged["AJCC_PATHOLOGIC_TUMOR_STAGE"].fillna("").astype(str).str.upper()
    merged["broad_stage"] = np.select(
        [
            stage_text.str.contains("STAGE IV", regex=False),
            stage_text.str.contains("STAGE III", regex=False),
            stage_text.str.contains("STAGE II", regex=False),
            stage_text.str.contains("STAGE I", regex=False),
        ],
        ["Stage IV", "Stage III", "Stage II", "Stage I"],
        default=np.nan,
    )
    return merged


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

    program_genes = read_program_genes()
    gene_expr = read_expression()
    clinical = read_clinical()

    available_program_genes = [g for g in program_genes if g in gene_expr.index]
    gene_expr_program = gene_expr.loc[available_program_genes].copy()
    gene_expr_program_z = zscore_rows(gene_expr_program)
    signature_score = gene_expr_program_z.mean(axis=0).rename("suppressive_myeloid_score")

    score_df = signature_score.reset_index().rename(columns={"index": "SAMPLE_ID"})
    merged = clinical.merge(score_df, on="SAMPLE_ID", how="inner")
    merged["score_group"] = np.where(
        merged["suppressive_myeloid_score"] >= merged["suppressive_myeloid_score"].median(),
        "high",
        "low",
    )
    merged["signature_score_std"] = (
        merged["suppressive_myeloid_score"] - merged["suppressive_myeloid_score"].mean()
    ) / merged["suppressive_myeloid_score"].std(ddof=0)

    survival_df = merged.dropna(subset=["os_months", "os_event", "signature_score_std"]).copy()
    adjusted_df = survival_df.dropna(subset=["age", "SEX", "broad_stage"]).copy()

    univ_rows = fit_cox("os_months ~ signature_score_std", survival_df)
    adjusted_rows = fit_cox(
        "os_months ~ signature_score_std + age + C(SEX) + C(broad_stage)",
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

    subtype_df = merged.dropna(subset=["SUBTYPE"]).copy()
    subtype_summary = (
        subtype_df.groupby("SUBTYPE")
        .agg(
            n_samples=("SAMPLE_ID", "count"),
            mean_score=("suppressive_myeloid_score", "mean"),
            median_score=("suppressive_myeloid_score", "median"),
            mean_msi_sensor=("msi_sensor_score", "mean"),
            mean_tmb=("tmb_nonsynonymous", "mean"),
        )
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )

    kruskal_stat, kruskal_p = kruskal(
        *[
            group["suppressive_myeloid_score"].to_numpy()
            for _, group in subtype_df.groupby("SUBTYPE")
            if group["suppressive_myeloid_score"].notna().sum() > 0
        ]
    )
    spearman_msi = spearmanr(merged["suppressive_myeloid_score"], merged["msi_sensor_score"], nan_policy="omit")
    spearman_tmb = spearmanr(merged["suppressive_myeloid_score"], merged["tmb_nonsynonymous"], nan_policy="omit")
    death_mwu = mannwhitneyu(
        merged.loc[merged["os_event"] == 1, "suppressive_myeloid_score"],
        merged.loc[merged["os_event"] == 0, "suppressive_myeloid_score"],
        alternative="two-sided",
    )
    ebv_mask = merged["SUBTYPE"].eq("STAD_EBV")
    ebv_mwu = mannwhitneyu(
        merged.loc[ebv_mask, "suppressive_myeloid_score"],
        merged.loc[~ebv_mask, "suppressive_myeloid_score"],
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

    export_cols = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "SUBTYPE",
        "SEX",
        "AGE",
        "AJCC_PATHOLOGIC_TUMOR_STAGE",
        "msi_sensor_score",
        "tmb_nonsynonymous",
        "OS_STATUS",
        "os_event",
        "os_months",
        "suppressive_myeloid_score",
        "signature_score_std",
        "score_group",
    ]
    write_csv(
        outdir / f"{DATASET_ID}_signature_scores.csv",
        merged[export_cols].sort_values("suppressive_myeloid_score", ascending=False),
    )
    write_csv(outdir / f"{DATASET_ID}_subtype_summary.csv", subtype_summary)

    cox_rows = []
    for row in univ_rows:
        cox_rows.append({"model": "univ_continuous", **row})
    for row in adjusted_rows:
        cox_rows.append({"model": "adjusted", **row})
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
        "subtype_kruskal": {
            "statistic": float(kruskal_stat),
            "p_value": float(kruskal_p),
        },
        "spearman_with_msi_sensor": {
            "rho": float(spearman_msi.statistic),
            "p_value": float(spearman_msi.pvalue),
        },
        "spearman_with_tmb": {
            "rho": float(spearman_tmb.statistic),
            "p_value": float(spearman_tmb.pvalue),
        },
        "ebv_vs_non_ebv": {
            "p_value": float(ebv_mwu.pvalue),
            "ebv_mean_score": float(merged.loc[ebv_mask, "suppressive_myeloid_score"].mean()),
            "non_ebv_mean_score": float(merged.loc[~ebv_mask, "suppressive_myeloid_score"].mean()),
        },
        "subtype_summary": subtype_summary.to_dict(orient="records"),
    }
    (outdir / f"{DATASET_ID}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
