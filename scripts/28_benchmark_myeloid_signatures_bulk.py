from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from project_setup import PROJECT_PATHS, main as setup_main


SIGNATURES = {
    "our_suppressive_axis": None,
    "trajectory_polarized_axis": None,
    "spp1_tam": ["SPP1", "APOE", "APOC1", "C1QA", "C1QB", "C1QC", "TREM2", "GPNMB", "LGALS3", "CTSB", "CTSD", "FABP5"],
    "trem2_tam": ["TREM2", "APOE", "APOC1", "C1QA", "C1QB", "C1QC", "TYROBP", "LIPA", "GPNMB", "CD68"],
    "m2_macrophage": ["CD163", "MRC1", "MSR1", "CCL18", "IL10", "TGFB1", "STAB1", "F13A1", "FOLR2", "SEPP1"],
    "mdsc_like": ["S100A8", "S100A9", "FCN1", "VCAN", "IL1B", "CXCL8", "TIMP1", "NAMPT", "THBS1", "EREG"],
}


def read_our_signature() -> list[str]:
    return (
        pd.read_csv(PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv")["gene"]
        .dropna()
        .astype(str)
        .tolist()
    )


def read_trajectory_signature() -> dict[str, list[str]]:
    summary = json.loads(
        (PROJECT_PATHS["results"] / "mechanism_trajectory" / "GSE206785_trajectory_summary.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "positive": summary["top_late_up_genes"][:30],
        "negative": summary["top_early_up_genes"][:20],
    }


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


def parse_characteristics(cell: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in str(cell).split(" | "):
        if ": " not in part:
            continue
        key, value = part.split(": ", 1)
        result[key.strip().lower().replace(" ", "_")] = value.strip()
    return result


def read_gse84437_clinical() -> pd.DataFrame:
    path = PROJECT_PATHS["metadata"] / "sample_manifests" / "GSE84437_sample_manifest.csv"
    df = pd.read_csv(path)
    parsed = df["characteristics_ch1"].apply(parse_characteristics)
    clinical = pd.DataFrame(parsed.tolist())
    clinical.insert(0, "sample_id", df["gsm_id"].astype(str))
    clinical["os_event"] = pd.to_numeric(clinical.get("death"), errors="coerce")
    clinical["os_months"] = pd.to_numeric(clinical.get("duration_overall_survival"), errors="coerce")
    clinical["age"] = pd.to_numeric(clinical.get("age"), errors="coerce")
    return clinical


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


def read_tcga_clinical() -> pd.DataFrame:
    base = PROJECT_PATHS["data_raw"] / "TCGA_STAD"
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
    merged["sample_id"] = merged["SAMPLE_ID"].astype(str)
    merged["os_event"] = merged["OS_STATUS"].astype(str).str.startswith("1:").astype(float)
    merged["os_months"] = pd.to_numeric(merged["OS_MONTHS"], errors="coerce")
    merged["age"] = pd.to_numeric(merged["AGE"], errors="coerce")
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


def compute_signature_scores(gene_expr: pd.DataFrame, signatures: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    zexpr = zscore_rows(gene_expr)
    score_rows = {}
    availability_rows = []
    for name, genes in signatures.items():
        if isinstance(genes, dict):
            positive = [gene for gene in genes.get("positive", []) if gene in zexpr.index]
            negative = [gene for gene in genes.get("negative", []) if gene in zexpr.index]
            availability_rows.extend(
                {
                    "signature_name": name,
                    "gene": gene,
                    "direction": "positive",
                    "available_in_dataset": gene in positive,
                }
                for gene in genes.get("positive", [])
            )
            availability_rows.extend(
                {
                    "signature_name": name,
                    "gene": gene,
                    "direction": "negative",
                    "available_in_dataset": gene in negative,
                }
                for gene in genes.get("negative", [])
            )
            if not positive and not negative:
                continue
            pos_score = zexpr.loc[positive].mean(axis=0) if positive else 0.0
            neg_score = zexpr.loc[negative].mean(axis=0) if negative else 0.0
            score_rows[name] = pos_score - neg_score
            continue

        available = [gene for gene in genes if gene in zexpr.index]
        availability_rows.extend(
            {
                "signature_name": name,
                "gene": gene,
                "direction": "positive",
                "available_in_dataset": gene in available,
            }
            for gene in genes
        )
        if not available:
            continue
        score_rows[name] = zexpr.loc[available].mean(axis=0)
    scores = pd.DataFrame(score_rows)
    scores.index.name = "sample_id"
    return scores, pd.DataFrame(availability_rows)


def analyze_dataset(
    dataset_id: str,
    gene_expr: pd.DataFrame,
    clinical: pd.DataFrame,
    adjusted_formula: str,
    adjusted_required_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signatures = {**SIGNATURES}
    signatures["our_suppressive_axis"] = read_our_signature()
    signatures["trajectory_polarized_axis"] = read_trajectory_signature()
    score_df, availability_df = compute_signature_scores(gene_expr, signatures)
    score_df = score_df.reset_index()
    merged = clinical.merge(score_df, on="sample_id", how="inner")

    survival_rows = []
    pairwise_rows = []
    correlation_rows = []
    score_cols = [col for col in score_df.columns if col != "sample_id"]

    for sig in score_cols:
        sig_std = f"{sig}_std"
        merged[sig_std] = (merged[sig] - merged[sig].mean()) / merged[sig].std(ddof=0)
        merged[f"{sig}_group"] = np.where(merged[sig] >= merged[sig].median(), "high", "low")
        survival_df = merged.dropna(subset=["os_months", "os_event", sig_std]).copy()
        if survival_df.empty:
            continue
        univ = fit_cox(f"os_months ~ {sig_std}", survival_df)
        adjusted_df = merged.dropna(subset=["os_months", "os_event", sig_std] + adjusted_required_cols).copy()
        adjusted = fit_cox(adjusted_formula.format(sig_std=sig_std), adjusted_df)
        high_df = survival_df.loc[survival_df[f"{sig}_group"] == "high"]
        low_df = survival_df.loc[survival_df[f"{sig}_group"] == "low"]
        logrank_stat, logrank_p = survdiff(
            survival_df["os_months"].to_numpy(),
            survival_df["os_event"].to_numpy(),
            (survival_df[f"{sig}_group"] == "high").astype(int).to_numpy(),
        )
        high_sf = SurvfuncRight(high_df["os_months"].to_numpy(), high_df["os_event"].to_numpy())
        low_sf = SurvfuncRight(low_df["os_months"].to_numpy(), low_df["os_event"].to_numpy())
        survival_rows.append(
            {
                "dataset_id": dataset_id,
                "signature_name": sig,
                "n_samples_survival": int(survival_df.shape[0]),
                "univ_hr": univ[0]["hazard_ratio"] if univ else np.nan,
                "univ_p": univ[0]["p_value"] if univ else np.nan,
                "adjusted_hr": adjusted[0]["hazard_ratio"] if adjusted else np.nan,
                "adjusted_p": adjusted[0]["p_value"] if adjusted else np.nan,
                "logrank_stat": float(logrank_stat),
                "logrank_p": float(logrank_p),
                "high_60m_survival": surv_prob_at(high_sf, 60),
                "low_60m_survival": surv_prob_at(low_sf, 60),
                "score_mean": float(merged[sig].mean()),
                "score_std": float(merged[sig].std(ddof=0)),
            }
        )

    for i, sig_a in enumerate(score_cols):
        for sig_b in score_cols[i + 1 :]:
            corr = spearmanr(merged[sig_a], merged[sig_b], nan_policy="omit")
            correlation_rows.append(
                {
                    "dataset_id": dataset_id,
                    "signature_a": sig_a,
                    "signature_b": sig_b,
                    "rho": float(corr.statistic),
                    "p_value": float(corr.pvalue),
                }
            )

    if "our_suppressive_axis" in score_cols:
        other_sigs = [sig for sig in score_cols if sig != "our_suppressive_axis"]
        for sig in other_sigs:
            pair_df = merged.dropna(subset=["os_months", "os_event", "our_suppressive_axis", sig]).copy()
            if pair_df.empty:
                continue
            pair_df["our_suppressive_axis_std"] = (
                pair_df["our_suppressive_axis"] - pair_df["our_suppressive_axis"].mean()
            ) / pair_df["our_suppressive_axis"].std(ddof=0)
            pair_df[f"{sig}_std"] = (pair_df[sig] - pair_df[sig].mean()) / pair_df[sig].std(ddof=0)
            pair_formula = f"os_months ~ our_suppressive_axis_std + {sig}_std"
            pair_rows = fit_cox(pair_formula, pair_df)
            for row in pair_rows:
                pairwise_rows.append({"dataset_id": dataset_id, "comparator_signature": sig, **row})

    survival_df = pd.DataFrame(survival_rows).sort_values(["dataset_id", "univ_p"], na_position="last")
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["dataset_id", "comparator_signature", "p_value"], na_position="last")
    corr_df = pd.DataFrame(correlation_rows).sort_values(["dataset_id", "rho"], ascending=[True, False])
    availability_df.insert(0, "dataset_id", dataset_id)
    score_export_cols = ["sample_id", "os_event", "os_months"] + [col for col in score_cols if col in merged.columns]
    for col in ["age", "sex", "ptstage", "pnstage", "SEX", "broad_stage"]:
        if col in merged.columns:
            score_export_cols.append(col)
    score_table = merged[score_export_cols].copy()
    score_table.insert(0, "dataset_id", dataset_id)
    return survival_df, pairwise_df, corr_df, availability_df, score_table


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    setup_main()
    outdir = PROJECT_PATHS["results"] / "signature_benchmark"
    outdir.mkdir(parents=True, exist_ok=True)

    gse_surv, gse_pair, gse_corr, gse_avail, gse_scores = analyze_dataset(
        "GSE84437",
        read_gse84437_expression(),
        read_gse84437_clinical(),
        "os_months ~ {sig_std} + age + C(sex) + C(ptstage) + C(pnstage)",
        ["age", "sex", "ptstage", "pnstage"],
    )
    tcga_surv, tcga_pair, tcga_corr, tcga_avail, tcga_scores = analyze_dataset(
        "TCGA_STAD",
        read_tcga_expression(),
        read_tcga_clinical(),
        "os_months ~ {sig_std} + age + C(SEX) + C(broad_stage)",
        ["age", "SEX", "broad_stage"],
    )

    survival_df = pd.concat([gse_surv, tcga_surv], ignore_index=True)
    pairwise_df = pd.concat([gse_pair, tcga_pair], ignore_index=True)
    corr_df = pd.concat([gse_corr, tcga_corr], ignore_index=True)
    availability_df = pd.concat([gse_avail, tcga_avail], ignore_index=True)
    score_tables = pd.concat([gse_scores, tcga_scores], ignore_index=True)

    write_csv(outdir / "bulk_signature_benchmark_survival.csv", survival_df)
    write_csv(outdir / "bulk_signature_benchmark_pairwise_cox.csv", pairwise_df)
    write_csv(outdir / "bulk_signature_benchmark_correlations.csv", corr_df)
    write_csv(outdir / "bulk_signature_benchmark_gene_availability.csv", availability_df)
    write_csv(outdir / "bulk_signature_benchmark_scores.csv", score_tables)

    summary = {
        "datasets": survival_df["dataset_id"].dropna().unique().tolist(),
        "signatures": survival_df["signature_name"].dropna().unique().tolist(),
        "top_survival_signals": survival_df.sort_values("univ_p", na_position="last").head(12).to_dict(orient="records"),
        "top_pairwise_independence_signals": pairwise_df.sort_values("p_value", na_position="last").head(12).to_dict(orient="records"),
    }
    (outdir / "bulk_signature_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
