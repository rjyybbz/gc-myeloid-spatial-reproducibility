from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUND2_ROOT = PROJECT_ROOT
OUT_DIR = ROUND2_ROOT / "supplementary_tables"
PERITONEAL_DIR = PROJECT_ROOT / "results" / "peritoneal_state_bias"


def fmt_float(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 100:
        return f"{value:.3g}"
    if abs(value) >= 1:
        return f"{value:.4g}"
    return f"{value:.4g}"


def summarize_group(values: pd.Series) -> tuple[int, float, float]:
    return int(values.notna().sum()), float(values.median()), float(values.mean())


def mann_whitney(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    if len(a) == 0 or len(b) == 0:
        return math.nan
    return float(mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROUND2_ROOT / "05_revision_analysis").mkdir(parents=True, exist_ok=True)
    means = pd.read_csv(PERITONEAL_DIR / "GSE183904_peritoneal_sample_means.csv")
    fractions = pd.read_csv(PERITONEAL_DIR / "GSE183904_peritoneal_sample_state_fractions.csv")

    pivot = fractions.pivot_table(
        index=["sample_id", "tissue_group"],
        columns="mechanism_state",
        values=["n_cells", "fraction_within_sample"],
        fill_value=0,
        aggfunc="sum",
    )
    pivot.columns = [f"{state}_{metric}" for metric, state in pivot.columns]
    pivot = pivot.reset_index()

    merged = means.merge(pivot, on=["sample_id", "tissue_group"], how="left")
    merged["terminal_to_early_ratio"] = (
        (merged["suppressive_like_n_cells"].astype(float) + 0.5)
        / (merged["early_like_n_cells"].astype(float) + 0.5)
    )
    merged["log2_terminal_to_early_ratio"] = merged["terminal_to_early_ratio"].map(lambda x: math.log2(x))

    merged.to_csv(OUT_DIR / "Supplementary_Table_3_GSE183904_sample_level_peritoneal_metrics.csv", index=False)

    comparisons = [
        ("receptor_switch_score", "Per-sample mean receptor-switching score"),
        ("early_like_fraction_within_sample", "Early-like state fraction"),
        ("intermediate_like_fraction_within_sample", "Intermediate-like state fraction"),
        ("suppressive_like_fraction_within_sample", "Terminal-like suppressive-like state fraction"),
        ("terminal_to_early_ratio", "Terminal-like-to-early-like cell-count ratio with 0.5 pseudocount"),
        ("log2_terminal_to_early_ratio", "log2 terminal-like-to-early-like ratio"),
        ("terminal_receptor_switch_index", "Terminal-state receptor-switching index"),
        ("suppressive_program_score", "Per-sample mean suppressive-program score"),
        ("trajectory_axis_score", "Per-sample mean trajectory-axis score"),
    ]
    rows = []
    for metric, description in comparisons:
        peri = merged.loc[merged["tissue_group"] == "peritoneum_tumor", metric]
        stomach = merged.loc[merged["tissue_group"] == "stomach_tumor", metric]
        n_peri, median_peri, mean_peri = summarize_group(peri)
        n_stomach, median_stomach, mean_stomach = summarize_group(stomach)
        rows.append(
            {
                "metric": metric,
                "description": description,
                "n_peritoneum_tumor_samples": n_peri,
                "n_stomach_tumor_samples": n_stomach,
                "median_peritoneum_tumor": median_peri,
                "median_stomach_tumor": median_stomach,
                "mean_peritoneum_tumor": mean_peri,
                "mean_stomach_tumor": mean_stomach,
                "mann_whitney_p_two_sided": mann_whitney(peri, stomach),
                "primary_inference_level": "sample",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "Supplementary_Table_3_GSE183904_sample_level_peritoneal_comparisons.csv", index=False)

    narrative = [
        "# Round 2 Sample-Level Peritoneal Summary",
        "",
        "GSE183904 peritoneal analyses were re-summarized at the sample level to avoid treating cells as independent biological replicates.",
        "The biological replicate unit is the sample: 3 peritoneum_tumor samples and 26 stomach_tumor samples.",
        "",
        "| Metric | Peritoneum median | Stomach median | P value |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        narrative.append(
            f"| {row['description']} | {fmt_float(row['median_peritoneum_tumor'])} | "
            f"{fmt_float(row['median_stomach_tumor'])} | {fmt_float(row['mann_whitney_p_two_sided'])} |"
        )
    (ROUND2_ROOT / "05_revision_analysis" / "round2_sample_level_peritoneal_summary.md").write_text(
        "\n".join(narrative) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
