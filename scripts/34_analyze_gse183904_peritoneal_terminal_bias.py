from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

from project_setup import PROJECT_PATHS, main as setup_main


TARGET_TISSUES = ["peritoneum_tumor", "stomach_tumor"]
STATE_ORDER = ["early_like", "intermediate_like", "suppressive_like"]


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
    if len(vals_a) < 2 or len(vals_b) < 2:
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


def main() -> None:
    setup_main()
    scores = pd.read_csv(
        PROJECT_PATHS["results"] / "myeloid_external_mechanism" / "GSE183904_mechanism_umap_scores.csv"
    )
    scores = scores.loc[scores["tissue_group"].isin(TARGET_TISSUES)].copy()

    state_counts = (
        scores.groupby(["tissue_group", "mechanism_state"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = state_counts.groupby("tissue_group", observed=True)["n_cells"].transform("sum")
    state_counts["fraction_within_tissue"] = state_counts["n_cells"] / totals
    state_counts["mechanism_state"] = pd.Categorical(state_counts["mechanism_state"], categories=STATE_ORDER, ordered=True)
    state_counts = state_counts.sort_values(["tissue_group", "mechanism_state"])

    suppressive_vs_early = scores.loc[scores["mechanism_state"].isin(["suppressive_like", "early_like"])].copy()
    contingency = pd.crosstab(suppressive_vs_early["tissue_group"], suppressive_vs_early["mechanism_state"]).reindex(
        index=TARGET_TISSUES,
        columns=["suppressive_like", "early_like"],
        fill_value=0,
    )
    odds_ratio, fisher_pvalue = fisher_exact(contingency.to_numpy())
    contingency_df = contingency.reset_index().rename_axis(None, axis=1)

    sample_means = (
        scores.groupby(["sample_id", "tissue_group"], observed=True)[
            ["trajectory_axis_score", "receptor_switch_score", "suppressive_program_score", "formation_input_score"]
        ]
        .mean()
        .reset_index()
    )
    sample_means["trajectory_axis_score_z"] = (
        sample_means["trajectory_axis_score"] - sample_means["trajectory_axis_score"].mean()
    ) / sample_means["trajectory_axis_score"].std(ddof=0)
    sample_means["receptor_switch_score_z"] = (
        sample_means["receptor_switch_score"] - sample_means["receptor_switch_score"].mean()
    ) / sample_means["receptor_switch_score"].std(ddof=0)
    sample_means["terminal_receptor_switch_index"] = sample_means[
        ["trajectory_axis_score_z", "receptor_switch_score_z"]
    ].mean(axis=1)

    sample_tests = pd.DataFrame(
        [
            safe_mwu(sample_means, "terminal_receptor_switch_index", "tissue_group"),
            safe_mwu(sample_means, "receptor_switch_score", "tissue_group"),
            safe_mwu(sample_means, "trajectory_axis_score", "tissue_group"),
            safe_mwu(sample_means, "suppressive_program_score", "tissue_group"),
            safe_mwu(sample_means, "formation_input_score", "tissue_group"),
        ]
    ).sort_values("pvalue", na_position="last")

    sample_state = (
        scores.groupby(["sample_id", "tissue_group", "mechanism_state"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    sample_totals = sample_state.groupby(["sample_id", "tissue_group"], observed=True)["n_cells"].transform("sum")
    sample_state["fraction_within_sample"] = sample_state["n_cells"] / sample_totals

    outdir = PROJECT_PATHS["results"] / "peritoneal_state_bias"
    write_csv(outdir / "GSE183904_peritoneal_state_counts.csv", state_counts)
    write_csv(outdir / "GSE183904_peritoneal_suppressive_vs_early_table.csv", contingency_df)
    write_csv(outdir / "GSE183904_peritoneal_sample_means.csv", sample_means)
    write_csv(outdir / "GSE183904_peritoneal_sample_state_fractions.csv", sample_state)
    write_csv(outdir / "GSE183904_peritoneal_sample_tests.csv", sample_tests)

    summary = {
        "dataset_id": "GSE183904",
        "analysis_module": "peritoneal_terminal_receptor_switched_bias",
        "n_cells_tissue_subset": int(len(scores)),
        "n_samples_by_tissue": sample_means["tissue_group"].value_counts().to_dict(),
        "state_counts": state_counts.to_dict(orient="records"),
        "suppressive_vs_early_table": contingency_df.to_dict(orient="records"),
        "suppressive_vs_early_odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else None,
        "suppressive_vs_early_fisher_pvalue": float(fisher_pvalue),
        "sample_tests": sample_tests.to_dict(orient="records"),
        "interpretation": [
            "Peritoneal tumor myeloid cells are shifted toward a more suppressive-like versus early-like state balance at the cell level.",
            "At the sample level, peritoneal tumors are most clearly distinguished by stronger receptor switching rather than simply higher suppressive-program burden.",
            "This supports writing the metastatic layer as terminal receptor-switched ecology, not as a generic increase in suppressive burden alone.",
        ],
    }
    summary = json_safe(summary)
    (outdir / "GSE183904_peritoneal_state_bias_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
