from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight

from project_setup import PROJECT_PATHS, main as setup_main


SCORE_PATH = PROJECT_PATHS["results"] / "signature_benchmark" / "bulk_signature_benchmark_scores.csv"


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


def assign_quadrant(row: pd.Series, burden_med: float, polar_med: float) -> str:
    burden_group = "burden_high" if row["our_suppressive_axis"] >= burden_med else "burden_low"
    polar_group = "polar_high" if row["trajectory_polarized_axis"] >= polar_med else "polar_low"
    return f"{burden_group}__{polar_group}"


def analyze_dataset(df: pd.DataFrame, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["dataset_id"].eq(dataset_id)].copy()
    df = df.dropna(subset=["os_event", "os_months", "our_suppressive_axis", "trajectory_polarized_axis"]).copy()
    df["our_axis_std"] = (df["our_suppressive_axis"] - df["our_suppressive_axis"].mean()) / df["our_suppressive_axis"].std(ddof=0)
    df["trajectory_axis_std"] = (
        (df["trajectory_polarized_axis"] - df["trajectory_polarized_axis"].mean()) / df["trajectory_polarized_axis"].std(ddof=0)
    )

    burden_median = float(df["our_suppressive_axis"].median())
    polar_median = float(df["trajectory_polarized_axis"].median())
    df["dual_axis_group"] = df.apply(assign_quadrant, axis=1, args=(burden_median, polar_median))

    cox_rows = fit_cox("os_months ~ our_axis_std + trajectory_axis_std", df)
    for row in cox_rows:
        row["dataset_id"] = dataset_id

    group_rows = []
    for group_name, group_df in df.groupby("dual_axis_group"):
        sf = SurvfuncRight(group_df["os_months"].to_numpy(), group_df["os_event"].to_numpy())
        group_rows.append(
            {
                "dataset_id": dataset_id,
                "dual_axis_group": group_name,
                "n_samples": int(group_df.shape[0]),
                "mean_our_axis": float(group_df["our_suppressive_axis"].mean()),
                "mean_trajectory_axis": float(group_df["trajectory_polarized_axis"].mean()),
                "death_rate": float(group_df["os_event"].mean()),
                "median_os_months": float(group_df["os_months"].median()),
                "survival_60m": surv_prob_at(sf, 60),
            }
        )
    group_out = pd.DataFrame(group_rows)
    if not group_out.empty:
        group_out["survival_60m_rank"] = (
            group_out["survival_60m"].fillna(-1).rank(method="dense", ascending=True).astype(int)
        )
        group_out["death_rate_rank"] = group_out["death_rate"].rank(method="dense", ascending=False).astype(int)
    return pd.DataFrame(cox_rows), group_out


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    setup_main()
    scores = pd.read_csv(SCORE_PATH)
    all_cox = []
    all_groups = []
    for dataset_id in sorted(scores["dataset_id"].dropna().unique()):
        cox_df, group_df = analyze_dataset(scores, dataset_id)
        all_cox.append(cox_df)
        all_groups.append(group_df)

    cox_out = pd.concat(all_cox, ignore_index=True)
    group_out = pd.concat(all_groups, ignore_index=True)

    outdir = PROJECT_PATHS["results"] / "signature_benchmark"
    write_csv(outdir / "dual_axis_bulk_cox.csv", cox_out)
    write_csv(outdir / "dual_axis_bulk_group_summary.csv", group_out)

    dataset_summaries = []
    for dataset_id, dataset_group in group_out.groupby("dataset_id"):
        worst_survival_group = (
            dataset_group.sort_values(
                ["survival_60m_rank", "death_rate_rank", "median_os_months"],
                ascending=[True, True, True],
            )
            .iloc[0]
        )
        best_survival_group = (
            dataset_group.sort_values(
                ["survival_60m_rank", "death_rate_rank", "median_os_months"],
                ascending=[False, False, False],
            )
            .iloc[0]
        )
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "worst_group_by_survival_60m": str(worst_survival_group["dual_axis_group"]),
                "worst_group_survival_60m": None
                if pd.isna(worst_survival_group["survival_60m"])
                else float(worst_survival_group["survival_60m"]),
                "best_group_by_survival_60m": str(best_survival_group["dual_axis_group"]),
                "best_group_survival_60m": None
                if pd.isna(best_survival_group["survival_60m"])
                else float(best_survival_group["survival_60m"]),
                "worst_group_by_death_rate": str(
                    dataset_group.sort_values(["death_rate_rank", "survival_60m_rank"], ascending=[True, True]).iloc[0][
                        "dual_axis_group"
                    ]
                ),
            }
        )

    dataset_summary_out = pd.DataFrame(dataset_summaries)
    write_csv(outdir / "dual_axis_bulk_dataset_summary.csv", dataset_summary_out)

    shared_worst_groups = sorted(dataset_summary_out["worst_group_by_survival_60m"].dropna().unique().tolist())
    concordant_worst_group = shared_worst_groups[0] if len(shared_worst_groups) == 1 else None

    summary = {
        "dual_axis_cox": cox_out.to_dict(orient="records"),
        "dual_axis_group_summary": group_out.to_dict(orient="records"),
        "dual_axis_dataset_summary": dataset_summary_out.to_dict(orient="records"),
        "cross_dataset_interpretation": {
            "concordant_worst_group_by_survival_60m": concordant_worst_group,
            "all_worst_groups_by_survival_60m": shared_worst_groups,
        },
    }
    (outdir / "dual_axis_bulk_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
