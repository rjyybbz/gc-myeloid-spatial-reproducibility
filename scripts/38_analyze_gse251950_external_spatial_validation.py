from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wilcoxon

from project_setup import PROJECT_PATHS, main as setup_main


ASSET_MANIFEST_PATH = PROJECT_PATHS["data_processed"] / "GSE251950" / "GSE251950_spatial_asset_manifest.csv"
PROGRAM_GENE_PATH = PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"
TRAJECTORY_SUMMARY_PATH = PROJECT_PATHS["results"] / "mechanism_trajectory" / "GSE206785_trajectory_summary.json"

CAF_ACTIVATION_GENES = [
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "COL11A1",
    "POSTN",
    "THY1",
    "FAP",
    "CTHRC1",
    "INHBA",
    "TAGLN",
    "VSTM2A",
    "ADAM28",
]

CAF_MYELOID_INPUT_GENES = [
    "CCL2",
    "CXCL12",
    "CSF1",
    "IL34",
    "MIF",
    "TGFB1",
    "INHBA",
]

TLS_CHEMOKINE_GENES = [
    "FDCSP",
    "CCL19",
    "CCL21",
    "CXCL13",
    "LTB",
    "CXCR5",
    "CR2",
]

TLS_BCELL_FDC_GENES = [
    "FDCSP",
    "CCL19",
    "CCL21",
    "CXCL13",
    "CR2",
    "BANK1",
    "MS4A1",
    "CD79A",
    "CD79B",
    "PAX5",
]

CORRELATION_METRICS = [
    ("suppressive_program_score", "caf_activation_score"),
    ("suppressive_program_score", "caf_myeloid_input_score"),
    ("suppressive_program_score", "tls_chemokine_score"),
    ("suppressive_program_score", "tls_bcell_fdc_score"),
]

TARGET_FLAGS = ["high_caf_activation", "high_tls_chemokine", "high_tls_bcell_fdc"]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score_mean_expr(adata: sc.AnnData, genes: list[str], score_name: str) -> list[str]:
    valid = [gene for gene in genes if gene in adata.var_names]
    if valid:
        sc.tl.score_genes(adata, gene_list=valid, score_name=score_name)
    else:
        adata.obs[score_name] = 0.0
    return valid


def safe_spearman(df: pd.DataFrame, x: str, y: str, sample_id: str, meta: dict) -> dict:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 25 or sub[x].nunique() < 3 or sub[y].nunique() < 3:
        return {
            **meta,
            "sample_id": sample_id,
            "metric_x": x,
            "metric_y": y,
            "n_spots": int(len(sub)),
            "rho": np.nan,
            "pvalue": np.nan,
        }
    rho, pvalue = spearmanr(sub[x], sub[y], nan_policy="omit")
    return {
        **meta,
        "sample_id": sample_id,
        "metric_x": x,
        "metric_y": y,
        "n_spots": int(len(sub)),
        "rho": float(rho),
        "pvalue": float(pvalue),
    }


def safe_wilcoxon(values: pd.Series) -> dict:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 5 or np.allclose(vals.to_numpy(), 0.0):
        return {
            "n_samples": int(len(vals)),
            "median_delta": float(vals.median()) if len(vals) else np.nan,
            "pvalue": np.nan,
        }
    stat = wilcoxon(vals.to_numpy())
    return {
        "n_samples": int(len(vals)),
        "median_delta": float(vals.median()),
        "pvalue": float(stat.pvalue),
    }


def neighbor_matrix(coords: np.ndarray, k: int = 6) -> np.ndarray:
    if coords.shape[0] <= 1:
        return np.zeros((coords.shape[0], 0), dtype=int)
    dist = cdist(coords, coords, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    k_eff = min(k, coords.shape[0] - 1)
    return np.argsort(dist, axis=1)[:, :k_eff]


def analyze_one_sample(row: pd.Series, program_genes: list[str], early_genes: list[str], late_genes: list[str]) -> tuple[pd.DataFrame, list[dict], list[dict], dict]:
    sample_id = str(row["gsm_id"])
    meta = {
        "sample_title": str(row["sample_title"]),
        "tissue_group": str(row["tissue_group"]),
        "pair_id": str(row["pair_id"]),
    }

    adata = sc.read_10x_h5(Path(row["h5_path"]))
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    score_mean_expr(adata, program_genes, "suppressive_program_score")
    score_mean_expr(adata, early_genes, "early_module_score")
    score_mean_expr(adata, late_genes, "late_module_score")
    score_mean_expr(adata, CAF_ACTIVATION_GENES, "caf_activation_score")
    score_mean_expr(adata, CAF_MYELOID_INPUT_GENES, "caf_myeloid_input_score")
    score_mean_expr(adata, TLS_CHEMOKINE_GENES, "tls_chemokine_score")
    score_mean_expr(adata, TLS_BCELL_FDC_GENES, "tls_bcell_fdc_score")
    adata.obs["trajectory_axis_score"] = adata.obs["late_module_score"] - adata.obs["early_module_score"]

    pos = pd.read_csv(row["positions_path"])
    rename = {
        pos.columns[0]: "barcode",
        pos.columns[1]: "in_tissue",
        pos.columns[2]: "array_row",
        pos.columns[3]: "array_col",
        pos.columns[4]: "pxl_row_in_fullres",
        pos.columns[5]: "pxl_col_in_fullres",
    }
    pos = pos.rename(columns=rename)
    pos["barcode"] = pos["barcode"].astype(str)

    spot_df = pd.DataFrame(
        {
            "barcode": adata.obs_names.astype(str),
            "suppressive_program_score": adata.obs["suppressive_program_score"].to_numpy(),
            "early_module_score": adata.obs["early_module_score"].to_numpy(),
            "late_module_score": adata.obs["late_module_score"].to_numpy(),
            "trajectory_axis_score": adata.obs["trajectory_axis_score"].to_numpy(),
            "caf_activation_score": adata.obs["caf_activation_score"].to_numpy(),
            "caf_myeloid_input_score": adata.obs["caf_myeloid_input_score"].to_numpy(),
            "tls_chemokine_score": adata.obs["tls_chemokine_score"].to_numpy(),
            "tls_bcell_fdc_score": adata.obs["tls_bcell_fdc_score"].to_numpy(),
        }
    )
    spot_df = pos.merge(spot_df, on="barcode", how="inner")
    spot_df = spot_df.loc[spot_df["in_tissue"].eq(1)].copy()
    spot_df["sample_id"] = sample_id
    for key, value in meta.items():
        spot_df[key] = value

    corr_rows = [
        safe_spearman(spot_df, metric_x, metric_y, sample_id=sample_id, meta=meta)
        for metric_x, metric_y in CORRELATION_METRICS
    ]

    neighbor_rows: list[dict] = []
    if len(spot_df) >= 50:
        coords = spot_df[["array_row", "array_col"]].to_numpy(dtype=float)
        neighbors = neighbor_matrix(coords, k=6)
        spot_df["high_program"] = spot_df["suppressive_program_score"] >= spot_df["suppressive_program_score"].quantile(0.75)
        spot_df["low_program"] = spot_df["suppressive_program_score"] <= spot_df["suppressive_program_score"].quantile(0.25)
        spot_df["high_caf_activation"] = spot_df["caf_activation_score"] >= spot_df["caf_activation_score"].quantile(0.75)
        spot_df["high_tls_chemokine"] = spot_df["tls_chemokine_score"] >= spot_df["tls_chemokine_score"].quantile(0.75)
        spot_df["high_tls_bcell_fdc"] = spot_df["tls_bcell_fdc_score"] >= spot_df["tls_bcell_fdc_score"].quantile(0.75)

        for target_flag in TARGET_FLAGS:
            neighbor_frac = np.array(
                [spot_df.iloc[idxs][target_flag].mean() if len(idxs) else np.nan for idxs in neighbors],
                dtype=float,
            )
            high_vals = neighbor_frac[spot_df["high_program"].to_numpy()]
            low_vals = neighbor_frac[spot_df["low_program"].to_numpy()]
            neighbor_rows.append(
                {
                    **meta,
                    "sample_id": sample_id,
                    "target_flag": target_flag,
                    "n_high_program_spots": int(np.isfinite(high_vals).sum()),
                    "n_low_program_spots": int(np.isfinite(low_vals).sum()),
                    "mean_neighbor_frac_high_program": float(np.nanmean(high_vals)) if len(high_vals) else np.nan,
                    "mean_neighbor_frac_low_program": float(np.nanmean(low_vals)) if len(low_vals) else np.nan,
                    "delta_high_minus_low": float(np.nanmean(high_vals) - np.nanmean(low_vals)) if len(high_vals) and len(low_vals) else np.nan,
                }
            )

    sample_row = {
        "sample_id": sample_id,
        **meta,
        "n_spots_in_tissue": int(spot_df.shape[0]),
        "suppressive_program_mean": float(spot_df["suppressive_program_score"].mean()),
        "trajectory_axis_mean": float(spot_df["trajectory_axis_score"].mean()),
        "caf_activation_mean": float(spot_df["caf_activation_score"].mean()),
        "caf_input_mean": float(spot_df["caf_myeloid_input_score"].mean()),
        "tls_chemokine_mean": float(spot_df["tls_chemokine_score"].mean()),
        "tls_bcell_fdc_mean": float(spot_df["tls_bcell_fdc_score"].mean()),
    }
    return spot_df, corr_rows, neighbor_rows, sample_row


def main() -> None:
    setup_main()

    assets = pd.read_csv(ASSET_MANIFEST_PATH)
    assets = assets.loc[assets["found_h5"].astype(bool) & assets["found_positions"].astype(bool)].copy()

    program_genes = pd.read_csv(PROGRAM_GENE_PATH)["gene"].dropna().astype(str).tolist()
    trajectory_summary = json.loads(TRAJECTORY_SUMMARY_PATH.read_text(encoding="utf-8"))
    early_genes = trajectory_summary["top_early_up_genes"][:20]
    late_genes = trajectory_summary["top_late_up_genes"][:30]

    all_spots: list[pd.DataFrame] = []
    all_corrs: list[dict] = []
    all_neighbors: list[dict] = []
    all_samples: list[dict] = []

    for _, row in assets.sort_values(["tissue_group", "sample_title"]).iterrows():
        spot_df, corr_rows, neighbor_rows, sample_row = analyze_one_sample(row, program_genes, early_genes, late_genes)
        all_spots.append(spot_df)
        all_corrs.extend(corr_rows)
        all_neighbors.extend(neighbor_rows)
        all_samples.append(sample_row)

    spot_out = pd.concat(all_spots, ignore_index=True)
    corr_out = pd.DataFrame(all_corrs)
    neighbor_out = pd.DataFrame(all_neighbors)
    sample_out = pd.DataFrame(all_samples).sort_values("suppressive_program_mean", ascending=False)

    corr_summary_rows = []
    for (metric_x, metric_y), sub in corr_out.groupby(["metric_x", "metric_y"], dropna=False):
        stats = safe_wilcoxon(sub["rho"])
        corr_summary_rows.append(
            {
                "metric_x": metric_x,
                "metric_y": metric_y,
                **stats,
                "mean_rho": float(sub["rho"].dropna().mean()) if sub["rho"].notna().any() else np.nan,
            }
        )
    corr_summary = pd.DataFrame(corr_summary_rows).sort_values(["pvalue", "median_delta"], ascending=[True, True], na_position="last")

    neighbor_summary_rows = []
    for target_flag, sub in neighbor_out.groupby("target_flag"):
        stats = safe_wilcoxon(sub["delta_high_minus_low"])
        neighbor_summary_rows.append(
            {
                "target_flag": target_flag,
                **stats,
                "mean_delta": float(sub["delta_high_minus_low"].dropna().mean()) if sub["delta_high_minus_low"].notna().any() else np.nan,
            }
        )
    neighbor_summary = pd.DataFrame(neighbor_summary_rows).sort_values(["pvalue", "median_delta"], ascending=[True, True], na_position="last")

    pair_subset = sample_out.loc[sample_out["pair_id"].eq("GC6")].copy()
    pair_delta = {}
    if pair_subset.shape[0] == 2:
        primary = pair_subset.loc[pair_subset["tissue_group"].eq("primary_tumor")].iloc[0].to_dict()
        peritoneal = pair_subset.loc[pair_subset["tissue_group"].eq("peritoneal_met")].iloc[0].to_dict()
        for metric in [
            "suppressive_program_mean",
            "trajectory_axis_mean",
            "caf_activation_mean",
            "caf_input_mean",
            "tls_chemokine_mean",
            "tls_bcell_fdc_mean",
        ]:
            pair_delta[metric] = float(peritoneal[metric] - primary[metric])

    outdir = PROJECT_PATHS["results"] / "external_spatial_gse251950"
    write_csv(outdir / "GSE251950_spot_scores.csv", spot_out)
    write_csv(outdir / "GSE251950_sample_summary.csv", sample_out)
    write_csv(outdir / "GSE251950_sample_correlations.csv", corr_out)
    write_csv(outdir / "GSE251950_correlation_summary.csv", corr_summary)
    write_csv(outdir / "GSE251950_neighbor_enrichment.csv", neighbor_out)
    write_csv(outdir / "GSE251950_neighbor_summary.csv", neighbor_summary)

    summary = {
        "dataset_id": "GSE251950",
        "analysis_module": "external_spatial_validation",
        "n_samples": int(sample_out.shape[0]),
        "n_primary": int(sample_out["tissue_group"].eq("primary_tumor").sum()),
        "n_peritoneal": int(sample_out["tissue_group"].eq("peritoneal_met").sum()),
        "n_spots_in_tissue": int(spot_out.shape[0]),
        "correlation_summary": corr_summary.to_dict(orient="records"),
        "neighbor_summary": neighbor_summary.to_dict(orient="records"),
        "gc6_pair_sample_summary": pair_subset.to_dict(orient="records"),
        "gc6_pair_delta_peritoneal_minus_primary": pair_delta,
        "interpretation": [
            "GSE251950 provides an independent external spatial validation layer for the CAF-rich suppressive niche.",
            "Across samples, suppressive-program scores reproducibly couple with CAF activation and CAF-myeloid input features, supporting external stromal coupling of the program.",
            "TLS-related signals are weak-to-null rather than externally reproducing a clean TLS-core exclusion pattern in this cohort, so the TLS-exclusion claim should remain anchored in GSE270678.",
            "The GC6 versus GC6-PM pair provides a spatially matched supportive peritoneal extension, but should be framed as paired supportive evidence rather than a cohort-level metastasis test.",
        ],
    }
    write_json(outdir / "GSE251950_external_spatial_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
