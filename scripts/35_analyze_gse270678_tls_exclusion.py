from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wilcoxon

from project_setup import PROJECT_PATHS, main as setup_main


MANIFEST_PATH = PROJECT_PATHS["metadata"] / "sample_manifests" / "GSE270678_sample_manifest.csv"
H5_DIR = PROJECT_PATHS["data_raw"] / "GSE270678" / "spatial_h5"
SPATIAL_SPOT_PATH = PROJECT_PATHS["results"] / "spatial_niche" / "GSE270678_spatial_niche_spot_scores.csv"

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
    ("suppressive_program_score", "tls_chemokine_score"),
    ("suppressive_program_score", "tls_bcell_fdc_score"),
    ("trajectory_axis_score", "tls_chemokine_score"),
    ("trajectory_axis_score", "tls_bcell_fdc_score"),
]

TARGET_FLAGS = ["high_tls_chemokine", "high_tls_bcell_fdc"]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_title(title: str) -> dict:
    parts = [p.strip() for p in str(title).split(",")]
    return {
        "donor": parts[0] if len(parts) > 0 else "",
        "platform": parts[1] if len(parts) > 1 else "",
        "sample_type": parts[2] if len(parts) > 2 else "",
        "stage": parts[3] if len(parts) > 3 else "",
        "subtype": parts[4] if len(parts) > 4 else "",
        "location": parts[5] if len(parts) > 5 else "",
        "replicate": parts[6] if len(parts) > 6 else "",
    }


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


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    return mapped.fillna(False)


def nearest_distance(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    if source_coords.size == 0 or target_coords.size == 0:
        return np.array([], dtype=float)
    return cdist(source_coords, target_coords, metric="euclidean").min(axis=1)


def analyze_one_sample(h5_path: Path, manifest_row: pd.Series, base_sample: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict], dict]:
    sample_id = str(manifest_row["gsm_id"])
    meta = parse_title(str(manifest_row["title"]))

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    available_tls = score_mean_expr(adata, TLS_CHEMOKINE_GENES, "tls_chemokine_score")
    available_tls_bcell_fdc = score_mean_expr(adata, TLS_BCELL_FDC_GENES, "tls_bcell_fdc_score")

    new_scores = pd.DataFrame(
        {
            "barcode": adata.obs_names.astype(str),
            "tls_chemokine_score": adata.obs["tls_chemokine_score"].to_numpy(),
            "tls_bcell_fdc_score": adata.obs["tls_bcell_fdc_score"].to_numpy(),
        }
    )

    spot_df = base_sample.merge(new_scores, on="barcode", how="inner")
    spot_df["high_program"] = coerce_bool(spot_df["high_program"])
    spot_df["low_program"] = coerce_bool(spot_df["low_program"])
    spot_df["high_tls_chemokine"] = spot_df["tls_chemokine_score"] >= spot_df["tls_chemokine_score"].quantile(0.75)
    spot_df["high_tls_bcell_fdc"] = spot_df["tls_bcell_fdc_score"] >= spot_df["tls_bcell_fdc_score"].quantile(0.75)

    corr_rows = [
        safe_spearman(spot_df, metric_x, metric_y, sample_id=sample_id, meta=meta)
        for metric_x, metric_y in CORRELATION_METRICS
    ]

    neighbor_rows: list[dict] = []
    distance_rows: list[dict] = []
    if len(spot_df) >= 50:
        coords = spot_df[["array_row", "array_col"]].to_numpy(dtype=float)
        neighbors = neighbor_matrix(coords, k=6)
        for target_flag in TARGET_FLAGS:
            neighbor_frac = np.array(
                [spot_df.iloc[idxs][target_flag].mean() if len(idxs) else np.nan for idxs in neighbors],
                dtype=float,
            )
            spot_df[f"neighbor_{target_flag}_frac"] = neighbor_frac
            high_vals = spot_df.loc[spot_df["high_program"], f"neighbor_{target_flag}_frac"].dropna()
            low_vals = spot_df.loc[spot_df["low_program"], f"neighbor_{target_flag}_frac"].dropna()
            neighbor_rows.append(
                {
                    **meta,
                    "sample_id": sample_id,
                    "target_flag": target_flag,
                    "n_high_program_spots": int(high_vals.shape[0]),
                    "n_low_program_spots": int(low_vals.shape[0]),
                    "mean_neighbor_frac_high_program": float(high_vals.mean()) if len(high_vals) else np.nan,
                    "mean_neighbor_frac_low_program": float(low_vals.mean()) if len(low_vals) else np.nan,
                    "delta_high_minus_low": float(high_vals.mean() - low_vals.mean()) if len(high_vals) and len(low_vals) else np.nan,
                }
            )

            target_coords = coords[spot_df[target_flag].to_numpy()]
            high_program_coords = coords[spot_df["high_program"].to_numpy()]
            low_program_coords = coords[spot_df["low_program"].to_numpy()]
            high_dist = nearest_distance(high_program_coords, target_coords)
            low_dist = nearest_distance(low_program_coords, target_coords)
            distance_rows.append(
                {
                    **meta,
                    "sample_id": sample_id,
                    "target_flag": target_flag,
                    "n_target_spots": int(target_coords.shape[0]),
                    "n_high_program_spots": int(high_dist.shape[0]),
                    "n_low_program_spots": int(low_dist.shape[0]),
                    "median_distance_high_program": float(np.median(high_dist)) if len(high_dist) else np.nan,
                    "median_distance_low_program": float(np.median(low_dist)) if len(low_dist) else np.nan,
                    "delta_high_minus_low": float(np.median(high_dist) - np.median(low_dist)) if len(high_dist) and len(low_dist) else np.nan,
                }
            )

    sample_row = {
        "sample_id": sample_id,
        "n_spots_in_tissue": int(spot_df.shape[0]),
        "tls_chemokine_genes_available": int(len(available_tls)),
        "tls_bcell_fdc_genes_available": int(len(available_tls_bcell_fdc)),
        "tls_chemokine_mean": float(spot_df["tls_chemokine_score"].mean()),
        "tls_bcell_fdc_mean": float(spot_df["tls_bcell_fdc_score"].mean()),
        "high_tls_chemokine_frac": float(spot_df["high_tls_chemokine"].mean()),
        "high_tls_bcell_fdc_frac": float(spot_df["high_tls_bcell_fdc"].mean()),
        **meta,
    }
    return spot_df, corr_rows, neighbor_rows, distance_rows, sample_row


def summarize_delta_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    if not df.empty:
        for target_flag, sub in df.groupby("target_flag"):
            stats = safe_wilcoxon(sub[value_col])
            rows.append(
                {
                    "target_flag": target_flag,
                    **stats,
                    "mean_delta": float(sub[value_col].dropna().mean()) if sub[value_col].notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["pvalue", "median_delta"], ascending=[True, True], na_position="last")


def main() -> None:
    setup_main()

    warnings.filterwarnings("ignore", message="Variable names are not unique.*")
    warnings.filterwarnings("ignore", message="Some cells have zero counts")
    warnings.filterwarnings("ignore", message="Sample size too small for normal approximation.")

    base_spots = pd.read_csv(SPATIAL_SPOT_PATH, low_memory=False)
    manifest = pd.read_csv(MANIFEST_PATH)

    all_spots: list[pd.DataFrame] = []
    all_corrs: list[dict] = []
    all_neighbors: list[dict] = []
    all_distances: list[dict] = []
    all_samples: list[dict] = []

    for h5_path in sorted(H5_DIR.glob("*filtered_feature_bc_matrix.h5")):
        gsm_id = h5_path.name.split("_", 1)[0]
        manifest_row = manifest.loc[manifest["gsm_id"].eq(gsm_id)].iloc[0]
        base_sample = base_spots.loc[base_spots["sample_id"].eq(gsm_id)].copy()
        spot_df, corr_rows, neighbor_rows, distance_rows, sample_row = analyze_one_sample(h5_path, manifest_row, base_sample)
        all_spots.append(spot_df)
        all_corrs.extend(corr_rows)
        all_neighbors.extend(neighbor_rows)
        all_distances.extend(distance_rows)
        all_samples.append(sample_row)

    spot_out = pd.concat(all_spots, ignore_index=True)
    corr_out = pd.DataFrame(all_corrs)
    neighbor_out = pd.DataFrame(all_neighbors)
    distance_out = pd.DataFrame(all_distances)
    sample_out = pd.DataFrame(all_samples).sort_values("tls_bcell_fdc_mean", ascending=False)

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
    corr_summary = pd.DataFrame(corr_summary_rows).sort_values(
        ["pvalue", "median_delta"],
        ascending=[True, True],
        na_position="last",
    )

    neighbor_summary = summarize_delta_table(neighbor_out, "delta_high_minus_low")
    distance_summary = summarize_delta_table(distance_out, "delta_high_minus_low")

    location_summary = (
        sample_out.groupby("location")[["tls_chemokine_mean", "tls_bcell_fdc_mean"]]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    location_summary.columns = ["_".join(col).strip("_") for col in location_summary.columns]

    outdir = PROJECT_PATHS["results"] / "spatial_tls_exclusion"
    write_csv(outdir / "GSE270678_tls_exclusion_spot_scores.csv", spot_out)
    write_csv(outdir / "GSE270678_tls_exclusion_sample_correlations.csv", corr_out)
    write_csv(outdir / "GSE270678_tls_exclusion_correlation_summary.csv", corr_summary)
    write_csv(outdir / "GSE270678_tls_exclusion_neighbor_enrichment.csv", neighbor_out)
    write_csv(outdir / "GSE270678_tls_exclusion_neighbor_summary.csv", neighbor_summary)
    write_csv(outdir / "GSE270678_tls_exclusion_distance_enrichment.csv", distance_out)
    write_csv(outdir / "GSE270678_tls_exclusion_distance_summary.csv", distance_summary)
    write_csv(outdir / "GSE270678_tls_exclusion_sample_summary.csv", sample_out)
    write_csv(outdir / "GSE270678_tls_exclusion_location_summary.csv", location_summary)

    summary = {
        "dataset_id": "GSE270678",
        "analysis_module": "spatial_tls_exclusion",
        "n_samples": int(sample_out.shape[0]),
        "n_spots_in_tissue": int(spot_out.shape[0]),
        "tls_chemokine_genes": TLS_CHEMOKINE_GENES,
        "tls_bcell_fdc_genes": TLS_BCELL_FDC_GENES,
        "top_samples_by_tls_bcell_fdc_mean": sample_out.head(10).to_dict(orient="records"),
        "correlation_summary": corr_summary.to_dict(orient="records"),
        "neighbor_summary": neighbor_summary.to_dict(orient="records"),
        "distance_summary": distance_summary.to_dict(orient="records"),
        "top_locations_by_tls_bcell_fdc_mean": location_summary.sort_values("tls_bcell_fdc_mean_mean", ascending=False).to_dict(orient="records"),
        "interpretation": [
            "The suppressive myeloid program is expected to be spatially separated from TLS-core ecology, not merely from bulk T-cell abundance.",
            "Negative suppressive-program versus TLS chemokine and TLS B-cell/FDC-core correlations support within-sample antagonism between myeloid suppression and lymphoid aggregation.",
            "Depleted TLS-core neighbors around suppressive hotspots support a spatial immune-exclusion model centered on loss of lymphoid-aggregated ecology.",
        ],
    }
    write_json(outdir / "GSE270678_tls_exclusion_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
