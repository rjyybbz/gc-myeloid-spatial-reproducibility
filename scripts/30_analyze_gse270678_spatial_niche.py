from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wilcoxon

from project_setup import PROJECT_PATHS, main as setup_main


PROGRAM_GENE_PATH = PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"
TRAJECTORY_SUMMARY_PATH = PROJECT_PATHS["results"] / "mechanism_trajectory" / "GSE206785_trajectory_summary.json"
MANIFEST_PATH = PROJECT_PATHS["metadata"] / "sample_manifests" / "GSE270678_sample_manifest.csv"
H5_DIR = PROJECT_PATHS["data_raw"] / "GSE270678" / "spatial_h5"
POSITION_DIR = PROJECT_PATHS["data_raw"] / "GSE270678" / "tissue_positions"

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

T_CELL_GENES = ["CD3D", "CD3E", "TRAC", "TRBC1", "IL7R", "LTB", "NKG7"]
T_CELL_EXHAUSTION_GENES = ["HAVCR2", "LAG3", "PDCD1", "TIGIT", "CTLA4", "TOX", "CXCL13"]
MYELOID_OUTPUT_GENES = ["LGALS9", "SPP1", "TGFB1", "IL10"]

CORRELATION_METRICS = [
    ("suppressive_program_score", "caf_activation_score"),
    ("suppressive_program_score", "caf_myeloid_input_score"),
    ("suppressive_program_score", "tcell_score"),
    ("suppressive_program_score", "tcell_exhaustion_score"),
    ("trajectory_axis_score", "caf_activation_score"),
    ("trajectory_axis_score", "caf_myeloid_input_score"),
    ("trajectory_axis_score", "tcell_exhaustion_score"),
    ("myeloid_output_score", "tcell_exhaustion_score"),
]


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


def ensure_position_file(manifest_row: pd.Series) -> Path:
    POSITION_DIR.mkdir(parents=True, exist_ok=True)
    url = str(manifest_row["supplementary_file_4"]).replace("ftp://", "https://")
    out_path = POSITION_DIR / Path(url).name
    if not out_path.exists():
        urllib.request.urlretrieve(url, out_path)
    return out_path


def load_positions(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip().split(",")
    has_header = first[0].lower() == "barcode"
    if has_header:
        pos = pd.read_csv(path, compression="gzip")
        rename = {
            pos.columns[0]: "barcode",
            pos.columns[1]: "in_tissue",
            pos.columns[2]: "array_row",
            pos.columns[3]: "array_col",
            pos.columns[4]: "pxl_row_in_fullres",
            pos.columns[5]: "pxl_col_in_fullres",
        }
        pos = pos.rename(columns=rename)
    else:
        pos = pd.read_csv(
            path,
            compression="gzip",
            header=None,
            names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
        )
    pos["barcode"] = pos["barcode"].astype(str)
    return pos


def safe_spearman(df: pd.DataFrame, x: str, y: str, sample_id: str, meta: dict) -> dict:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 25 or sub[x].nunique() < 3 or sub[y].nunique() < 3:
        return {**meta, "sample_id": sample_id, "metric_x": x, "metric_y": y, "n_spots": int(len(sub)), "rho": np.nan, "pvalue": np.nan}
    rho, pvalue = spearmanr(sub[x], sub[y], nan_policy="omit")
    return {**meta, "sample_id": sample_id, "metric_x": x, "metric_y": y, "n_spots": int(len(sub)), "rho": float(rho), "pvalue": float(pvalue)}


def neighbor_matrix(coords: np.ndarray, k: int = 6) -> np.ndarray:
    if coords.shape[0] <= 1:
        return np.zeros((coords.shape[0], 0), dtype=int)
    dist = cdist(coords, coords, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    k_eff = min(k, coords.shape[0] - 1)
    return np.argsort(dist, axis=1)[:, :k_eff]


def safe_wilcoxon(values: pd.Series) -> dict:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 5 or np.allclose(vals.to_numpy(), 0.0):
        return {"n_samples": int(len(vals)), "median_delta": float(vals.median()) if len(vals) else np.nan, "pvalue": np.nan}
    stat = wilcoxon(vals.to_numpy())
    return {"n_samples": int(len(vals)), "median_delta": float(vals.median()), "pvalue": float(stat.pvalue)}


def analyze_one_sample(h5_path: Path, manifest_row: pd.Series, early_genes: list[str], late_genes: list[str]) -> tuple[pd.DataFrame, list[dict], dict, list[dict]]:
    meta = parse_title(str(manifest_row["title"]))
    sample_id = str(manifest_row["gsm_id"])
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    available_program = score_mean_expr(
        adata,
        pd.read_csv(PROGRAM_GENE_PATH)["gene"].dropna().astype(str).tolist(),
        "suppressive_program_score",
    )
    available_early = score_mean_expr(adata, early_genes, "early_module_score")
    available_late = score_mean_expr(adata, late_genes, "late_module_score")
    available_caf = score_mean_expr(adata, CAF_ACTIVATION_GENES, "caf_activation_score")
    available_caf_input = score_mean_expr(adata, CAF_MYELOID_INPUT_GENES, "caf_myeloid_input_score")
    available_tcell = score_mean_expr(adata, T_CELL_GENES, "tcell_score")
    available_tex = score_mean_expr(adata, T_CELL_EXHAUSTION_GENES, "tcell_exhaustion_score")
    available_output = score_mean_expr(adata, MYELOID_OUTPUT_GENES, "myeloid_output_score")

    adata.obs["trajectory_axis_score"] = adata.obs["late_module_score"] - adata.obs["early_module_score"]

    pos = load_positions(ensure_position_file(manifest_row))
    spot_df = pd.DataFrame(
        {
            "barcode": adata.obs_names.astype(str),
            "suppressive_program_score": adata.obs["suppressive_program_score"].to_numpy(),
            "early_module_score": adata.obs["early_module_score"].to_numpy(),
            "late_module_score": adata.obs["late_module_score"].to_numpy(),
            "trajectory_axis_score": adata.obs["trajectory_axis_score"].to_numpy(),
            "caf_activation_score": adata.obs["caf_activation_score"].to_numpy(),
            "caf_myeloid_input_score": adata.obs["caf_myeloid_input_score"].to_numpy(),
            "tcell_score": adata.obs["tcell_score"].to_numpy(),
            "tcell_exhaustion_score": adata.obs["tcell_exhaustion_score"].to_numpy(),
            "myeloid_output_score": adata.obs["myeloid_output_score"].to_numpy(),
        }
    )
    spot_df = pos.merge(spot_df, on="barcode", how="inner")
    spot_df = spot_df[spot_df["in_tissue"].eq(1)].copy()
    spot_df["sample_id"] = sample_id
    for key, value in meta.items():
        spot_df[key] = value

    corr_rows = [
        safe_spearman(spot_df, metric_x, metric_y, sample_id=sample_id, meta=meta) for metric_x, metric_y in CORRELATION_METRICS
    ]

    neighbor_rows: list[dict] = []
    if len(spot_df) >= 50:
        coords = spot_df[["array_row", "array_col"]].to_numpy(dtype=float)
        neighbors = neighbor_matrix(coords, k=6)
        spot_df["high_program"] = spot_df["suppressive_program_score"] >= spot_df["suppressive_program_score"].quantile(0.75)
        spot_df["low_program"] = spot_df["suppressive_program_score"] <= spot_df["suppressive_program_score"].quantile(0.25)
        spot_df["high_caf_input"] = spot_df["caf_myeloid_input_score"] >= spot_df["caf_myeloid_input_score"].quantile(0.75)
        spot_df["high_caf_activation"] = spot_df["caf_activation_score"] >= spot_df["caf_activation_score"].quantile(0.75)
        spot_df["high_tex"] = spot_df["tcell_exhaustion_score"] >= spot_df["tcell_exhaustion_score"].quantile(0.75)
        spot_df["high_tcell"] = spot_df["tcell_score"] >= spot_df["tcell_score"].quantile(0.75)

        for target_flag in ["high_caf_input", "high_caf_activation", "high_tex", "high_tcell"]:
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

    sample_row = {
        "sample_id": sample_id,
        "n_spots_total": int(adata.n_obs),
        "n_spots_in_tissue": int(spot_df.shape[0]),
        "program_genes_available": int(len(available_program)),
        "early_genes_available": int(len(available_early)),
        "late_genes_available": int(len(available_late)),
        "caf_activation_genes_available": int(len(available_caf)),
        "caf_input_genes_available": int(len(available_caf_input)),
        "tcell_genes_available": int(len(available_tcell)),
        "tex_genes_available": int(len(available_tex)),
        "output_genes_available": int(len(available_output)),
        "suppressive_program_mean": float(spot_df["suppressive_program_score"].mean()),
        "trajectory_axis_mean": float(spot_df["trajectory_axis_score"].mean()),
        "caf_input_mean": float(spot_df["caf_myeloid_input_score"].mean()),
        "caf_activation_mean": float(spot_df["caf_activation_score"].mean()),
        "tcell_exhaustion_mean": float(spot_df["tcell_exhaustion_score"].mean()),
        "tcell_score_mean": float(spot_df["tcell_score"].mean()),
        **meta,
    }
    return spot_df, corr_rows, sample_row, neighbor_rows


def main() -> None:
    setup_main()
    manifest = pd.read_csv(MANIFEST_PATH)
    trajectory_summary = json.loads(TRAJECTORY_SUMMARY_PATH.read_text(encoding="utf-8"))
    early_genes = trajectory_summary["top_early_up_genes"][:20]
    late_genes = trajectory_summary["top_late_up_genes"][:30]

    all_spots: list[pd.DataFrame] = []
    all_corrs: list[dict] = []
    all_samples: list[dict] = []
    all_neighbors: list[dict] = []

    for h5_path in sorted(H5_DIR.glob("*filtered_feature_bc_matrix.h5")):
        gsm_id = h5_path.name.split("_", 1)[0]
        manifest_row = manifest.loc[manifest["gsm_id"].eq(gsm_id)].iloc[0]
        spot_df, corr_rows, sample_row, neighbor_rows = analyze_one_sample(h5_path, manifest_row, early_genes, late_genes)
        all_spots.append(spot_df)
        all_corrs.extend(corr_rows)
        all_samples.append(sample_row)
        all_neighbors.extend(neighbor_rows)

    spot_out = pd.concat(all_spots, ignore_index=True)
    corr_out = pd.DataFrame(all_corrs)
    sample_out = pd.DataFrame(all_samples).sort_values("suppressive_program_mean", ascending=False)
    neighbor_out = pd.DataFrame(all_neighbors)

    location_summary = (
        sample_out.groupby("location")[
            ["suppressive_program_mean", "trajectory_axis_mean", "caf_input_mean", "caf_activation_mean", "tcell_exhaustion_mean"]
        ]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    location_summary.columns = ["_".join(col).strip("_") for col in location_summary.columns]

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
    corr_summary = pd.DataFrame(corr_summary_rows).sort_values(["pvalue", "median_delta"], ascending=[True, False], na_position="last")

    neighbor_summary_rows = []
    if not neighbor_out.empty:
        for target_flag, sub in neighbor_out.groupby("target_flag"):
            stats = safe_wilcoxon(sub["delta_high_minus_low"])
            neighbor_summary_rows.append(
                {
                    "target_flag": target_flag,
                    **stats,
                    "mean_delta": float(sub["delta_high_minus_low"].dropna().mean()) if sub["delta_high_minus_low"].notna().any() else np.nan,
                }
            )
    neighbor_summary = pd.DataFrame(neighbor_summary_rows).sort_values(["pvalue", "median_delta"], ascending=[True, False], na_position="last")

    outdir = PROJECT_PATHS["results"] / "spatial_niche"
    write_csv(outdir / "GSE270678_spatial_niche_spot_scores.csv", spot_out)
    write_csv(outdir / "GSE270678_spatial_niche_sample_summary.csv", sample_out)
    write_csv(outdir / "GSE270678_spatial_niche_sample_correlations.csv", corr_out)
    write_csv(outdir / "GSE270678_spatial_niche_correlation_summary.csv", corr_summary)
    write_csv(outdir / "GSE270678_spatial_niche_neighbor_enrichment.csv", neighbor_out)
    write_csv(outdir / "GSE270678_spatial_niche_neighbor_summary.csv", neighbor_summary)
    write_csv(outdir / "GSE270678_spatial_niche_location_summary.csv", location_summary)

    summary = {
        "dataset_id": "GSE270678",
        "n_samples": int(sample_out.shape[0]),
        "n_spots_in_tissue": int(spot_out.shape[0]),
        "top_sample_by_program_mean": sample_out.head(10).to_dict(orient="records"),
        "correlation_summary": corr_summary.to_dict(orient="records"),
        "neighbor_summary": neighbor_summary.to_dict(orient="records"),
        "top_locations_by_program_mean": location_summary.sort_values(
            "suppressive_program_mean_mean", ascending=False
        ).to_dict(orient="records"),
    }
    write_json(outdir / "GSE270678_spatial_niche_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
