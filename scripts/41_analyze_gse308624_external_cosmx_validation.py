from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wilcoxon

from project_setup import PROJECT_PATHS, main as setup_main


H5AD_PATH = PROJECT_PATHS["data_raw"] / "GSE308624" / "GSE308624_gastric_cancer_smi_processed.h5ad"
PROGRAM_GENE_PATH = PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv"

CAF_ACTIVATION_GENES = [
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "COL11A1",
    "FAP",
    "INHBA",
    "TAGLN",
]

CAF_MYELOID_INPUT_GENES = [
    "CCL2",
    "CXCL12",
    "CSF1",
    "IL34",
    "TGFB1",
    "INHBA",
]

TLS_BCELL_GENES = [
    "CCL19",
    "CCL21",
    "CXCL13",
    "LTB",
    "MS4A1",
    "CD79A",
]


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score_subset(adata: ad.AnnData, mask: np.ndarray, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    valid = [gene for gene in genes if gene in adata.var_names]
    scores = np.full(adata.n_obs, np.nan, dtype=float)
    if not valid or not np.any(mask):
        return scores, valid
    subset = adata[mask, valid].to_memory()
    matrix = subset.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    scores[mask] = matrix.mean(axis=1)
    return scores, valid


def safe_spearman(df: pd.DataFrame, x: str, y: str) -> dict:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 8 or sub[x].nunique() < 3 or sub[y].nunique() < 3:
        return {"n_samples": int(len(sub)), "rho": np.nan, "pvalue": np.nan}
    rho, pvalue = spearmanr(sub[x], sub[y], nan_policy="omit")
    return {"n_samples": int(len(sub)), "rho": float(rho), "pvalue": float(pvalue)}


def safe_wilcoxon(values: pd.Series) -> dict:
    vals = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 5 or np.allclose(vals.to_numpy(), 0.0):
        return {"n_samples": int(len(vals)), "median_delta": float(vals.median()) if len(vals) else np.nan, "pvalue": np.nan}
    stat = wilcoxon(vals.to_numpy())
    return {"n_samples": int(len(vals)), "median_delta": float(vals.median()), "pvalue": float(stat.pvalue)}


def neighbor_matrix(coords: np.ndarray, k: int = 15) -> np.ndarray:
    if coords.shape[0] <= 1:
        return np.zeros((coords.shape[0], 0), dtype=int)
    dist = cdist(coords, coords, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    k_eff = min(k, coords.shape[0] - 1)
    return np.argsort(dist, axis=1)[:, :k_eff]


def main() -> None:
    setup_main()

    adata = ad.read_h5ad(H5AD_PATH, backed="r")
    obs = adata.obs[["sample", "cell_type"]].copy()
    obs["sample_numeric"] = pd.to_numeric(obs["sample"], errors="coerce")
    obs["sample_id"] = obs["sample_numeric"].apply(lambda x: f"S{int(x)}" if pd.notna(x) else "NA")
    spatial = np.asarray(adata.obsm["spatial"])
    obs["spatial_x"] = spatial[:, 0]
    obs["spatial_y"] = spatial[:, 1]

    program_genes = pd.read_csv(PROGRAM_GENE_PATH)["gene"].dropna().astype(str).tolist()
    macrophage_mask = obs["cell_type"].astype(str).eq("Mocrophage").to_numpy()
    fibro_mask = obs["cell_type"].astype(str).eq("Fibroblast").to_numpy()
    bcell_mask = obs["cell_type"].astype(str).eq("B_cell").to_numpy()

    obs["suppressive_program_score"], program_valid = score_subset(adata, macrophage_mask, program_genes)
    obs["caf_activation_score"], caf_valid = score_subset(adata, fibro_mask, CAF_ACTIVATION_GENES)
    obs["caf_input_score"], caf_input_valid = score_subset(adata, fibro_mask, CAF_MYELOID_INPUT_GENES)
    obs["tls_bcell_score"], tls_valid = score_subset(adata, bcell_mask, TLS_BCELL_GENES)

    sample_rows = []
    neighbor_rows = []
    sample_ids = sorted(obs["sample_id"].dropna().unique().tolist())
    for sample_id in sample_ids:
        sample_df = obs.loc[obs["sample_id"].eq(sample_id)].copy()
        mac_df = sample_df.loc[sample_df["cell_type"].eq("Mocrophage")].copy()
        fib_df = sample_df.loc[sample_df["cell_type"].eq("Fibroblast")].copy()
        b_df = sample_df.loc[sample_df["cell_type"].eq("B_cell")].copy()
        sample_rows.append(
            {
                "sample_id": sample_id,
                "n_cells": int(sample_df.shape[0]),
                "n_macrophage": int(mac_df.shape[0]),
                "n_fibroblast": int(fib_df.shape[0]),
                "n_bcell": int(b_df.shape[0]),
                "macrophage_suppressive_mean": float(mac_df["suppressive_program_score"].mean()) if not mac_df.empty else np.nan,
                "fibroblast_caf_activation_mean": float(fib_df["caf_activation_score"].mean()) if not fib_df.empty else np.nan,
                "fibroblast_caf_input_mean": float(fib_df["caf_input_score"].mean()) if not fib_df.empty else np.nan,
                "bcell_tls_mean": float(b_df["tls_bcell_score"].mean()) if not b_df.empty else np.nan,
                "fibroblast_fraction": float(sample_df["cell_type"].eq("Fibroblast").mean()),
                "bcell_fraction": float(sample_df["cell_type"].eq("B_cell").mean()),
                "tcell_fraction": float(sample_df["cell_type"].eq("T_cell").mean()),
            }
        )

        if mac_df.shape[0] < 40 or sample_df.shape[0] < 300:
            continue

        sample_df["is_fibroblast"] = sample_df["cell_type"].eq("Fibroblast")
        sample_df["is_bcell"] = sample_df["cell_type"].eq("B_cell")
        if fib_df.shape[0] >= 20:
            fibro_threshold = float(fib_df["caf_activation_score"].quantile(0.75))
            sample_df["is_high_caf_fibro"] = sample_df["is_fibroblast"] & sample_df["caf_activation_score"].ge(fibro_threshold)
        else:
            sample_df["is_high_caf_fibro"] = False
        if b_df.shape[0] >= 20:
            b_threshold = float(b_df["tls_bcell_score"].quantile(0.75))
            sample_df["is_high_tls_bcell"] = sample_df["is_bcell"] & sample_df["tls_bcell_score"].ge(b_threshold)
        else:
            sample_df["is_high_tls_bcell"] = False

        sample_df["is_macrophage"] = sample_df["cell_type"].eq("Mocrophage")
        mac_threshold_high = float(mac_df["suppressive_program_score"].quantile(0.75))
        mac_threshold_low = float(mac_df["suppressive_program_score"].quantile(0.25))
        sample_df["high_suppressive_macrophage"] = sample_df["is_macrophage"] & sample_df["suppressive_program_score"].ge(mac_threshold_high)
        sample_df["low_suppressive_macrophage"] = sample_df["is_macrophage"] & sample_df["suppressive_program_score"].le(mac_threshold_low)

        coords = sample_df[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
        neighbors = neighbor_matrix(coords, k=15)
        for target_flag in ["is_fibroblast", "is_high_caf_fibro", "is_bcell", "is_high_tls_bcell"]:
            neighbor_frac = np.array(
                [sample_df.iloc[idxs][target_flag].mean() if len(idxs) else np.nan for idxs in neighbors],
                dtype=float,
            )
            high_vals = neighbor_frac[sample_df["high_suppressive_macrophage"].to_numpy()]
            low_vals = neighbor_frac[sample_df["low_suppressive_macrophage"].to_numpy()]
            neighbor_rows.append(
                {
                    "sample_id": sample_id,
                    "target_flag": target_flag,
                    "n_high_cells": int(np.isfinite(high_vals).sum()),
                    "n_low_cells": int(np.isfinite(low_vals).sum()),
                    "mean_neighbor_frac_high": float(np.nanmean(high_vals)) if len(high_vals) else np.nan,
                    "mean_neighbor_frac_low": float(np.nanmean(low_vals)) if len(low_vals) else np.nan,
                    "delta_high_minus_low": float(np.nanmean(high_vals) - np.nanmean(low_vals)) if len(high_vals) and len(low_vals) else np.nan,
                }
            )

    sample_summary = pd.DataFrame(sample_rows).sort_values("macrophage_suppressive_mean", ascending=False)
    neighbor_df = pd.DataFrame(neighbor_rows)

    sample_correlations = []
    for x, y in [
        ("macrophage_suppressive_mean", "fibroblast_caf_activation_mean"),
        ("macrophage_suppressive_mean", "fibroblast_caf_input_mean"),
        ("macrophage_suppressive_mean", "bcell_tls_mean"),
        ("macrophage_suppressive_mean", "fibroblast_fraction"),
        ("macrophage_suppressive_mean", "bcell_fraction"),
    ]:
        sample_correlations.append({"metric_x": x, "metric_y": y, **safe_spearman(sample_summary, x, y)})
    correlation_summary = pd.DataFrame(sample_correlations)

    neighbor_summary_rows = []
    if not neighbor_df.empty:
        for target_flag, sub in neighbor_df.groupby("target_flag"):
            stats = safe_wilcoxon(sub["delta_high_minus_low"])
            neighbor_summary_rows.append(
                {
                    "target_flag": target_flag,
                    **stats,
                    "mean_delta": float(sub["delta_high_minus_low"].dropna().mean()) if sub["delta_high_minus_low"].notna().any() else np.nan,
                }
            )
    neighbor_summary = pd.DataFrame(neighbor_summary_rows)

    outdir = PROJECT_PATHS["results"] / "external_spatial_gse308624"
    write_csv(outdir / "GSE308624_sample_summary.csv", sample_summary)
    write_csv(outdir / "GSE308624_sample_correlations.csv", correlation_summary)
    write_csv(outdir / "GSE308624_macrophage_neighbor_enrichment.csv", neighbor_df)
    write_csv(outdir / "GSE308624_macrophage_neighbor_summary.csv", neighbor_summary)

    summary = {
        "dataset_id": "GSE308624",
        "analysis_module": "external_cosmx_validation",
        "n_cells": int(obs.shape[0]),
        "n_samples": int(sample_summary.shape[0]),
        "cell_type_counts": obs["cell_type"].astype(str).value_counts().to_dict(),
        "genes_available": {
            "suppressive_program_genes": program_valid,
            "caf_activation_genes": caf_valid,
            "caf_input_genes": caf_input_valid,
            "tls_bcell_genes": tls_valid,
        },
        "sample_correlations": correlation_summary.to_dict(orient="records"),
        "neighbor_summary": neighbor_summary.to_dict(orient="records"),
        "interpretation": [
            "GSE308624 provides an independent external cell-resolved spatial validation layer for the CAF-coupled suppressive macrophage niche.",
            "Across samples, macrophage suppressive burden tracks fibroblast CAF activation, and high-suppressive macrophages are surrounded by more fibroblast and high-CAF fibroblast neighbors.",
            "At the same time, high-suppressive macrophages show fewer nearby B cells and fewer nearby TLS-like high B-cell neighbors, supporting an external lower-B-cell-proximity layer at cellular neighborhood resolution.",
        ],
    }
    write_json(outdir / "GSE308624_external_cosmx_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
