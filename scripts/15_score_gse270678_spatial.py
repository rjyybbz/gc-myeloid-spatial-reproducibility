from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import scanpy as sc

from project_setup import PROJECT_PATHS, main as setup_main


def read_manifest() -> pd.DataFrame:
    path = PROJECT_PATHS["metadata"] / "sample_manifests" / "GSE270678_sample_manifest.csv"
    return pd.read_csv(path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_title(title: str) -> dict:
    parts = [p.strip() for p in str(title).split(",")]
    result = {
        "donor": parts[0] if len(parts) > 0 else "",
        "platform": parts[1] if len(parts) > 1 else "",
        "sample_type": parts[2] if len(parts) > 2 else "",
        "stage": parts[3] if len(parts) > 3 else "",
        "subtype": parts[4] if len(parts) > 4 else "",
        "location": parts[5] if len(parts) > 5 else "",
    }
    return result


def main() -> None:
    setup_main()
    program_genes = pd.read_csv(PROJECT_PATHS["results"] / "myeloid_discovery" / "GSE206785_candidate_program_genes.csv")["gene"].tolist()
    manifest = read_manifest()
    h5_dir = PROJECT_PATHS["data_raw"] / "GSE270678" / "spatial_h5"
    rows = []

    for h5_path in sorted(h5_dir.glob("*filtered_feature_bc_matrix.h5")):
        gsm_id = h5_path.name.split("_", 1)[0]
        manifest_row = manifest.loc[manifest["gsm_id"] == gsm_id].iloc[0]
        meta = parse_title(manifest_row["title"])

        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        valid_program = [g for g in program_genes if g in adata.var_names]
        sc.tl.score_genes(adata, gene_list=valid_program, score_name="suppressive_program_score")

        rows.append(
            {
                "gsm_id": gsm_id,
                "title": manifest_row["title"],
                "n_spots": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "program_genes_available": len(valid_program),
                "program_score_mean": float(adata.obs["suppressive_program_score"].mean()),
                "program_score_median": float(adata.obs["suppressive_program_score"].median()),
                "program_score_q75": float(adata.obs["suppressive_program_score"].quantile(0.75)),
                **meta,
            }
        )

    sample_summary = pd.DataFrame(rows).sort_values("program_score_mean", ascending=False)
    location_summary = sample_summary.groupby("location")[["program_score_mean", "program_score_median"]].agg(["mean", "median", "count"])
    location_summary.columns = ["_".join(col).strip("_") for col in location_summary.columns]
    location_summary = location_summary.reset_index()

    outdir = PROJECT_PATHS["results"] / "spatial_validation"
    write_csv(outdir / "GSE270678_spatial_sample_scores.csv", sample_summary)
    write_csv(outdir / "GSE270678_spatial_location_summary.csv", location_summary)

    summary = {
        "dataset_id": "GSE270678",
        "n_samples": int(sample_summary.shape[0]),
        "top_location_by_mean_program_score": location_summary.sort_values("program_score_mean_mean", ascending=False).to_dict(orient="records"),
    }
    (outdir / "GSE270678_spatial_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
