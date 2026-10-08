from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import pandas as pd

from project_setup import PROJECT_PATHS, main as setup_main


H5AD_PATH = PROJECT_PATHS["data_raw"] / "GSE308624" / "GSE308624_gastric_cancer_smi_processed.h5ad"


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_obs_column(obs: pd.DataFrame, column: str, top_n: int = 30) -> pd.DataFrame:
    counts = obs[column].astype(str).fillna("NA").value_counts(dropna=False).head(top_n)
    return counts.rename_axis(column).reset_index(name="n_cells")


def main() -> None:
    setup_main()

    if not H5AD_PATH.exists():
        raise FileNotFoundError(f"Missing processed h5ad: {H5AD_PATH}")

    adata = ad.read_h5ad(H5AD_PATH, backed="r")
    obs = adata.obs.copy()
    obs_cols = [str(col) for col in obs.columns]
    var_cols = [str(col) for col in adata.var.columns]
    uns_keys = list(adata.uns.keys())
    obsm_keys = list(adata.obsm.keys())
    layers = list(adata.layers.keys())

    candidate_columns = [
        col
        for col in obs_cols
        if any(token in col.lower() for token in ["sample", "cell", "type", "cluster", "annot", "fov", "patient"])
    ]

    summary_tables = []
    outdir = PROJECT_PATHS["results"] / "external_spatial_gse308624"
    for column in candidate_columns[:12]:
        table = summarize_obs_column(obs, column)
        write_csv(outdir / f"GSE308624_obs_summary_{column}.csv", table)
        summary_tables.append(
            {
                "column": column,
                "n_unique": int(obs[column].astype(str).nunique(dropna=False)),
                "top_values_path": str(outdir / f"GSE308624_obs_summary_{column}.csv"),
            }
        )

    summary = {
        "dataset_id": "GSE308624",
        "processed_h5ad_path": str(H5AD_PATH),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": obs_cols,
        "var_columns": var_cols,
        "uns_keys": uns_keys,
        "obsm_keys": obsm_keys,
        "layers": layers,
        "candidate_annotation_columns": candidate_columns,
        "candidate_column_summaries": summary_tables,
    }
    write_json(outdir / "GSE308624_h5ad_inspect_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
