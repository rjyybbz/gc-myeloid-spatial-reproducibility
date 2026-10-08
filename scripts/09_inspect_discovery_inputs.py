from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pandas as pd

from project_setup import PROJECT_PATHS, main as setup_main


def inspect_gzip_table(path: Path, nrows: int = 5) -> dict:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        df = pd.read_csv(handle, sep=None, engine="python", nrows=nrows)
    return {
        "path": str(path),
        "columns": list(df.columns[:20]),
        "n_preview_rows": int(df.shape[0]),
        "n_preview_cols": int(df.shape[1]),
    }


def inspect_plain_table(path: Path, nrows: int = 5) -> dict:
    df = pd.read_csv(path, sep=None, engine="python", nrows=nrows)
    return {
        "path": str(path),
        "columns": list(df.columns[:20]),
        "n_preview_rows": int(df.shape[0]),
        "n_preview_cols": int(df.shape[1]),
    }


def inspect_h5ad(path: Path) -> dict:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    result = {
        "path": str(path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "obs_columns": list(adata.obs.columns[:40]),
        "var_columns": list(adata.var.columns[:20]),
    }
    adata.file.close()
    return result


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    setup_main()
    outputs = {}

    outputs["GSE206785_metadata"] = inspect_gzip_table(
        PROJECT_PATHS["data_raw"] / "GSE206785" / "supplementary" / "GSE206785_metadata.txt.gz"
    )
    outputs["GSE206785_scgex"] = inspect_gzip_table(
        PROJECT_PATHS["data_raw"] / "GSE206785" / "supplementary" / "GSE206785_scgex.txt.gz"
    )

    first_183904 = sorted((PROJECT_PATHS["data_raw"] / "GSE183904" / "supplementary").glob("*.csv.gz"))[0]
    outputs["GSE183904_sample_preview"] = inspect_gzip_table(first_183904)

    first_315928 = sorted((PROJECT_PATHS["data_raw"] / "GSE315928" / "supplementary_preview").glob("*.h5ad"))[0]
    outputs["GSE315928_h5ad_preview"] = inspect_h5ad(first_315928)

    write_json(PROJECT_PATHS["results"] / "qc" / "initial_input_inspection.json", outputs)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
