from __future__ import annotations

import csv
import gzip
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import io, sparse

from project_setup import PROJECT_PATHS, main as setup_main


def read_manifest() -> pd.DataFrame:
    path = PROJECT_PATHS["metadata"] / "sample_manifests" / "GSE163558_sample_manifest.csv"
    return pd.read_csv(path)


def sample_prefixes() -> list[str]:
    return sorted({p.name.replace("_barcodes.tsv.gz", "") for p in (PROJECT_PATHS["data_raw"] / "GSE163558" / "supplementary").glob("*_barcodes.tsv.gz")})


def load_one_sample(prefix: str, manifest: pd.DataFrame) -> ad.AnnData:
    base = PROJECT_PATHS["data_raw"] / "GSE163558" / "supplementary"
    with gzip.open(base / f"{prefix}_matrix.mtx.gz", "rb") as handle:
        matrix = io.mmread(handle).tocsr().T
    barcodes = pd.read_csv(base / f"{prefix}_barcodes.tsv.gz", header=None, sep="\t").iloc[:, 0].astype(str).tolist()
    features = pd.read_csv(base / f"{prefix}_features.tsv.gz", header=None, sep="\t")
    gene_names = features.iloc[:, 1].astype(str).tolist() if features.shape[1] > 1 else features.iloc[:, 0].astype(str).tolist()

    short_id = prefix.split("_", 1)[1]
    sample_row = manifest[manifest["title"].astype(str).str.contains(short_id, regex=False)].iloc[0]
    obs = pd.DataFrame(index=[f"{prefix}:{bc}" for bc in barcodes])
    obs["sample_prefix"] = prefix
    obs["source_name_ch1"] = sample_row["source_name_ch1"]
    obs["title"] = sample_row["title"]
    obs["tissue"] = str(sample_row["source_name_ch1"])
    obs["treatment"] = str(sample_row.get("treatment_protocol_ch1", ""))

    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_symbol"))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    return adata


def main() -> None:
    setup_main()
    manifest = read_manifest()
    prefixes = sample_prefixes()
    adatas = [load_one_sample(prefix, manifest) for prefix in prefixes]
    adata = ad.concat(adatas, join="outer", merge="same")
    adata.var_names_make_unique()
    adata.uns["dataset_id"] = "GSE163558"
    output_dir = PROJECT_PATHS["data_processed"] / "GSE163558"
    output_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_dir / "GSE163558_full.h5ad")
    print(f"[write] shape={adata.shape}")


if __name__ == "__main__":
    main()
