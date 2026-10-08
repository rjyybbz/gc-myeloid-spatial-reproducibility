from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from project_setup import PROJECT_PATHS, main as setup_main


CHUNK_SIZE = 2000


def main() -> None:
    setup_main()

    metadata_path = PROJECT_PATHS["data_raw"] / "GSE206785" / "supplementary" / "GSE206785_metadata.txt.gz"
    expression_path = PROJECT_PATHS["data_raw"] / "GSE206785" / "supplementary" / "GSE206785_scgex.txt.gz"
    output_path = PROJECT_PATHS["data_processed"] / "GSE206785" / "GSE206785_myeloid_logexpr.h5ad"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_path, sep="\t")
    myeloid_mask = meta["Annotation"].astype(str).str.match(r"^M\d+")
    myeloid_meta = meta.loc[myeloid_mask].copy().reset_index(drop=True)
    selected_positions = set(meta.index[myeloid_mask].tolist())

    sparse_chunks = []
    cell_ids = []
    gene_names = None
    row_start = 0
    for chunk in pd.read_csv(expression_path, sep=",", chunksize=CHUNK_SIZE):
        if gene_names is None:
            gene_names = chunk.columns[1:].tolist()
        row_end = row_start + len(chunk)
        keep_local_idx = [i for i, global_idx in enumerate(range(row_start, row_end)) if global_idx in selected_positions]
        if keep_local_idx:
            subset = chunk.iloc[keep_local_idx].copy()
            cell_ids.extend(subset.iloc[:, 0].astype(str).tolist())
            matrix = sparse.csr_matrix(subset.iloc[:, 1:].to_numpy(dtype=np.float32))
            sparse_chunks.append(matrix)
        row_start = row_end
        print(f"[chunk] processed_rows={row_end} kept_cells={len(cell_ids)}")

    X = sparse.vstack(sparse_chunks, format="csr")
    obs = myeloid_meta.copy()
    obs.index = pd.Index(cell_ids, name="cell_id")
    obs["candidate_state"] = obs["Annotation"].isin(["M08_Mono_INHBA", "M04_Macro_C3", "M07_Macro_APOE", "M11_Macro_PLA2G7"])
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene_symbol"))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["dataset_id"] = "GSE206785"
    adata.uns["input_scale"] = "log_normalized_from_public_supplementary"
    adata.write_h5ad(output_path)

    print(f"[write] {output_path} shape={adata.shape}")


if __name__ == "__main__":
    main()
