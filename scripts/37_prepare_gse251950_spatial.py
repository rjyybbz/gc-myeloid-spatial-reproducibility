from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pandas as pd

from project_setup import PROJECT_PATHS, main as setup_main


RAW_TAR_PATH = PROJECT_PATHS["data_raw"] / "GSE251950" / "GSE251950_RAW.tar"
MANIFEST_PATH = PROJECT_PATHS["metadata"] / "sample_manifests" / "GSE251950_sample_manifest.csv"
OUT_H5_DIR = PROJECT_PATHS["data_raw"] / "GSE251950" / "spatial_h5"
OUT_POS_DIR = PROJECT_PATHS["data_raw"] / "GSE251950" / "tissue_positions"
OUT_META_DIR = PROJECT_PATHS["data_raw"] / "GSE251950" / "sample_assets"


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_tissue_group(row: pd.Series) -> str:
    source = str(row.get("source_name_ch1", "")).lower()
    title = str(row.get("title", ""))
    if "metastasis" in source or title.endswith("-PM"):
        return "peritoneal_met"
    return "primary_tumor"


def infer_pair_id(title: str) -> str:
    return str(title).replace("-PM", "")


def infer_asset_stub(row: pd.Series) -> str:
    url_name = Path(str(row["supplementary_file_1"]).replace("ftp://", "https://")).name.replace(".tar.gz", "")
    gsm_id = str(row["gsm_id"])
    prefix = f"{gsm_id}_"
    if url_name.startswith(prefix):
        return url_name[len(prefix):]
    return url_name


def extract_nested_member(nested_tar: tarfile.TarFile, member_name: str, output_path: Path) -> bool:
    members = {m.name: m for m in nested_tar.getmembers()}
    member = members.get(member_name)
    if member is None:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return True
    handle = nested_tar.extractfile(member)
    if handle is None:
        return False
    with output_path.open("wb") as out:
        out.write(handle.read())
    return True


def main() -> None:
    setup_main()

    if not RAW_TAR_PATH.exists():
        raise FileNotFoundError(f"Missing raw tar: {RAW_TAR_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    nested_map = {
        row["gsm_id"]: {
            "gsm_id": row["gsm_id"],
            "sample_title": row["title"],
            "asset_stub": infer_asset_stub(row),
            "tissue_group": infer_tissue_group(row),
            "pair_id": infer_pair_id(str(row["title"])),
        }
        for _, row in manifest.iterrows()
    }

    extracted_rows: list[dict] = []
    with tarfile.open(RAW_TAR_PATH, "r") as outer:
        members = [m for m in outer.getmembers() if m.isfile() and m.name.endswith(".tar.gz")]
        for member in members:
            gsm_id = member.name.split("_", 1)[0]
            if gsm_id not in nested_map:
                continue
            sample_info = nested_map[gsm_id]
            nested_bytes = outer.extractfile(member)
            if nested_bytes is None:
                continue
            fileobj = io.BytesIO(nested_bytes.read())
            with tarfile.open(fileobj=fileobj, mode="r:gz") as nested:
                stub = sample_info["asset_stub"]
                h5_name = f"{stub}/{stub}_filtered_feature_bc_matrix.h5"
                pos_name = f"{stub}/{stub}_tissue_positions_list.csv"
                scale_name = f"{stub}/{stub}_scalefactors_json.json"

                h5_out = OUT_H5_DIR / f"{gsm_id}_{stub}_filtered_feature_bc_matrix.h5"
                pos_out = OUT_POS_DIR / f"{gsm_id}_{stub}_tissue_positions_list.csv"
                scale_out = OUT_META_DIR / f"{gsm_id}_{stub}_scalefactors_json.json"

                found_h5 = extract_nested_member(nested, h5_name, h5_out)
                found_pos = extract_nested_member(nested, pos_name, pos_out)
                found_scale = extract_nested_member(nested, scale_name, scale_out)

                extracted_rows.append(
                    {
                        **sample_info,
                        "outer_member": member.name,
                        "found_h5": bool(found_h5),
                        "found_positions": bool(found_pos),
                        "found_scalefactors": bool(found_scale),
                        "h5_path": str(h5_out),
                        "positions_path": str(pos_out),
                        "scalefactors_path": str(scale_out),
                    }
                )

    extracted_df = pd.DataFrame(extracted_rows).sort_values(["tissue_group", "sample_title", "gsm_id"])
    write_csv(PROJECT_PATHS["data_processed"] / "GSE251950" / "GSE251950_spatial_asset_manifest.csv", extracted_df)

    summary = {
        "dataset_id": "GSE251950",
        "n_samples_manifest": int(manifest.shape[0]),
        "n_samples_extracted": int(extracted_df.shape[0]),
        "n_h5_found": int(extracted_df["found_h5"].sum()) if not extracted_df.empty else 0,
        "n_positions_found": int(extracted_df["found_positions"].sum()) if not extracted_df.empty else 0,
        "tissue_group_counts": extracted_df["tissue_group"].value_counts().to_dict() if not extracted_df.empty else {},
        "samples": extracted_df.to_dict(orient="records"),
    }
    write_json(PROJECT_PATHS["results"] / "external_spatial_gse251950" / "GSE251950_prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
