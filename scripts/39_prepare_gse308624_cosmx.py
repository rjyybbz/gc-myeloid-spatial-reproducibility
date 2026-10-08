from __future__ import annotations

import json
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from project_setup import PROJECT_PATHS, main as setup_main


DATASET_ID = "GSE308624"
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE308nnn/GSE308624"
RAW_DIR = PROJECT_PATHS["data_raw"] / DATASET_ID
RESULT_DIR = PROJECT_PATHS["results"] / "external_spatial_gse308624"
META_DIR = PROJECT_PATHS["metadata"] / "sample_manifests"

FILES = {
    "family_xml_tgz": f"{BASE_URL}/miniml/GSE308624_family.xml.tgz",
    "family_soft_gz": f"{BASE_URL}/soft/GSE308624_family.soft.gz",
    "clinical_xlsx": f"{BASE_URL}/suppl/GSE308624_gastric_sample_clinical.xlsx",
    "processed_h5ad": f"{BASE_URL}/suppl/GSE308624_gastric_cancer_smi_processed.h5ad",
}

NS = {"geo": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    command = ["curl.exe", "-L", url, "-o", str(out_path)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to download {url}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def parse_family_xml(xml_tgz_path: Path) -> tuple[dict, pd.DataFrame]:
    with tarfile.open(xml_tgz_path, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name.endswith(".xml")]
        if not members:
            raise FileNotFoundError(f"No XML member found in {xml_tgz_path}")
        xml_bytes = tar.extractfile(members[0]).read()

    root = ET.fromstring(xml_bytes)
    series = root.find("geo:Series", NS)
    if series is None:
        raise ValueError("Series node not found in MINiML XML")

    series_info = {
        "title": (series.findtext("geo:Title", default="", namespaces=NS) or "").strip(),
        "summary": (series.findtext("geo:Summary", default="", namespaces=NS) or "").strip(),
        "overall_design": (series.findtext("geo:Overall-Design", default="", namespaces=NS) or "").strip(),
        "pubmed_id": (series.findtext("geo:Pubmed-ID", default="", namespaces=NS) or "").strip(),
    }
    series_info["supplementary_files"] = [
        {
            "type": supp.attrib.get("type", ""),
            "url": (supp.text or "").strip().replace("ftp://", "https://"),
        }
        for supp in series.findall("geo:Supplementary-Data", NS)
    ]

    sample_rows: list[dict] = []
    for sample in root.findall("geo:Sample", NS):
        row = {
            "series_id": DATASET_ID,
            "gsm_id": sample.attrib.get("iid", "").strip(),
            "title": (sample.findtext("geo:Title", default="", namespaces=NS) or "").strip(),
            "source_name_ch1": "",
        }
        channel = sample.find("geo:Channel", NS)
        if channel is not None:
            row["source_name_ch1"] = (channel.findtext("geo:Source", default="", namespaces=NS) or "").strip()
            for char_node in channel.findall("geo:Characteristics", NS):
                tag = str(char_node.attrib.get("tag", "")).strip()
                key = tag.lower().replace(" ", "_").replace("#", "_num")
                row[f"characteristic_{key}"] = (char_node.text or "").strip()
        sample_rows.append(row)

    sample_manifest = pd.DataFrame(sample_rows).sort_values(["title", "gsm_id"]).reset_index(drop=True)
    return series_info, sample_manifest


def main() -> None:
    setup_main()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        suffix = Path(url).name
        download_file(url, RAW_DIR / suffix)

    series_info, sample_manifest = parse_family_xml(RAW_DIR / "GSE308624_family.xml.tgz")
    write_csv(META_DIR / "GSE308624_sample_manifest.csv", sample_manifest)

    clinical_xlsx_path = RAW_DIR / "GSE308624_gastric_sample_clinical.xlsx"
    clinical_sheets: dict[str, pd.DataFrame] = pd.read_excel(clinical_xlsx_path, sheet_name=None)
    clinical_sheet_summaries = []
    for sheet_name, sheet_df in clinical_sheets.items():
        safe_name = sheet_name.replace(" ", "_").replace("/", "_")
        write_csv(PROJECT_PATHS["data_processed"] / DATASET_ID / f"{safe_name}.csv", sheet_df)
        clinical_sheet_summaries.append(
            {
                "sheet_name": sheet_name,
                "n_rows": int(sheet_df.shape[0]),
                "n_cols": int(sheet_df.shape[1]),
                "columns": [str(col) for col in sheet_df.columns],
            }
        )

    summary = {
        "dataset_id": DATASET_ID,
        "series_title": series_info["title"],
        "pubmed_id": series_info["pubmed_id"],
        "n_samples": int(sample_manifest.shape[0]),
        "sample_manifest_path": str(META_DIR / "GSE308624_sample_manifest.csv"),
        "supplementary_files": series_info["supplementary_files"],
        "downloaded_files": {
            path.name: {"path": str(path), "size_bytes": int(path.stat().st_size)}
            for path in RAW_DIR.iterdir()
            if path.is_file()
        },
        "clinical_sheet_summaries": clinical_sheet_summaries,
    }
    write_json(RESULT_DIR / "GSE308624_prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
