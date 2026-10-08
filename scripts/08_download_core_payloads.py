from __future__ import annotations

import csv
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from project_setup import PROJECT_PATHS, main as setup_main


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[skip] {output_path}")
        return
    last_error = None
    for attempt in range(1, 6):
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with output_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            print(f"[download] {url} -> {output_path}")
            return
        except Exception as exc:
            last_error = exc
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink(missing_ok=True)
            print(f"[retry] attempt={attempt} url={url} error={exc}")
            time.sleep(attempt * 2)
    print(f"[failed] {url} -> {output_path} | {last_error}")


def download_gse206785() -> None:
    base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206785/suppl"
    for filename in ["GSE206785_metadata.txt.gz", "GSE206785_scgex.txt.gz"]:
        download_file(
            f"{base}/{filename}",
            PROJECT_PATHS["data_raw"] / "GSE206785" / "supplementary" / filename,
        )


def download_from_manifest(accession: str, limit: int | None = None) -> None:
    manifest_path = PROJECT_PATHS["metadata"] / "sample_manifests" / f"{accession}_sample_manifest.csv"
    rows = read_csv(manifest_path)
    urls: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key.startswith("supplementary_file_") and value and value.startswith("ftp://"):
                urls.append(value.replace("ftp://", "https://"))
    if limit is not None:
        urls = urls[:limit]
    for url in urls:
        filename = Path(urlparse(url).path).name
        output_path = PROJECT_PATHS["data_raw"] / accession / "supplementary" / filename
        download_file(url, output_path)


def main() -> None:
    setup_main()
    download_gse206785()
    download_from_manifest("GSE163558")
    download_from_manifest("GSE183904")


if __name__ == "__main__":
    main()
