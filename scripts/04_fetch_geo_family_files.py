from __future__ import annotations

import argparse
from pathlib import Path

import requests

from project_setup import PROJECT_PATHS, main as setup_main


def gse_series_prefix(accession: str) -> str:
    numeric = accession.replace("GSE", "")
    return f"GSE{numeric[:-3]}nnn"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def build_geo_urls(accession: str) -> dict[str, str]:
    prefix = gse_series_prefix(accession)
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}"
    return {
        "family_soft": f"{base}/soft/{accession}_family.soft.gz",
        "series_matrix": f"{base}/matrix/{accession}_series_matrix.txt.gz",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GEO family-level metadata files.")
    parser.add_argument("--accessions", nargs="+", required=True, help="One or more GEO GSE accessions.")
    args = parser.parse_args()

    setup_main()
    for accession in args.accessions:
        urls = build_geo_urls(accession)
        target_dir = PROJECT_PATHS["data_raw"] / accession / "geo_family"
        for label, url in urls.items():
            output_path = target_dir / Path(url).name
            try:
                download_file(url, output_path)
                print(f"[download] {accession} {label} -> {output_path}")
            except Exception as exc:
                print(f"[download] FAILED {accession} {label}: {exc}")


if __name__ == "__main__":
    main()
