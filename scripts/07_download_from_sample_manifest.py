from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import urlparse

import requests

from project_setup import PROJECT_PATHS, main as setup_main


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_urls(row: dict) -> list[str]:
    urls = []
    for key, value in row.items():
        if key.startswith("supplementary_file_") and value and value.startswith("ftp://"):
            urls.append(value.replace("ftp://", "https://"))
    return urls


def download_file(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    output_path = output_dir / filename
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download supplementary files listed in a parsed sample manifest.")
    parser.add_argument("--accession", required=True, help="GEO accession such as GSE315928")
    parser.add_argument("--pattern", default="", help="Optional substring filter applied to URLs")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of files to download")
    args = parser.parse_args()

    setup_main()
    manifest_path = PROJECT_PATHS["metadata"] / "sample_manifests" / f"{args.accession}_sample_manifest.csv"
    rows = read_csv(manifest_path)
    urls = []
    for row in rows:
        urls.extend(candidate_urls(row))
    urls = [url for url in urls if args.pattern in url]
    urls = urls[: args.limit]

    if not urls:
        print(f"[download] no matching supplementary URLs for {args.accession}")
        return

    output_dir = PROJECT_PATHS["data_raw"] / args.accession / "supplementary_preview"
    for url in urls:
        try:
            output_path = download_file(url, output_dir)
            print(f"[download] {url} -> {output_path}")
        except Exception as exc:
            print(f"[download] FAILED {url}: {exc}")


if __name__ == "__main__":
    main()
