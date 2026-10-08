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


def download(url: str, output_path: Path) -> None:
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
            print(f"[retry] attempt={attempt} url={url} error={exc}")
            time.sleep(attempt * 2)
    print(f"[failed] {url} | {last_error}")


def main() -> None:
    setup_main()
    manifest = read_csv(PROJECT_PATHS["metadata"] / "sample_manifests" / "GSE270678_sample_manifest.csv")
    for row in manifest:
        title = row["title"]
        for key, value in row.items():
            if key.startswith("supplementary_file_") and value and "filtered_feature_bc_matrix.h5" in value:
                url = value.replace("ftp://", "https://")
                filename = Path(urlparse(url).path).name
                download(url, PROJECT_PATHS["data_raw"] / "GSE270678" / "spatial_h5" / filename)


if __name__ == "__main__":
    main()
