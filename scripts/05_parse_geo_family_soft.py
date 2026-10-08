from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

from project_setup import PROJECT_PATHS, main as setup_main


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_soft(soft_path: Path) -> list[dict]:
    rows: list[dict] = []
    current: dict | None = None
    with gzip.open(soft_path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current:
                    rows.append(current)
                current = {"gsm_id": line.split(" = ", 1)[1]}
                continue
            if current is None:
                continue
            if line.startswith("!Sample_"):
                key, value = line[1:].split(" = ", 1)
                key = key.replace("Sample_", "")
                if key in current and current[key]:
                    current[key] = f"{current[key]} | {value}"
                else:
                    current[key] = value
        if current:
            rows.append(current)
    return rows


def infer_soft_path(accession: str) -> Path:
    return PROJECT_PATHS["data_raw"] / accession / "geo_family" / f"{accession}_family.soft.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse GEO family.soft.gz files into sample manifests.")
    parser.add_argument("--accessions", nargs="+", required=True, help="One or more GEO GSE accessions.")
    args = parser.parse_args()

    setup_main()
    for accession in args.accessions:
        soft_path = infer_soft_path(accession)
        if not soft_path.exists():
            print(f"[parse] missing file for {accession}: {soft_path}")
            continue
        rows = parse_soft(soft_path)
        output_path = PROJECT_PATHS["metadata"] / "sample_manifests" / f"{accession}_sample_manifest.csv"
        write_csv(output_path, rows)
        print(f"[parse] {accession}: {len(rows)} samples -> {output_path}")


if __name__ == "__main__":
    main()
