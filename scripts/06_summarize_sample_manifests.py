from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from project_setup import PROJECT_PATHS, main as setup_main


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_gse206785(rows: list[dict]) -> list[dict]:
    labels = []
    for row in rows:
        title = row.get("title", "")
        if title.endswith("_T"):
            labels.append("tumor")
        elif title.endswith("_N"):
            labels.append("adjacent_normal")
        else:
            labels.append("unknown")
    counter = Counter(labels)
    return [{"dataset_id": "GSE206785", "feature": key, "count": value} for key, value in sorted(counter.items())]


def summarize_gse270678(rows: list[dict]) -> list[dict]:
    location_counter = Counter()
    stage_counter = Counter()
    subtype_counter = Counter()
    for row in rows:
        title = row.get("title", "")
        parts = [part.strip() for part in title.split(",")]
        if len(parts) >= 6:
            stage_counter[parts[3]] += 1
            subtype_counter[parts[4]] += 1
            location_counter[parts[5]] += 1
    output = []
    output.extend({"dataset_id": "GSE270678", "feature": f"location:{k}", "count": v} for k, v in sorted(location_counter.items()))
    output.extend({"dataset_id": "GSE270678", "feature": f"stage:{k}", "count": v} for k, v in sorted(stage_counter.items()))
    output.extend({"dataset_id": "GSE270678", "feature": f"subtype:{k}", "count": v} for k, v in sorted(subtype_counter.items()))
    return output


def summarize_gse315928(rows: list[dict]) -> list[dict]:
    timepoint_counter = Counter()
    for row in rows:
        title = row.get("title", "")
        if title.endswith("_B"):
            timepoint_counter["baseline"] += 1
        elif title.endswith("_F1"):
            timepoint_counter["followup_1"] += 1
        elif title.endswith("_F2"):
            timepoint_counter["followup_2"] += 1
        else:
            timepoint_counter["other"] += 1
    return [{"dataset_id": "GSE315928", "feature": key, "count": value} for key, value in sorted(timepoint_counter.items())]


def summarize_gse229379(rows: list[dict]) -> list[dict]:
    library_counter = Counter()
    antibiotics_counter = Counter()
    for row in rows:
        characteristics = row.get("characteristics_ch1", "")
        if "library type: TCR" in characteristics:
            library_counter["TCR"] += 1
        elif "library type: mRNA" in characteristics:
            library_counter["mRNA"] += 1
        else:
            library_counter["unknown"] += 1
        used = len(re.findall(r"Used", characteristics))
        not_used = len(re.findall(r"Not used", characteristics))
        antibiotics_counter[f"pooled_used_{used}_not_used_{not_used}"] += 1
    output = []
    output.extend({"dataset_id": "GSE229379", "feature": f"library:{k}", "count": v} for k, v in sorted(library_counter.items()))
    output.extend({"dataset_id": "GSE229379", "feature": f"antibiotics_pool:{k}", "count": v} for k, v in sorted(antibiotics_counter.items()))
    return output


def build_general_summary(manifest_dir: Path) -> list[dict]:
    rows = []
    for manifest_path in sorted(manifest_dir.glob("*_sample_manifest.csv")):
        dataset_id = manifest_path.name.replace("_sample_manifest.csv", "")
        dataset_rows = read_csv(manifest_path)
        rows.append(
            {
                "dataset_id": dataset_id,
                "n_rows": len(dataset_rows),
                "n_columns": len(dataset_rows[0]) if dataset_rows else 0,
                "has_supplementary_links": sum(1 for row in dataset_rows if row.get("supplementary_file_1")),
            }
        )
    return rows


def main() -> None:
    setup_main()
    manifest_dir = PROJECT_PATHS["metadata"] / "sample_manifests"
    general_summary = build_general_summary(manifest_dir)
    detail_rows = []
    detail_rows.extend(summarize_gse206785(read_csv(manifest_dir / "GSE206785_sample_manifest.csv")))
    detail_rows.extend(summarize_gse270678(read_csv(manifest_dir / "GSE270678_sample_manifest.csv")))
    detail_rows.extend(summarize_gse315928(read_csv(manifest_dir / "GSE315928_sample_manifest.csv")))
    detail_rows.extend(summarize_gse229379(read_csv(manifest_dir / "GSE229379_sample_manifest.csv")))

    write_csv(PROJECT_PATHS["results"] / "tables" / "sample_manifest_overview.csv", general_summary)
    write_csv(PROJECT_PATHS["results"] / "tables" / "sample_manifest_key_counts.csv", detail_rows)

    print(f"[summary] overview rows: {len(general_summary)}")
    print(f"[summary] key count rows: {len(detail_rows)}")


if __name__ == "__main__":
    main()
