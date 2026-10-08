from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import requests

from project_setup import PROJECT_PATHS, main as setup_main


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

SEARCH_TERMS = [
    "gastric cancer single cell RNA-seq",
    "gastric cancer spatial transcriptomics",
    "gastric cancer immunotherapy single cell",
    "advanced gastric cancer chemoimmunotherapy single cell",
    "gastric cancer anti-PD-1 single cell",
]


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


def ncbi_get(endpoint: str, params: dict, retries: int = 5, delay: float = 2.0) -> dict:
    session = requests.Session()
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(f"{EUTILS_BASE}/{endpoint}", params=params, timeout=45)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = str(exc)
            time.sleep(delay * attempt)
    raise RuntimeError(f"NCBI request failed after {retries} retries: {last_error}")


def load_curated_accessions() -> list[str]:
    curated_path = PROJECT_PATHS["metadata"] / "dataset_inventory_curated.csv"
    accessions: list[str] = []
    with curated_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row["dataset_id"].strip()
            if accession.startswith("GSE"):
                accessions.append(accession)
    return accessions


def fetch_accession_summary(accession: str) -> dict:
    search = ncbi_get(
        "esearch.fcgi",
        {"db": "gds", "term": f"\"{accession}\"", "retmode": "json", "retmax": 5},
    )
    ids = search["esearchresult"]["idlist"]
    if not ids:
        return {
            "query_accession": accession,
            "uid": "",
            "matched_accession": "",
            "title": "",
            "summary": "",
            "n_samples": "",
            "taxon": "",
            "notes": "not_found_in_gds",
        }
    summary_json = ncbi_get(
        "esummary.fcgi",
        {"db": "gds", "id": ",".join(ids), "retmode": "json"},
    )
    matched_uid = ""
    item = {}
    for uid in ids:
        candidate = summary_json["result"].get(uid, {})
        if candidate.get("accession") == accession:
            matched_uid = uid
            item = candidate
            break
    if not item:
        matched_uid = ids[0]
        item = summary_json["result"][matched_uid]
        note = "accession_mismatch"
    else:
        note = ""
    return {
        "query_accession": accession,
        "uid": matched_uid,
        "matched_accession": item.get("accession", ""),
        "title": item.get("title", ""),
        "summary": (item.get("summary", "") or "").replace("\n", " "),
        "n_samples": item.get("n_samples", ""),
        "taxon": item.get("taxon", ""),
        "notes": note,
    }


def search_by_term(term: str) -> list[dict]:
    search = ncbi_get(
        "esearch.fcgi",
        {"db": "gds", "term": term, "retmode": "json", "retmax": 10},
    )
    ids = search["esearchresult"]["idlist"]
    if not ids:
        return []
    summary_json = ncbi_get(
        "esummary.fcgi",
        {"db": "gds", "id": ",".join(ids), "retmode": "json"},
    )
    rows = []
    for uid in ids:
        item = summary_json["result"].get(uid, {})
        rows.append(
            {
                "search_term": term,
                "uid": uid,
                "accession": item.get("accession", ""),
                "title": item.get("title", ""),
                "n_samples": item.get("n_samples", ""),
                "summary": (item.get("summary", "") or "").replace("\n", " "),
            }
        )
    return rows


def main() -> None:
    setup_main()
    curated_accessions = load_curated_accessions()
    accession_rows = [fetch_accession_summary(accession) for accession in curated_accessions]
    search_rows = []
    for term in SEARCH_TERMS:
        search_rows.extend(search_by_term(term))

    accession_path = PROJECT_PATHS["metadata"] / "dataset_inventory_ncbi_summary.csv"
    search_path = PROJECT_PATHS["inventory"] / "ncbi_geo_search_results.csv"
    write_csv(accession_path, accession_rows)
    write_csv(search_path, search_rows)

    manifest = {
        "queried_accessions": curated_accessions,
        "search_terms": SEARCH_TERMS,
        "records_written": {
            "accession_summaries": len(accession_rows),
            "search_hits": len(search_rows),
        },
    }
    (PROJECT_PATHS["logs"] / "ncbi_query_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[ncbi] accession summaries: {len(accession_rows)} -> {accession_path}")
    print(f"[ncbi] search hits: {len(search_rows)} -> {search_path}")


if __name__ == "__main__":
    main()
