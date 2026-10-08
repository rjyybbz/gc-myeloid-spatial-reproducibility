from __future__ import annotations

import csv
from pathlib import Path

from project_setup import PROJECT_PATHS, main as setup_main


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_inventory() -> list[dict]:
    return [
        {
            "dataset_id": "GSE206785",
            "priority": 1,
            "tier": "core",
            "data_type": "single_cell_rnaseq",
            "platform_scope": "10x_scRNA_processed_in_GEO",
            "disease_context": "gastric_cancer_treatment_naive",
            "therapy_context": "none",
            "reported_samples": 48,
            "reported_patients": 24,
            "response_annotation": "none",
            "study_role": "single_cell_discovery_primary",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE206785",
            "status": "confirmed",
            "notes": "Tumor and matched normal tissues from 24 treatment-naive gastric cancer patients; primary single-cell discovery cohort."
        },
        {
            "dataset_id": "GSE183904",
            "priority": 2,
            "tier": "core",
            "data_type": "single_cell_rnaseq",
            "platform_scope": "10x_scRNA_processed_in_GEO",
            "disease_context": "gastric_cancer_primary_plus_peritoneal",
            "therapy_context": "none",
            "reported_samples": 40,
            "reported_patients": 31,
            "response_annotation": "none",
            "study_role": "single_cell_discovery_reference",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183904",
            "status": "confirmed",
            "notes": "Large gastric cancer atlas with strong value for replication and subtype context."
        },
        {
            "dataset_id": "GSE163558",
            "priority": 3,
            "tier": "core",
            "data_type": "single_cell_rnaseq",
            "platform_scope": "10x_scRNA_raw_plus_processed",
            "disease_context": "gastric_cancer_primary_and_metastatic",
            "therapy_context": "none",
            "reported_samples": 10,
            "reported_patients": 6,
            "response_annotation": "none",
            "study_role": "single_cell_metastatic_extension",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE163558",
            "status": "confirmed",
            "notes": "Primary plus metastatic gastric cancer cohort for transferability and advanced-disease extension."
        },
        {
            "dataset_id": "GSE212212",
            "priority": 4,
            "tier": "supportive",
            "data_type": "single_cell_immune_focused",
            "platform_scope": "targeted_mRNA_TCR_BCR",
            "disease_context": "gastric_cancer_immune_microenvironment",
            "therapy_context": "none",
            "reported_samples": 44,
            "reported_patients": 11,
            "response_annotation": "none",
            "study_role": "immune_reference_support",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212212",
            "status": "confirmed",
            "notes": "Immune-focused support dataset with receptor data."
        },
        {
            "dataset_id": "GSE239676",
            "priority": 5,
            "tier": "supportive",
            "data_type": "single_cell_rnaseq_plus_immune_repertoire",
            "platform_scope": "tumor_metastasis_blood",
            "disease_context": "metastatic_gastric_cancer",
            "therapy_context": "mixed_or_unspecified",
            "reported_samples": 68,
            "reported_patients": 20,
            "response_annotation": "progression_and_immunotherapy_relevance",
            "study_role": "metastatic_biologic_support",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE239676",
            "status": "confirmed",
            "notes": "Primary, liver metastasis, peritoneal carcinomatosis, and blood; suitable for biologic extension."
        },
        {
            "dataset_id": "GSE270678",
            "priority": 6,
            "tier": "core",
            "data_type": "spatial_transcriptomics",
            "platform_scope": "10x_spatial_transcriptomics",
            "disease_context": "gastric_cancer_spatial_architecture",
            "therapy_context": "none",
            "reported_samples": 72,
            "reported_patients": 19,
            "response_annotation": "none",
            "study_role": "spatial_validation_primary",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270678",
            "status": "confirmed",
            "notes": "Primary spatial cohort for myeloid niche localization."
        },
        {
            "dataset_id": "GSE270680",
            "priority": 7,
            "tier": "core",
            "data_type": "single_cell_rnaseq",
            "platform_scope": "scRNA_companion_to_spatial",
            "disease_context": "gastric_cancer_spatial_atlas_companion",
            "therapy_context": "none",
            "reported_samples": 77,
            "reported_patients": "",
            "response_annotation": "none",
            "study_role": "single_cell_reference_for_spatial_mapping",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270680",
            "status": "confirmed",
            "notes": "Companion scRNA dataset for spatial reference mapping."
        },
        {
            "dataset_id": "GSE305116",
            "priority": 8,
            "tier": "supportive",
            "data_type": "spatial_transcriptomics",
            "platform_scope": "gastric_tumor_vs_normal",
            "disease_context": "intestinal_type_gastric_cancer",
            "therapy_context": "none",
            "reported_samples": 2,
            "reported_patients": "",
            "response_annotation": "none",
            "study_role": "spatial_supportive_secondary",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305116",
            "status": "confirmed",
            "notes": "Recent small spatial support dataset."
        },
        {
            "dataset_id": "GSE315928",
            "priority": 9,
            "tier": "core_supportive",
            "data_type": "single_cell_rnaseq_TCR",
            "platform_scope": "tumor_biopsy_chemoimmunotherapy",
            "disease_context": "advanced_gastric_cancer",
            "therapy_context": "frontline_5FU_platinum_plus_pembrolizumab",
            "reported_samples": 94,
            "reported_patients": 33,
            "response_annotation": "slow_progressor_vs_fast_progressor",
            "study_role": "treatment_linked_single_cell_validation",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE315928",
            "status": "confirmed_recent",
            "notes": "Key treatment-linked cohort, likely T-cell enriched but still strategic for coupling analysis."
        },
        {
            "dataset_id": "GSE229379",
            "priority": 10,
            "tier": "supportive",
            "data_type": "single_cell_rnaseq_TCR",
            "platform_scope": "circulating_immune_cells",
            "disease_context": "advanced_gastric_cancer",
            "therapy_context": "PD_1_blockade",
            "reported_samples": 12,
            "reported_patients": 6,
            "response_annotation": "treatment_outcome_context",
            "study_role": "peripheral_immune_validation",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE229379",
            "status": "confirmed",
            "notes": "Peripheral immune validation with treatment context."
        },
        {
            "dataset_id": "GSE183136",
            "priority": 11,
            "tier": "supportive",
            "data_type": "bulk_expression",
            "platform_scope": "tumor_bulk_signature_context",
            "disease_context": "gastric_cancer",
            "therapy_context": "contains_ICI_predictive_analysis_but_not_primary_response_cohort",
            "reported_samples": "",
            "reported_patients": "",
            "response_annotation": "indirect",
            "study_role": "bulk_supportive",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183136",
            "status": "candidate_supportive",
            "notes": "Secondary biologic or prognostic context layer."
        },
        {
            "dataset_id": "TCGA_STAD",
            "priority": 12,
            "tier": "core_extension",
            "data_type": "bulk_rnaseq_multiomics",
            "platform_scope": "transcriptome_multiomics",
            "disease_context": "stomach_adenocarcinoma",
            "therapy_context": "none",
            "reported_samples": "",
            "reported_patients": "",
            "response_annotation": "overall_survival_only",
            "study_role": "bulk_extension_core",
            "source_repository": "GDC",
            "source_url": "https://portal.gdc.cancer.gov/projects/TCGA-STAD",
            "status": "confirmed",
            "notes": "Use for biologic extension, not immunotherapy-proof claims."
        },
        {
            "dataset_id": "GSE62254",
            "priority": 13,
            "tier": "extension",
            "data_type": "bulk_expression",
            "platform_scope": "asian_gastric_bulk",
            "disease_context": "gastric_cancer",
            "therapy_context": "mixed_non_ICI",
            "reported_samples": "",
            "reported_patients": "",
            "response_annotation": "survival",
            "study_role": "bulk_external_extension",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE62254",
            "status": "confirmed",
            "notes": "Common gastric bulk validation cohort."
        },
        {
            "dataset_id": "GSE84437",
            "priority": 14,
            "tier": "extension",
            "data_type": "bulk_expression",
            "platform_scope": "large_gastric_bulk",
            "disease_context": "gastric_cancer",
            "therapy_context": "mixed_non_ICI",
            "reported_samples": "",
            "reported_patients": "",
            "response_annotation": "survival",
            "study_role": "bulk_external_extension",
            "source_repository": "GEO",
            "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84437",
            "status": "confirmed",
            "notes": "Large external gastric bulk cohort."
        }
    ]


def main() -> None:
    setup_main()
    rows = build_inventory()
    metadata_path = PROJECT_PATHS["metadata"] / "dataset_inventory_curated.csv"
    inventory_path = PROJECT_PATHS["inventory"] / "dataset_inventory_curated.csv"
    write_csv(metadata_path, rows)
    write_csv(inventory_path, rows)
    print(f"[inventory] wrote curated dataset inventory to {metadata_path}")
    print(f"[inventory] wrote curated dataset inventory to {inventory_path}")


if __name__ == "__main__":
    main()
