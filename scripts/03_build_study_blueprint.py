from __future__ import annotations

import csv
from pathlib import Path

from project_setup import PROJECT_PATHS, main as setup_main


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_cohort_manifest(curated_rows: list[dict]) -> list[dict]:
    output = []
    core_required = {"GSE206785", "GSE183904", "GSE163558", "GSE270678", "GSE270680"}
    for row in sorted(curated_rows, key=lambda x: int(x["priority"])):
        output.append(
            {
                "priority": row["priority"],
                "dataset_id": row["dataset_id"],
                "tier": row["tier"],
                "study_role": row["study_role"],
                "must_have_for_core_paper": "TRUE" if row["dataset_id"] in core_required else "FALSE",
                "reason": row["notes"],
                "data_type": row["data_type"],
                "therapy_context": row["therapy_context"],
                "response_annotation": row["response_annotation"],
                "source_repository": row["source_repository"],
                "source_url": row["source_url"],
                "status": row["status"],
            }
        )
    return output


def build_analysis_plan() -> list[dict]:
    return [
        {
            "step_id": "A1",
            "analysis_step": "multi_cohort_inventory_and_metadata_lock",
            "primary_input": "curated public datasets",
            "primary_output": "dataset_inventory_and_manifest",
            "goal": "Lock the reproducible cohort framework before formal analysis."
        },
        {
            "step_id": "A2",
            "analysis_step": "single_cell_myeloid_discovery",
            "primary_input": "GSE206785 + GSE183904 + GSE163558",
            "primary_output": "candidate_suppressive_myeloid_states",
            "goal": "Define recurrent gastric myeloid states and identify a suppressive program."
        },
        {
            "step_id": "A3",
            "analysis_step": "trajectory_and_regulatory_inference",
            "primary_input": "myeloid_subclusters",
            "primary_output": "state_transition_and_regulon_evidence",
            "goal": "Infer monocyte-to-suppressive-state formation logic."
        },
        {
            "step_id": "A4",
            "analysis_step": "cell_cell_communication_and_tcell_coupling",
            "primary_input": "tumor_CAF_myeloid_Tcell_compartments",
            "primary_output": "ligand_receptor_and_coupling_table",
            "goal": "Connect the suppressive myeloid state to T-cell dysfunction and immune exclusion."
        },
        {
            "step_id": "A5",
            "analysis_step": "spatial_niche_localization",
            "primary_input": "GSE270678 + GSE270680",
            "primary_output": "spatial_enrichment_and_colocalization_metrics",
            "goal": "Place the suppressive myeloid program into a spatial niche."
        },
        {
            "step_id": "A6",
            "analysis_step": "treatment_linked_validation",
            "primary_input": "GSE315928 + GSE229379",
            "primary_output": "treatment_association_support",
            "goal": "Test whether the program associates with unfavorable treatment-linked immune states."
        },
        {
            "step_id": "A7",
            "analysis_step": "bulk_transfer_and_extension",
            "primary_input": "TCGA_STAD + GSE62254 + GSE84437 + optional_GSE183136",
            "primary_output": "bulk_score_and_external_context",
            "goal": "Transfer the single-cell program into bulk cohorts without overclaiming immunotherapy proof."
        }
    ]


def build_figure_plan() -> list[dict]:
    return [
        {
            "figure_id": "Figure 1",
            "working_title": "Study design and public cohort map",
            "data_source": "all curated cohorts",
            "main_message": "Show the discovery-validation-spatial architecture of the project."
        },
        {
            "figure_id": "Figure 2",
            "working_title": "Single-cell discovery of a suppressive myeloid state in gastric cancer",
            "data_source": "GSE206785; GSE183904; GSE163558",
            "main_message": "Identify recurrent gastric myeloid states and define the candidate suppressive axis."
        },
        {
            "figure_id": "Figure 3",
            "working_title": "Formation trajectory and regulatory network of the suppressive myeloid state",
            "data_source": "single-cell discovery cohorts",
            "main_message": "Reveal monocyte-to-suppressive-state transition and upstream regulators."
        },
        {
            "figure_id": "Figure 4",
            "working_title": "Tumor-myeloid-T cell communication links the state to dysfunction",
            "data_source": "single-cell discovery cohorts",
            "main_message": "Connect the myeloid state to T-cell exhaustion and immune exclusion."
        },
        {
            "figure_id": "Figure 5",
            "working_title": "Spatial mapping of the suppressive niche",
            "data_source": "GSE270678; GSE270680; optional GSE305116",
            "main_message": "Show where the program localizes within gastric tumor architecture."
        },
        {
            "figure_id": "Figure 6",
            "working_title": "Treatment-linked validation in gastric immunotherapy cohorts",
            "data_source": "GSE315928; GSE229379",
            "main_message": "Demonstrate treatment-related immune coupling and adverse response context."
        },
        {
            "figure_id": "Figure 7",
            "working_title": "Bulk transfer and integrated working model",
            "data_source": "TCGA-STAD; GSE62254; GSE84437",
            "main_message": "Translate the state into a transferable bulk score and summarize the model."
        }
    ]


def build_risk_register() -> list[dict]:
    return [
        {
            "risk_id": "R1",
            "risk": "Public gastric immunotherapy cohorts are limited and may not contain full myeloid compartments.",
            "impact": "high",
            "mitigation": "Use treatment-linked cohorts as supportive validation while keeping untreated tumor scRNA and spatial evidence as the mechanistic core."
        },
        {
            "risk_id": "R2",
            "risk": "Recent GEO records may have incomplete processed data or off-site raw files in NGDC/GSA.",
            "impact": "high",
            "mitigation": "Preserve direct source URLs, ftp family files, and external accession references in metadata before heavy analysis."
        },
        {
            "risk_id": "R3",
            "risk": "Overcalling MDSC identity without orthogonal validation.",
            "impact": "high",
            "mitigation": "Use suppressive myeloid state or MDSC-like terminology and anchor claims to transcriptomic programs."
        },
        {
            "risk_id": "R4",
            "risk": "Spatial datasets may emphasize lymphoid niches more than myeloid niches.",
            "impact": "medium",
            "mitigation": "Test both myeloid enrichment and myeloid-T cell exclusion patterns and keep a secondary spatial cohort as backup."
        }
    ]


def main() -> None:
    setup_main()
    curated_rows = read_csv(PROJECT_PATHS["metadata"] / "dataset_inventory_curated.csv")
    cohort_manifest = build_cohort_manifest(curated_rows)
    analysis_plan = build_analysis_plan()
    figure_plan = build_figure_plan()
    risk_register = build_risk_register()

    write_csv(PROJECT_PATHS["metadata"] / "cohort_manifest.csv", cohort_manifest)
    write_csv(PROJECT_PATHS["inventory"] / "cohort_manifest.csv", cohort_manifest)
    write_csv(PROJECT_PATHS["metadata"] / "analysis_plan.csv", analysis_plan)
    write_csv(PROJECT_PATHS["docs"] / "figure_plan.csv", figure_plan)
    write_csv(PROJECT_PATHS["docs"] / "risk_register.csv", risk_register)

    print(f"[blueprint] cohort manifest written to {PROJECT_PATHS['metadata'] / 'cohort_manifest.csv'}")
    print(f"[blueprint] analysis plan written to {PROJECT_PATHS['metadata'] / 'analysis_plan.csv'}")
    print(f"[blueprint] figure plan written to {PROJECT_PATHS['docs'] / 'figure_plan.csv'}")
    print(f"[blueprint] risk register written to {PROJECT_PATHS['docs'] / 'risk_register.csv'}")


if __name__ == "__main__":
    main()
