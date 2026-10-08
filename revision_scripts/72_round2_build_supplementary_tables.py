from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUND2_ROOT = PROJECT_ROOT
RESULTS = PROJECT_ROOT / "results"
OUT = ROUND2_ROOT / "supplementary_tables"
OUT.mkdir(parents=True, exist_ok=True)


SUPPRESSIVE_GENES = [
    "GPX1",
    "PSAP",
    "CSTB",
    "CTSB",
    "CTSC",
    "CTSD",
    "CTSL",
    "NPC2",
    "PLD3",
    "CD68",
    "PRDX1",
    "PLA2G7",
    "S100A11",
    "FABP5",
    "MS4A7",
    "FTL",
    "GPNMB",
    "ACP5",
    "GRN",
    "LIPA",
    "LGALS3",
    "CD63",
    "CTSA",
    "LGALS1",
    "APOC1",
    "TREM2",
    "C1QA",
    "C1QB",
    "APOE",
    "SOD2",
    "C1QC",
    "ATP6V1F",
    "TYROBP",
    "ANXA5",
]

REDUCED_SUPPRESSIVE_GENES = [
    "CTSB",
    "CTSC",
    "CTSL",
    "NPC2",
    "PRDX1",
    "S100A11",
    "FABP5",
    "MS4A7",
    "FTL",
    "ACP5",
    "LIPA",
    "LGALS3",
    "C1QA",
    "C1QB",
    "C1QC",
    "APOE",
    "SOD2",
    "ATP6V1F",
    "TYROBP",
    "ANXA5",
]

CAF_ACTIVATION_GENES = [
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "COL11A1",
    "POSTN",
    "THY1",
    "FAP",
    "CTHRC1",
    "INHBA",
    "TAGLN",
    "VSTM2A",
    "ADAM28",
]

CAF_INPUT_GENES = ["CCL2", "CXCL12", "CSF1", "IL34", "MIF", "TGFB1", "INHBA"]
FORMATION_LIGANDS = ["CCL2", "CSF1", "IL34", "INHBA", "MIF", "TGFB1", "CXCL12"]
EXPLORATORY_LIGANDS = ["CCL7", "IL6", "CXCL8"]
TLS_CHEMOKINE_GENES = ["FDCSP", "CCL19", "CCL21", "CXCL13", "LTB", "CXCR5", "CR2"]
TLS_BCELL_FDC_GENES = ["FDCSP", "CCL19", "CCL21", "CXCL13", "CR2", "BANK1", "MS4A1", "CD79A", "CD79B", "PAX5"]
RECEPTOR_SWITCH_GENES = [
    "CCR2",
    "CXCR4",
    "CSF1R",
    "CD74",
    "IL1R1",
    "IL1RAP",
    "TNFRSF1A",
    "TNFRSF1B",
    "TGFBR1",
    "TGFBR2",
    "CD44",
    "ITGAV",
    "ITGB1",
    "IL10RA",
    "IL10RB",
]
T_CELL_GENES = ["CD3D", "CD3E", "TRAC", "TRBC1", "IL7R", "LTB", "NKG7"]
T_CELL_EXHAUSTION_GENES = ["HAVCR2", "LAG3", "PDCD1", "TIGIT", "CTLA4", "TOX", "CXCL13"]
MYELOID_OUTPUT_GENES = ["LGALS9", "SPP1", "TGFB1", "IL10"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def n_rows(path: Path) -> int:
    return len(pd.read_csv(path))


def sum_col(path: Path, column: str) -> int:
    return int(pd.read_csv(path)[column].sum())


def unique_count(path: Path, column: str) -> int | str:
    df = pd.read_csv(path)
    if column not in df.columns:
        return "not available in analyzed manifest"
    return int(df[column].nunique())


def build_cohort_summary() -> pd.DataFrame:
    gse206785 = load_json(RESULTS / "myeloid_discovery" / "GSE206785_analysis_summary.json")
    gse163558 = load_json(RESULTS / "myeloid_validation" / "GSE163558_validation_summary.json")
    gse183904 = load_json(RESULTS / "myeloid_external_atlas" / "GSE183904_myeloid_validation_summary.json")
    gse270678 = load_json(RESULTS / "spatial_niche" / "GSE270678_spatial_niche_summary.json")
    gse251950 = load_json(RESULTS / "external_spatial_gse251950" / "GSE251950_external_spatial_summary.json")
    gse308624 = load_json(RESULTS / "external_spatial_gse308624" / "GSE308624_external_cosmx_summary.json")
    tcga = load_json(RESULTS / "bulk_transfer" / "TCGA_STAD_summary.json")
    gse84437 = load_json(RESULTS / "bulk_transfer" / "GSE84437_survival_summary.json")

    rows = [
        {
            "cohort": "GSE206785",
            "platform": "single-cell RNA-seq",
            "patients_or_donors": 24,
            "samples": 48,
            "cells_or_spots_analyzed": gse206785["myeloid_cells"],
            "tissue_type": "tumor and adjacent normal",
            "primary_metastatic_status": "primary gastric tumor and adjacent normal",
            "repeated_samples_same_patient": "yes; paired tumor/adjacent normal samples",
            "analytical_role": "primary single-cell discovery of myeloid state axis, CAF association, and curated LR hypotheses",
        },
        {
            "cohort": "GSE163558",
            "platform": "single-cell RNA-seq",
            "patients_or_donors": "not available in analyzed manifest",
            "samples": 10,
            "cells_or_spots_analyzed": int(gse163558["shape"][0]),
            "tissue_type": "primary tumor, adjacent normal, and metastatic sites",
            "primary_metastatic_status": "primary and metastatic gastric cancer",
            "repeated_samples_same_patient": "not available in analyzed manifest",
            "analytical_role": "external single-cell validation of suppressive-like program across tissue sites",
        },
        {
            "cohort": "GSE183904",
            "platform": "single-cell RNA-seq atlas",
            "patients_or_donors": "not available in analyzed manifest",
            "samples": 40,
            "cells_or_spots_analyzed": f"{gse183904['shape'][0]} total cells; {gse183904['myeloid_like_cells']} myeloid-like cells",
            "tissue_type": "stomach/peritoneum tumor and normal samples",
            "primary_metastatic_status": "primary gastric tumor plus peritoneal tumor subset",
            "repeated_samples_same_patient": "not available in analyzed manifest",
            "analytical_role": "external atlas validation and exploratory sample-level peritoneal analysis",
        },
        {
            "cohort": "GSE270678",
            "platform": "10x spatial transcriptomics",
            "patients_or_donors": unique_count(RESULTS / "spatial_niche" / "GSE270678_spatial_niche_sample_summary.csv", "donor"),
            "samples": gse270678["n_samples"],
            "cells_or_spots_analyzed": gse270678["n_spots_in_tissue"],
            "tissue_type": "gastric tumor tissue sections",
            "primary_metastatic_status": "primary tumor spatial sections",
            "repeated_samples_same_patient": "yes; multiple regions/replicates for some donors",
            "analytical_role": "primary spatial niche analysis and coordinate/module-adjusted sensitivity analysis",
        },
        {
            "cohort": "GSE251950",
            "platform": "spatial transcriptomics",
            "patients_or_donors": unique_count(RESULTS / "external_spatial_gse251950" / "GSE251950_sample_summary.csv", "pair_id"),
            "samples": gse251950["n_samples"],
            "cells_or_spots_analyzed": gse251950["n_spots_in_tissue"],
            "tissue_type": "primary gastric tumor and one paired peritoneal metastasis sample",
            "primary_metastatic_status": f"{gse251950['n_primary']} primary and {gse251950['n_peritoneal']} peritoneal-metastasis spatial samples",
            "repeated_samples_same_patient": "yes for the GC6 primary/peritoneal pair",
            "analytical_role": "external spatial support for CAF-coupled suppressive regions",
        },
        {
            "cohort": "GSE308624",
            "platform": "CosMx spatial molecular imaging",
            "patients_or_donors": "not available in analyzed manifest",
            "samples": gse308624["n_samples"],
            "cells_or_spots_analyzed": gse308624["n_cells"],
            "tissue_type": "gastric cancer tissue sections",
            "primary_metastatic_status": "tumor tissue sections; metastasis status not used for inference",
            "repeated_samples_same_patient": "not available in analyzed manifest",
            "analytical_role": "external cell-resolved spatial support for macrophage, fibroblast, and B-cell neighborhoods",
        },
        {
            "cohort": "TCGA-STAD",
            "platform": "bulk RNA-seq",
            "patients_or_donors": tcga["n_samples_total"],
            "samples": tcga["n_samples_total"],
            "cells_or_spots_analyzed": "bulk tumors",
            "tissue_type": "gastric adenocarcinoma bulk tumor",
            "primary_metastatic_status": "mostly primary tumors; survival/covariate metadata used where available",
            "repeated_samples_same_patient": "one bulk profile per case in analysis",
            "analytical_role": "bulk translational context and clinical-covariate-adjusted survival analysis",
        },
        {
            "cohort": "GSE84437",
            "platform": "bulk microarray",
            "patients_or_donors": gse84437["n_samples_total"],
            "samples": gse84437["n_samples_total"],
            "cells_or_spots_analyzed": "bulk tumors",
            "tissue_type": "gastric cancer bulk tumor",
            "primary_metastatic_status": "bulk tumor cohort with survival metadata",
            "repeated_samples_same_patient": "one bulk profile per case in analysis",
            "analytical_role": "external bulk translational context and exploratory dual-axis survival/ecology analysis",
        },
    ]
    return pd.DataFrame(rows)


def panel_row(name: str, genes: list[str], formula: str, derivation: str, validation: str, rationale: str) -> dict:
    return {
        "score_or_panel": name,
        "n_genes_or_features": len(genes),
        "genes_or_features": "; ".join(genes),
        "formula_or_algorithm": formula,
        "normalization_and_scaling": "Single-cell/spatial scores used normalized log-expression or scaled-expression module scoring as described in the Supplementary Methods; bulk scores used available gene expression transformed to within-cohort standardized module scores.",
        "weighting": "Equal weighting of available genes unless otherwise stated",
        "derivation_dataset": derivation,
        "validation_or_application_dataset": validation,
        "rationale_or_reference": rationale,
    }


def build_gene_panel_table() -> pd.DataFrame:
    rows = [
        panel_row(
            "putative myeloid suppressive-like program",
            SUPPRESSIVE_GENES,
            "mean module score across available program genes; discovery markers required support by at least two candidate annotations",
            "GSE206785 myeloid annotation markers",
            "GSE163558, GSE183904, GSE270678, GSE251950, GSE308624, TCGA-STAD, GSE84437",
            "Represents a mature TAM-related suppressive-like transcriptional ecology; not direct functional proof of suppression",
        ),
        panel_row(
            "reduced suppressive-like program excluding trajectory-overlap genes",
            REDUCED_SUPPRESSIVE_GENES,
            "mean module score after excluding genes overlapping top trajectory-axis early/late genes",
            "GSE206785 overlap audit",
            "GSE206785 sensitivity analysis",
            "Circularity sensitivity analysis requested during peer review",
        ),
        panel_row(
            "CAF activation score",
            CAF_ACTIVATION_GENES,
            "mean module score across CAF activation genes",
            "literature-informed and CAF marker-based panel applied to GSE206785",
            "GSE270678, GSE251950, GSE308624, bulk ecology context",
            "Captures activated fibroblast/stromal context",
        ),
        panel_row(
            "CAF myeloid-input score",
            CAF_INPUT_GENES,
            "mean module score across candidate CAF-to-myeloid ligands",
            "GSE206785 fibroblast ligand analysis",
            "GSE270678, GSE251950, GSE308624",
            "Captures candidate ligand inputs associated with myeloid-state organization",
        ),
        panel_row(
            "formation ligand panel",
            FORMATION_LIGANDS,
            "sender-group mean expression and detection fractions used in curated LR score",
            "GSE206785 fibroblast/myeloid formation analysis",
            "not used as an independent validation score",
            "Prespecified candidate ligands for CAF-myeloid communication hypotheses",
        ),
        panel_row(
            "exploratory ligand panel",
            EXPLORATORY_LIGANDS,
            "sender-group mean expression and detection fractions used for exploratory checks",
            "GSE206785",
            "not used for primary inference",
            "Exploratory inflammatory ligands only",
        ),
        panel_row(
            "TLS chemokine module",
            TLS_CHEMOKINE_GENES,
            "mean module score across TLS chemokine genes",
            "literature-informed TLS chemokine panel",
            "GSE270678, GSE251950",
            "Transcriptional/spatial module; not histologically confirmed TLS",
        ),
        panel_row(
            "TLS-related B-cell/FDC-core module",
            TLS_BCELL_FDC_GENES,
            "mean module score across B-cell/FDC-core genes",
            "literature-informed TLS/B-cell/FDC panel",
            "GSE270678, GSE251950, GSE308624",
            "Distinguishes TLS-like transcriptional signal from pathology-confirmed TLS",
        ),
        panel_row(
            "myeloid receptor-switching score",
            RECEPTOR_SWITCH_GENES,
            "mean module score across receptor panel; group ratios used a small offset where needed",
            "GSE206785 curated CAF/myeloid receptor panel",
            "GSE183904 peritoneal sample-level analysis",
            "Exploratory receptor-expression state descriptor",
        ),
        panel_row(
            "T-cell score",
            T_CELL_GENES,
            "mean module score across T-cell genes",
            "literature-informed T-cell marker panel",
            "GSE270678, bulk ecology context",
            "Adaptive immune-context comparator",
        ),
        panel_row(
            "T-cell exhaustion score",
            T_CELL_EXHAUSTION_GENES,
            "mean module score across exhaustion-associated genes",
            "literature-informed exhaustion marker panel",
            "GSE270678, bulk ecology context",
            "T-cell context comparator, not used to infer functional exhaustion alone",
        ),
        panel_row(
            "myeloid output score",
            MYELOID_OUTPUT_GENES,
            "mean module score across representative myeloid output genes",
            "GSE206785 and literature-informed output markers",
            "GSE270678",
            "Candidate output context, interpreted as transcriptional association",
        ),
        {
            "score_or_panel": "trajectory-axis score",
            "n_genes_or_features": "first 20 PCs",
            "genes_or_features": "highly variable genes used for PCA; root annotations: M02_Mono_CD14, M05_Mono_VCAN, M13_Mono_VCAN; terminal-like annotations: M04_Macro_C3, M07_Macro_APOE, M11_Macro_PLA2G7",
            "formula_or_algorithm": "normalized PCA-space projection of each myeloid cell from the root-annotation centroid toward the terminal-like centroid, min-max normalized to [0,1]",
            "normalization_and_scaling": "highly variable genes selected from normalized log-expression; scaled expression used for PCA; first 20 PCs used for the projection",
            "weighting": "PC-space projection rather than gene-level equal weighting",
            "derivation_dataset": "GSE206785",
            "validation_or_application_dataset": "GSE163558, GSE183904, GSE270678-derived spatial early/late module approximation",
            "rationale_or_reference": "Computational ordering of cross-sectional states; not lineage proof",
        },
        {
            "score_or_panel": "terminal-state receptor-switching index",
            "n_genes_or_features": "two sample-level standardized components",
            "genes_or_features": "trajectory-axis score z-score; receptor-switching score z-score",
            "formula_or_algorithm": "mean(z(trajectory-axis sample mean), z(receptor-switching sample mean))",
            "normalization_and_scaling": "z-scores calculated across the compared GSE183904 stomach/peritoneum tumor samples",
            "weighting": "equal weighting of the two standardized components",
            "derivation_dataset": "GSE183904 peritoneal analysis subset",
            "validation_or_application_dataset": "exploratory only; no independent peritoneal validation cohort",
            "rationale_or_reference": "Compact descriptor of terminal-like axis position plus receptor-panel state",
        },
    ]
    return pd.DataFrame(rows)


def build_lr_definition_table() -> pd.DataFrame:
    lr = pd.read_csv(RESULTS / "cell_crosstalk" / "GSE206785_curated_lr_scores.csv")
    lr.insert(0, "score_scope", "curated CAF/myeloid or suppressive-output LR hypothesis")
    lr["expression_definition"] = "group-level mean normalized log-expression for the ligand in sender cells and receptor in receiver cells"
    lr["detection_threshold"] = "fraction of cells in the group with normalized expression greater than 0"
    lr["raw_product_formula"] = "ligand_mean_expr * receptor_mean_expr * ligand_pct_expr * receptor_pct_expr"
    lr["stabilized_score_formula"] = "log1p(ligand_mean_expr) * log1p(receptor_mean_expr) * ligand_pct_expr * receptor_pct_expr"
    lr["aggregation_unit"] = "sender cell group to receiver cell group in GSE206785 tumor cells"
    lr["model_note"] = "curated expression-product score only; no NicheNet ligand-activity or target-gene modeling"
    return lr


def main() -> None:
    tables = {
        "Supplementary_Table_1_Cohort_summary.csv": build_cohort_summary(),
        "Supplementary_Table_2_Score_gene_panels_and_formulas.csv": build_gene_panel_table(),
        "Supplementary_Table_4_LR_expression_product_definition_and_pairs.csv": build_lr_definition_table(),
    }
    for name, table in tables.items():
        path = OUT / name
        table.to_csv(path, index=False)
        print(f"[table] wrote {path} ({len(table)} rows)")

    readme = OUT / "README_supplementary_tables_round2.txt"
    readme.write_text(
        "Round 2 supplementary tables for reviewer-requested reproducibility additions.\n"
        "Table 1: cohort-level platform/sample/cell or spot summary.\n"
        "Table 2: complete score gene panels, formulas, scaling notes, derivation datasets, validation/application datasets, and rationale.\n"
        "Table 3: GSE183904 sample-level peritoneal metrics and comparisons generated by scripts/71_round2_sample_level_peritoneal_tables.py.\n"
        "Table 4: curated ligand-receptor expression-product score definition, formulas, sender/receiver groups, and pair-level values.\n",
        encoding="utf-8",
    )
    print(f"[readme] wrote {readme}")


if __name__ == "__main__":
    main()
