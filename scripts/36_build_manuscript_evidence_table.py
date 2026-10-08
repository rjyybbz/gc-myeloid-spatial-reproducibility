from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from project_setup import PROJECT_PATHS, main as setup_main


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def find_record(records: list[dict], **filters) -> dict:
    for record in records:
        matched = True
        for key, value in filters.items():
            if str(record.get(key)) != str(value):
                matched = False
                break
        if matched:
            return record
    raise KeyError(f"Record not found for filters: {filters}")


def format_pvalue(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if value == 0:
        return "0"
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def build_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "evidence_id",
        "section",
        "claim_short",
        "effect_summary",
        "pvalue",
        "manuscript_role",
    ]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def add_row(rows: list[dict], **kwargs) -> None:
    rows.append(kwargs)


def main() -> None:
    setup_main()

    formation = load_json(PROJECT_PATHS["results"] / "formation_inputs" / "GSE206785_formation_summary.json")
    spatial_niche = load_json(PROJECT_PATHS["results"] / "spatial_niche" / "GSE270678_spatial_niche_summary.json")
    spatial_tls = load_json(PROJECT_PATHS["results"] / "spatial_tls_exclusion" / "GSE270678_tls_exclusion_summary.json")
    external_spatial = load_json(
        PROJECT_PATHS["results"] / "external_spatial_gse251950" / "GSE251950_external_spatial_summary.json"
    )
    external_cosmx = load_json(
        PROJECT_PATHS["results"] / "external_spatial_gse308624" / "GSE308624_external_cosmx_summary.json"
    )
    mechanism = load_json(PROJECT_PATHS["results"] / "myeloid_external_mechanism" / "GSE183904_mechanism_summary.json")
    peritoneal = load_json(PROJECT_PATHS["results"] / "peritoneal_state_bias" / "GSE183904_peritoneal_state_bias_summary.json")
    benchmark = load_json(PROJECT_PATHS["results"] / "signature_benchmark" / "bulk_signature_benchmark_summary.json")
    bulk_ecology = load_json(PROJECT_PATHS["results"] / "signature_benchmark" / "bulk_ecology_context_summary.json")
    kim_avg = pd.read_csv(
        PROJECT_PATHS["results"] / "treatment_validation_direct_kim" / "Kim_direct_icb_composite_tests_avg.csv"
    )

    rows: list[dict] = []

    formation_axis = find_record(
        formation["key_correlations"],
        metric_x="act_vs_quies_fibro_fraction",
        metric_y="mean_myeloid_axis_score",
    )
    add_row(
        rows,
        evidence_id="E01",
        section="formation",
        dataset_id="GSE206785",
        claim_short="Activated CAF dominance tracks terminal myeloid polarization",
        claim_statement="Within tumor patients, activated-versus-quiescent fibroblast dominance is strongly associated with higher myeloid trajectory-axis score.",
        effect_summary=f"rho={formation_axis['rho']:.3f}",
        pvalue=format_pvalue(formation_axis["pvalue"]),
        statistic_type="Spearman rho",
        manuscript_role="main",
        source_file="results/formation_inputs/GSE206785_formation_summary.json",
    )

    inhb_bridge = find_record(
        formation["key_correlations"],
        metric_x="fibro_act_INHBA",
        metric_y="transition_frac_within_myeloid",
    )
    add_row(
        rows,
        evidence_id="E02",
        section="formation",
        dataset_id="GSE206785",
        claim_short="Fibroblast INHBA links to the transitional bridge",
        claim_statement="Fibroblast INHBA expression is connected to the transitional myeloid intermediate rather than only the terminal endpoint.",
        effect_summary=f"rho={inhb_bridge['rho']:.3f}",
        pvalue=format_pvalue(inhb_bridge["pvalue"]),
        statistic_type="Spearman rho",
        manuscript_role="main",
        source_file="results/formation_inputs/GSE206785_formation_summary.json",
    )

    spatial_caf_neighbor = find_record(spatial_niche["neighbor_summary"], target_flag="high_caf_activation")
    add_row(
        rows,
        evidence_id="E03",
        section="spatial",
        dataset_id="GSE270678",
        claim_short="Suppressive hotspots are CAF-rich",
        claim_statement="High suppressive-program spots are surrounded by more CAF-high neighbors than low-program spots.",
        effect_summary=f"neighbor delta={spatial_caf_neighbor['median_delta']:.3f}",
        pvalue=format_pvalue(spatial_caf_neighbor["pvalue"]),
        statistic_type="Wilcoxon on per-sample neighbor delta",
        manuscript_role="main",
        source_file="results/spatial_niche/GSE270678_spatial_niche_summary.json",
    )

    tls_corr = find_record(
        spatial_tls["correlation_summary"],
        metric_x="suppressive_program_score",
        metric_y="tls_bcell_fdc_score",
    )
    add_row(
        rows,
        evidence_id="E04",
        section="spatial",
        dataset_id="GSE270678",
        claim_short="Suppressive burden is inversely linked to TLS B-cell/FDC-core ecology",
        claim_statement="Within slices, higher suppressive-program spots are negatively associated with TLS B-cell/FDC-core signal.",
        effect_summary=f"median rho={tls_corr['median_delta']:.3f}",
        pvalue=format_pvalue(tls_corr["pvalue"]),
        statistic_type="Wilcoxon on per-sample rho",
        manuscript_role="main",
        source_file="results/spatial_tls_exclusion/GSE270678_tls_exclusion_summary.json",
    )

    tls_neighbor = find_record(spatial_tls["neighbor_summary"], target_flag="high_tls_bcell_fdc")
    add_row(
        rows,
        evidence_id="E05",
        section="spatial",
        dataset_id="GSE270678",
        claim_short="Suppressive hotspots exclude TLS-core neighbors",
        claim_statement="High suppressive-program spots have fewer nearby TLS B-cell/FDC-core hotspots than low-program spots.",
        effect_summary=f"neighbor delta={tls_neighbor['median_delta']:.3f}",
        pvalue=format_pvalue(tls_neighbor["pvalue"]),
        statistic_type="Wilcoxon on per-sample neighbor delta",
        manuscript_role="main",
        source_file="results/spatial_tls_exclusion/GSE270678_tls_exclusion_summary.json",
    )

    external_caf_corr = find_record(
        external_spatial["correlation_summary"],
        metric_x="suppressive_program_score",
        metric_y="caf_activation_score",
    )
    external_caf_neighbor = find_record(
        external_spatial["neighbor_summary"],
        target_flag="high_caf_activation",
    )
    add_row(
        rows,
        evidence_id="E05b",
        section="external_spatial",
        dataset_id="GSE251950",
        claim_short="Independent spatial cohort replicates the CAF-rich suppressive niche",
        claim_statement="In an independent gastric spatial cohort, higher suppressive-program areas again co-localize with CAF activation and are surrounded by more CAF-high neighbors.",
        effect_summary=(
            f"median rho={external_caf_corr['median_delta']:.3f}; "
            f"neighbor delta={external_caf_neighbor['median_delta']:.3f}"
        ),
        pvalue=format_pvalue(external_caf_neighbor["pvalue"]),
        statistic_type="Wilcoxon on per-sample rho / neighbor delta",
        manuscript_role="secondary",
        source_file="results/external_spatial_gse251950/GSE251950_external_spatial_summary.json",
    )

    external_cosmx_caf = find_record(external_cosmx["neighbor_summary"], target_flag="is_high_caf_fibro")
    add_row(
        rows,
        evidence_id="E05c",
        section="external_spatial",
        dataset_id="GSE308624",
        claim_short="External CosMx cohort confirms CAF-high fibroblast proximity around suppressive macrophages",
        claim_statement="In an independent 71-sample gastric CosMx cohort, high-suppressive macrophages are surrounded by more high-CAF fibroblast neighbors than low-suppressive macrophages.",
        effect_summary=f"neighbor delta={external_cosmx_caf['median_delta']:.3f}",
        pvalue=format_pvalue(external_cosmx_caf["pvalue"]),
        statistic_type="Wilcoxon on per-sample macrophage neighbor delta",
        manuscript_role="main",
        source_file="results/external_spatial_gse308624/GSE308624_external_cosmx_summary.json",
    )

    external_cosmx_tls = find_record(external_cosmx["neighbor_summary"], target_flag="is_high_tls_bcell")
    add_row(
        rows,
        evidence_id="E05d",
        section="external_spatial",
        dataset_id="GSE308624",
        claim_short="External CosMx cohort supports B-cell/TLS-like exclusion around suppressive macrophages",
        claim_statement="In the independent gastric CosMx cohort, high-suppressive macrophages have fewer nearby TLS-like high B-cell neighbors than low-suppressive macrophages.",
        effect_summary=f"neighbor delta={external_cosmx_tls['median_delta']:.3f}",
        pvalue=format_pvalue(external_cosmx_tls["pvalue"]),
        statistic_type="Wilcoxon on per-sample macrophage neighbor delta",
        manuscript_role="main",
        source_file="results/external_spatial_gse308624/GSE308624_external_cosmx_summary.json",
    )

    mechanism_switch = find_record(
        mechanism["state_tests"],
        metric="receptor_switch_score",
        group_col="mechanism_state",
        group_a="suppressive_like",
        group_b="early_like",
    )
    add_row(
        rows,
        evidence_id="E06",
        section="external_conservation",
        dataset_id="GSE183904",
        claim_short="External suppressive-like state is receptor-switched",
        claim_statement="In the external gastric atlas, suppressive-like myeloid cells show stronger receptor switching than early-like cells.",
        effect_summary=f"median {mechanism_switch['group_a']}={mechanism_switch['median_group_a']:.3f} vs {mechanism_switch['group_b']}={mechanism_switch['median_group_b']:.3f}",
        pvalue=format_pvalue(mechanism_switch["pvalue"]),
        statistic_type="Mann-Whitney U",
        manuscript_role="main",
        source_file="results/myeloid_external_mechanism/GSE183904_mechanism_summary.json",
    )

    add_row(
        rows,
        evidence_id="E07",
        section="peritoneal",
        dataset_id="GSE183904",
        claim_short="Peritoneum shifts toward suppressive-like over early-like balance",
        claim_statement="Peritoneal tumor myeloid cells are enriched for suppressive-like versus early-like states compared with stomach tumor.",
        effect_summary=f"OR={peritoneal['suppressive_vs_early_odds_ratio']:.3f}",
        pvalue=format_pvalue(peritoneal["suppressive_vs_early_fisher_pvalue"]),
        statistic_type="Fisher exact",
        manuscript_role="main",
        source_file="results/peritoneal_state_bias/GSE183904_peritoneal_state_bias_summary.json",
    )

    peritoneal_switch = find_record(
        peritoneal["sample_tests"],
        metric="receptor_switch_score",
        group_col="tissue_group",
    )
    add_row(
        rows,
        evidence_id="E08",
        section="peritoneal",
        dataset_id="GSE183904",
        claim_short="Peritoneal ecology is distinguished by receptor switching, not generic burden",
        claim_statement="At the sample level, peritoneal tumors are more strongly separated by receptor-switch score than by suppressive-program burden.",
        effect_summary=f"median peritoneum={peritoneal_switch['median_group_a']:.3f} vs stomach={peritoneal_switch['median_group_b']:.3f}",
        pvalue=format_pvalue(peritoneal_switch["pvalue"]),
        statistic_type="Mann-Whitney U",
        manuscript_role="main",
        source_file="results/peritoneal_state_bias/GSE183904_peritoneal_state_bias_summary.json",
    )

    pairwise_bulk = find_record(
        benchmark["top_pairwise_independence_signals"],
        dataset_id="TCGA_STAD",
        comparator_signature="trajectory_polarized_axis",
        term="our_suppressive_axis_std",
    )
    add_row(
        rows,
        evidence_id="E09",
        section="bulk_translation",
        dataset_id="TCGA_STAD",
        claim_short="Bulk suppressive burden retains signal beyond the trajectory axis",
        claim_statement="In TCGA-STAD, the bulk suppressive axis remains prognostically informative when modeled against the trajectory-polarized axis.",
        effect_summary=f"HR={pairwise_bulk['hazard_ratio']:.3f}",
        pvalue=format_pvalue(pairwise_bulk["p_value"]),
        statistic_type="Pairwise Cox model",
        manuscript_role="secondary",
        source_file="results/signature_benchmark/bulk_signature_benchmark_summary.json",
    )

    ecology_imbalance = find_record(
        bulk_ecology["key_group_tests"],
        dataset_id="TCGA_STAD",
        group_a="burden_high__polar_low",
        group_b="burden_high__polar_high",
        metric="stromal_adaptive_imbalance",
    )
    add_row(
        rows,
        evidence_id="E09b",
        section="bulk_translation",
        dataset_id="TCGA_STAD + GSE84437",
        claim_short="The worst dual-axis bulk group is stromal-high and adaptive-low",
        claim_statement="Across independent bulk gastric cohorts, the clinically unfavorable burden-high/polar-low group shows the highest stromal-adaptive imbalance, with higher CAF/TGFb context and lower Teff/IFNg/TLS-like context than burden-high/polar-high tumors.",
        effect_summary=f"TCGA imbalance p={format_pvalue(ecology_imbalance['pvalue'])}",
        pvalue=format_pvalue(ecology_imbalance["pvalue"]),
        statistic_type="Cross-dataset group ecology comparison",
        manuscript_role="secondary",
        source_file="results/signature_benchmark/bulk_ecology_context_summary.json",
    )

    kim_suppressive = kim_avg.loc[kim_avg["metric"].eq("suppressive_ecology_index")].iloc[0].to_dict()
    add_row(
        rows,
        evidence_id="E10",
        section="appendix_icb_context",
        dataset_id="Kim_STAD_pembrolizumab",
        claim_short="Published gastric ICB context supports the ecology model",
        claim_statement="In the published gastric pembrolizumab cohort, the suppressive ecology composite is enriched in nonresponders at the processed signature-context level.",
        effect_summary=f"AUC={kim_suppressive['response_oriented_auc']:.3f}",
        pvalue=format_pvalue(kim_suppressive["pvalue"]),
        statistic_type="Mann-Whitney + ROC AUC",
        manuscript_role="appendix",
        source_file="results/treatment_validation_direct_kim/Kim_direct_icb_composite_tests_avg.csv",
    )

    evidence_df = pd.DataFrame(rows)
    role_summary = evidence_df["manuscript_role"].value_counts().rename_axis("manuscript_role").reset_index(name="n_items")

    summary = {
        "n_evidence_items": int(evidence_df.shape[0]),
        "role_summary": role_summary.to_dict(orient="records"),
        "main_text_priority_order": evidence_df.loc[evidence_df["manuscript_role"].eq("main"), "evidence_id"].tolist(),
        "secondary_priority_order": evidence_df.loc[evidence_df["manuscript_role"].eq("secondary"), "evidence_id"].tolist(),
        "appendix_priority_order": evidence_df.loc[evidence_df["manuscript_role"].eq("appendix"), "evidence_id"].tolist(),
    }

    outdir = PROJECT_PATHS["results"] / "tables"
    write_csv(outdir / "manuscript_evidence_summary.csv", evidence_df)
    write_csv(outdir / "manuscript_evidence_role_summary.csv", role_summary)
    write_json(outdir / "manuscript_evidence_summary.json", summary)
    write_markdown(
        outdir / "manuscript_evidence_summary.md",
        "# Manuscript Evidence Summary\n\n" + build_markdown_table(evidence_df),
    )
    print(evidence_df.to_string(index=False))


if __name__ == "__main__":
    main()
