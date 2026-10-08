from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from project_setup import PROJECT_PATHS, write_json


MAIN_DIR = PROJECT_PATHS["figures"] / "manuscript" / "main"
SUPP_DIR = PROJECT_PATHS["figures"] / "manuscript" / "supplementary"
TABLE_DIR = PROJECT_PATHS["figures"] / "manuscript"

FIGURE_MANIFEST: list[dict] = []

COLORS = {
    "early": "#2B8C7E",
    "intermediate": "#D8A03D",
    "late": "#B24A39",
    "caf": "#D76B57",
    "caf_input": "#C95C45",
    "tls": "#325D88",
    "bcell": "#426F9B",
    "peritoneum": "#8F5D3A",
    "stomach": "#6C7A89",
    "grey": "#95A1AD",
    "dark": "#243447",
    "light": "#F6F4EF",
    "accent": "#5D8AA8",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D5D8DC",
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "text.color": COLORS["dark"],
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E4E7EB",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.85,
        }
    )


def load_csv(relative_path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(PROJECT_PATHS["root"] / relative_path, **kwargs)


def format_pvalue(pvalue: float) -> str:
    if pvalue < 1e-4:
        return f"P={pvalue:.1e}"
    return f"P={pvalue:.4f}"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.06,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["dark"],
    )


def soften_axes(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#D5D8DC")
    ax.spines["bottom"].set_color("#D5D8DC")
    ax.tick_params(length=3, color="#C7CDD4")


def add_stat_box(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    coords = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = coords[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": "#D6D8DB", "alpha": 0.96},
    )


def draw_round_box(
    ax: plt.Axes,
    xywh: tuple[float, float, float, float],
    edgecolor: str,
    facecolor: str = "#FFFFFF",
    linewidth: float = 1.5,
    alpha: float = 1.0,
    rounding: float = 0.03,
) -> FancyBboxPatch:
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def draw_chip(
    ax: plt.Axes,
    xywh: tuple[float, float, float, float],
    text: str,
    facecolor: str,
    *,
    textcolor: str = "#FFFFFF",
    edgecolor: str | None = None,
    fontsize: float = 8.0,
    bold: bool = True,
) -> FancyBboxPatch:
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=edgecolor or facecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else None,
        color=textcolor,
    )
    return patch


def add_section_tag(ax: plt.Axes, text: str, color: str, x: float = 0.02, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": mcolors.to_rgba(color, 0.10), "edgecolor": color, "linewidth": 1.0},
    )


def add_arrow_path(
    ax: plt.Axes,
    centers: list[tuple[float, float]],
    color: str = "#6A7682",
    rad: float = 0.12,
) -> None:
    for (x1, y1), (x2, y2) in zip(centers[:-1], centers[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=14,
                linewidth=1.6,
                color=color,
                alpha=0.92,
                zorder=5,
            )
        )


def add_state_centroids(
    ax: plt.Axes,
    df: pd.DataFrame,
    state_col: str,
    order: list[str],
    label_map: dict[str, str],
    x_col: str = "umap_1",
    y_col: str = "umap_2",
) -> None:
    centers: list[tuple[float, float]] = []
    for state in order:
        sub = df[df[state_col] == state]
        if sub.empty:
            continue
        center = (float(sub[x_col].median()), float(sub[y_col].median()))
        centers.append(center)
        ax.text(
            center[0],
            center[1],
            label_map.get(state, state),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["dark"],
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FFFFFF", "edgecolor": "#D8DDE3", "alpha": 0.95},
            zorder=6,
        )
    if len(centers) >= 2:
        add_arrow_path(ax, centers)


def draw_cell_cloud(
    ax: plt.Axes,
    center: tuple[float, float],
    color: str,
    *,
    scale: float = 1.0,
    transform=None,
    halo_alpha: float = 0.11,
    cell_alpha: float = 0.90,
    edgecolor: str | None = None,
) -> None:
    transform = transform or ax.transAxes
    cx, cy = center
    halo_w = 0.17 * scale
    halo_h = 0.12 * scale
    offsets = [(-0.042, 0.000), (0.000, 0.032), (0.040, -0.020), (-0.008, -0.042)]
    ax.add_patch(
        Ellipse(
            (cx, cy),
            halo_w,
            halo_h,
            transform=transform,
            facecolor=mcolors.to_rgba(color, halo_alpha),
            edgecolor=edgecolor or color,
            linewidth=1.4,
        )
    )
    for dx, dy in offsets:
        ax.add_patch(
            Circle(
                (cx + dx * scale, cy + dy * scale),
                0.015 * scale,
                transform=transform,
                facecolor=mcolors.to_rgba(color, cell_alpha),
                edgecolor="white",
                linewidth=0.7,
            )
        )


def plot_bubble_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap,
    cbar_label: str,
    size_scale: float = 700.0,
    xtick_rotation: float = 35.0,
    tick_fontsize: float = 8.5,
    cbar_width: str = "5%",
) -> None:
    mat = matrix.astype(float).copy()
    rows = list(mat.index)
    cols = list(mat.columns)
    values = mat.to_numpy()
    vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    xs: list[int] = []
    ys: list[int] = []
    colors: list[float] = []
    sizes: list[float] = []
    for yi, row in enumerate(rows):
        for xi, col in enumerate(cols):
            value = float(mat.loc[row, col])
            xs.append(xi)
            ys.append(yi)
            colors.append(value)
            sizes.append(40 + size_scale * abs(value) / vmax)

    sc = ax.scatter(
        xs,
        ys,
        c=colors,
        s=sizes,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    for x in np.arange(-0.5, len(cols), 1.0):
        ax.axvline(x, color="#EEF1F4", lw=0.8, zorder=0)
    for y in np.arange(-0.5, len(rows), 1.0):
        ax.axhline(y, color="#EEF1F4", lw=0.8, zorder=0)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=xtick_rotation, ha="right", fontsize=tick_fontsize)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=tick_fontsize)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = inset_axes(ax, width=cbar_width, height="72%", loc="lower left", bbox_to_anchor=(1.02, 0.14, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = plt.colorbar(sc, cax=cax)
    cb.outline.set_edgecolor("#D6DCE3")
    cb.set_label(cbar_label, fontsize=8.2)
    cb.ax.tick_params(labelsize=7.8)


def compute_focus_window(
    sample_df: pd.DataFrame,
    metric: str,
    quantile: float = 0.94,
    bins: int = 6,
    pad: float = 5.0,
) -> tuple[float, float, float, float]:
    work = sample_df.copy()
    high = work[work[metric] >= work[metric].quantile(quantile)].copy()
    if high.empty:
        return (
            float(work["array_col"].min()),
            float(work["array_col"].max()),
            float(work["array_row"].min()),
            float(work["array_row"].max()),
        )
    high["xbin"] = pd.cut(high["array_col"], bins=bins)
    high["ybin"] = pd.cut(high["array_row"], bins=bins)
    counts = high.groupby(["xbin", "ybin"], observed=False).size().reset_index(name="n").sort_values("n", ascending=False)
    top = counts.iloc[0]
    xmin = max(float(work["array_col"].min()), float(top["xbin"].left) - pad)
    xmax = min(float(work["array_col"].max()), float(top["xbin"].right) + pad)
    ymin = max(float(work["array_row"].min()), float(top["ybin"].left) - pad)
    ymax = min(float(work["array_row"].max()), float(top["ybin"].right) + pad)
    return xmin, xmax, ymin, ymax


def save_figure(fig: plt.Figure, stem: str, title: str, source_files: list[str], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "pdf", "svg"):
        out_path = folder / f"{stem}.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        outputs.append(str(out_path.relative_to(PROJECT_PATHS["root"])).replace("\\", "/"))
    plt.close(fig)
    FIGURE_MANIFEST.append(
        {
            "figure_id": stem,
            "title": title,
            "outputs": outputs,
            "source_files": source_files,
        }
    )


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0), axis=0)


def km_curve(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(durations) & np.isfinite(events)
    durations = durations[mask].astype(float)
    events = events[mask].astype(int)
    if len(durations) == 0:
        return np.array([0.0]), np.array([1.0])
    event_times = np.sort(np.unique(durations[events == 1]))
    times = [0.0]
    surv = [1.0]
    current = 1.0
    for t in event_times:
        at_risk = np.sum(durations >= t)
        n_event = np.sum((durations == t) & (events == 1))
        if at_risk > 0:
            current *= 1.0 - (n_event / at_risk)
        times.extend([t, t])
        surv.extend([surv[-1], current])
    times.append(float(np.nanmax(durations)))
    surv.append(current)
    return np.array(times), np.array(surv)


def km_survival_at(durations: np.ndarray, events: np.ndarray, timepoint: float) -> float:
    times, surv = km_curve(durations, events)
    idx = np.searchsorted(times, timepoint, side="right") - 1
    idx = max(0, min(idx, len(surv) - 1))
    return float(surv[idx])


def plot_km(ax: plt.Axes, df: pd.DataFrame, dataset_id: str, order: list[str], title: str) -> None:
    dataset = df[df["dataset_id"] == dataset_id].copy()
    dataset = dataset.dropna(subset=["os_months", "os_event", "dual_axis_group"])
    palette = {
        "burden_high__polar_low": COLORS["late"],
        "burden_high__polar_high": COLORS["caf"],
        "burden_low__polar_high": COLORS["early"],
        "burden_low__polar_low": COLORS["grey"],
    }
    label_map = {
        "burden_high__polar_low": "HS | LAI",
        "burden_high__polar_high": "HS | HAI",
        "burden_low__polar_high": "LS | HAI",
        "burden_low__polar_low": "LS | LAI",
    }
    highlight_group = "burden_high__polar_low"
    ax.axvline(60, color="#D3D8DE", lw=1.0, ls="--")
    for group in order:
        sub = dataset[dataset["dual_axis_group"] == group]
        if sub.empty:
            continue
        times, surv = km_curve(sub["os_months"].to_numpy(), sub["os_event"].to_numpy())
        lw = 2.7 if group == highlight_group else 2.1
        alpha = 1.0 if group == highlight_group else 0.88
        ax.step(times, surv, where="post", lw=lw, color=palette[group], alpha=alpha, label=f"{label_map[group]} (n={len(sub)})")
        y60 = km_survival_at(sub["os_months"].to_numpy(), sub["os_event"].to_numpy(), 60)
        ax.scatter([60], [y60], s=28 if group == highlight_group else 22, c=palette[group], edgecolor="white", linewidth=0.7, zorder=4, alpha=alpha)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(title, fontsize=11.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.text(60.5, 0.98, "60 m", fontsize=7.8, color="#7B8794", va="top")
    soften_axes(ax)


def plot_study_schematic(ax: plt.Axes) -> None:
    ax.axis("off")
    evidence = load_csv("results/tables/manuscript_evidence_summary.csv")

    ax.text(0.04, 0.93, "Gastric myeloid suppressive ecology: study design", transform=ax.transAxes, fontsize=17, fontweight="bold")
    ax.text(
        0.04,
        0.885,
        "Public single-cell, spatial, CosMx, and bulk cohorts were integrated into a state-to-context evidence chain.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#52606D",
    )

    draw_round_box(ax, (0.035, 0.67, 0.93, 0.16), edgecolor="#D5DDE6", facecolor="#FBFCFD", linewidth=1.2, rounding=0.02)
    ax.text(0.055, 0.79, "Data foundation", transform=ax.transAxes, fontsize=11.2, fontweight="bold")
    cohort_cards = [
        ("Discovery scRNA", "GSE206785\nmyeloid atlas / trajectory", COLORS["early"]),
        ("External scRNA", "GSE163558 + GSE183904\nmetastatic / atlas extension", COLORS["intermediate"]),
        ("Spatial layers", "GSE270678 + GSE251950\nprimary + replicated ST", COLORS["tls"]),
        ("Cell-resolved niche", "GSE308624\nCosMx macrophage niches", COLORS["accent"]),
        ("Clinical transfer", "TCGA-STAD + GSE84437\nbulk risk transfer", COLORS["peritoneum"]),
    ]
    x_positions = [0.055, 0.245, 0.435, 0.625, 0.815]
    widths = [0.16, 0.16, 0.16, 0.16, 0.13]
    for (title, body, color), x, w in zip(cohort_cards, x_positions, widths):
        draw_round_box(ax, (x, 0.695, w, 0.10), edgecolor=color, facecolor=mcolors.to_rgba(color, 0.95), linewidth=1.2, rounding=0.02)
        ax.text(x + 0.012, 0.767, title, transform=ax.transAxes, fontsize=9.0, fontweight="bold", color="#FFFFFF")
        ax.text(x + 0.012, 0.728, body, transform=ax.transAxes, fontsize=7.5, color="#F8FBFD", va="top")

    ax.text(0.055, 0.60, "Core evidence chain", transform=ax.transAxes, fontsize=11.2, fontweight="bold")
    lane_specs = [
        ("Discovery", "Early-bridge-late axis\nfrom tumor myeloid cells", COLORS["early"], (0.055, 0.42, 0.17, 0.12)),
        ("Formation", "Activated CAF signals and\nINHBA-associated bridge", COLORS["caf"], (0.275, 0.42, 0.17, 0.12)),
        ("Spatial niche", "CAF-rich suppressive hotspots\nwith lower TLS-related core signal", COLORS["tls"], (0.495, 0.42, 0.18, 0.12)),
        ("Translation", "Exploratory peritoneal receptor panel\nand bulk ecological context", COLORS["peritoneum"], (0.725, 0.42, 0.18, 0.12)),
    ]
    for title, body, color, xywh in lane_specs:
        x, y, w, h = xywh
        draw_round_box(ax, xywh, edgecolor=color, facecolor="#FFFFFF", linewidth=1.6, rounding=0.025)
        ax.text(x + 0.018, y + h - 0.045, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=color)
        ax.text(x + 0.018, y + 0.03, body, transform=ax.transAxes, fontsize=9.7, color=COLORS["dark"], va="bottom")
    for start, end in [((0.225, 0.48), (0.275, 0.48)), ((0.445, 0.48), (0.495, 0.48)), ((0.675, 0.48), (0.725, 0.48))]:
        ax.add_patch(
            FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=18, linewidth=1.9, color="#7B8794")
        )

    ax.text(0.055, 0.365, "Headline evidence", transform=ax.transAxes, fontsize=11.2, fontweight="bold")
    evidence_items = [
        ("E01", "CAF dominance associated with myeloid-axis score", "rho=0.806", COLORS["caf"]),
        ("E02", "INHBA marks a candidate bridge state", "rho=0.468", COLORS["intermediate"]),
        ("E03", "Suppressive hotspots are CAF-rich", "delta=0.095", COLORS["caf"]),
        ("E04-E05", "TLS-related core signal is lower", "rho=-0.102; delta=-0.028", COLORS["tls"]),
        ("E07-E08", "Peritoneal receptor-panel signal is exploratory", "index P=0.045", COLORS["peritoneum"]),
        ("E09-E09b", "Bulk context is stromal-high/adaptive-low", "HR=1.254", COLORS["late"]),
    ]
    x0, y0 = 0.055, 0.055
    w, h, gap = 0.145, 0.12, 0.015
    for idx, (eid, title, metric, color) in enumerate(evidence_items):
        col = idx % 3
        row = idx // 3
        x = x0 + col * (w + gap)
        y = y0 + (1 - row) * (h + 0.03)
        draw_round_box(ax, (x, y, w, h), edgecolor=color, facecolor=mcolors.to_rgba(color, 0.08), linewidth=1.3, rounding=0.02)
        ax.text(x + 0.012, y + h - 0.035, eid, transform=ax.transAxes, fontsize=8.8, fontweight="bold", color=color)
        ax.text(x + 0.012, y + 0.048, title, transform=ax.transAxes, fontsize=8.9, color=COLORS["dark"], va="bottom")
        ax.text(x + 0.012, y + 0.018, metric, transform=ax.transAxes, fontsize=8.3, color="#6B7785")

    ax.text(
        0.705,
        0.20,
        "Cohorts used in the main narrative:\nGSE206785, GSE163558, GSE183904, GSE270678,\nGSE251950, GSE308624, TCGA-STAD, GSE84437",
        transform=ax.transAxes,
        fontsize=9.4,
        color=COLORS["dark"],
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFFFFF", "edgecolor": "#D6D8DB"},
    )


def plot_discovery_umap(ax: plt.Axes, umap_df: pd.DataFrame) -> None:
    group_map = {
        "Bridge": ["M08_Mono_INHBA"],
        "Early": ["M02_Mono_CD14", "M05_Mono_VCAN", "M13_Mono_VCAN"],
        "Late suppressive": [
            "M04_Macro_C3",
            "M06_Macro_SLC40A1",
            "M07_Macro_APOE",
            "M09_Macro_F13A1",
            "M11_Macro_PLA2G7",
            "M14_Macro_CCL2",
        ],
    }
    umap_df = umap_df.copy()
    umap_df["state_group"] = "Other myeloid"
    for label, clusters in group_map.items():
        umap_df.loc[umap_df["Annotation"].isin(clusters), "state_group"] = label
    palette = {
        "Other myeloid": "#D8DDE3",
        "Early": COLORS["early"],
        "Bridge": COLORS["intermediate"],
        "Late suppressive": COLORS["late"],
    }
    order = ["Other myeloid", "Early", "Bridge", "Late suppressive"]
    for group in order:
        sub = umap_df[umap_df["state_group"] == group]
        ax.scatter(sub["umap_1"], sub["umap_2"], s=6, c=palette[group], linewidths=0, alpha=0.82, rasterized=True)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    handles = [mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=palette[g], markersize=6, label=g) for g in order[1:]]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    add_state_centroids(
        ax,
        umap_df,
        state_col="state_group",
        order=["Early", "Bridge", "Late suppressive"],
        label_map={"Early": "Early", "Bridge": "Bridge", "Late suppressive": "Late"},
    )
    add_section_tag(ax, "Discovery atlas", COLORS["early"])
    soften_axes(ax)


def plot_trajectory_umap(ax: plt.Axes, traj_df: pd.DataFrame) -> None:
    order = ["early", "intermediate", "late"]
    palette = {"early": COLORS["early"], "intermediate": COLORS["intermediate"], "late": COLORS["late"]}
    for state in order:
        sub = traj_df[traj_df["trajectory_state"] == state]
        ax.scatter(sub["umap_1"], sub["umap_2"], s=6, c=palette[state], linewidths=0, alpha=0.84, rasterized=True)
    ax.set_xlabel("Trajectory UMAP-1")
    ax.set_ylabel("Trajectory UMAP-2")
    handles = [mpl.lines.Line2D([0], [0], marker="o", linestyle="", color=palette[g], markersize=6, label={"early": "Early", "intermediate": "Bridge", "late": "Late"}[g]) for g in order]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    add_state_centroids(
        ax,
        traj_df,
        state_col="trajectory_state",
        order=["early", "intermediate", "late"],
        label_map={"early": "Early", "intermediate": "Bridge", "late": "Late"},
    )
    add_section_tag(ax, "Trajectory embedding", COLORS["intermediate"])
    soften_axes(ax)


def plot_validation_lollipop(ax: plt.Axes, tissue_df: pd.DataFrame) -> None:
    tissue_df = tissue_df.sort_values("mean")
    colors = [COLORS["peritoneum"] if "peritoneal" in x else COLORS["stomach"] if "primary" in x or "normal" in x else COLORS["accent"] for x in tissue_df["tissue_group"]]
    y = np.arange(len(tissue_df))
    ax.hlines(y, xmin=0, xmax=tissue_df["mean"], color="#C8CED6", linewidth=2)
    ax.scatter(tissue_df["mean"], y, s=90, c=colors, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_yticks(y)
    short_labels = {
        "adjacent_normal": "Adjacent normal",
        "liver_met": "Liver met",
        "lymph_node_met": "Lymph-node met",
        "primary_tumor": "Primary",
        "peritoneal_met": "Peritoneal met",
        "ovary_met": "Ovary met",
    }
    ax.set_yticklabels([short_labels.get(x, x.replace("_", " ")) for x in tissue_df["tissue_group"]], fontsize=8.5)
    ax.set_xlabel("Mean suppressive-program score")
    add_section_tag(ax, "Metastatic extension", COLORS["peritoneum"])
    soften_axes(ax)


def plot_patient_scatter(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, color: str, evidence_text: str) -> None:
    sub = df[[x, y, "Patient"]].dropna().copy()
    sns.regplot(
        data=sub,
        x=x,
        y=y,
        scatter=False,
        line_kws={"color": color, "lw": 2.2},
        ci=None,
        ax=ax,
    )
    ax.scatter(sub[x], sub[y], s=48, c=color, edgecolor="white", linewidth=1.0, alpha=0.88)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    add_stat_box(ax, evidence_text)
    soften_axes(ax)


def plot_ligand_heatmap(ax: plt.Axes, ligand_df: pd.DataFrame) -> None:
    ligand_df = ligand_df.copy().sort_values("activated_minus_quiescent", ascending=False).head(8)
    mat = ligand_df.set_index("gene")[["fibro_activated", "fibro_quiescent", "epithelial", "myeloid_transition"]]
    mat = zscore_frame(np.log1p(mat))
    sns.heatmap(
        mat,
        cmap=sns.color_palette("blend:#FBFAF7,#D76B57,#7D2E1F", as_cmap=True),
        ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Within-column z-score"},
        linewidths=0.6,
        linecolor="#F3F0EA",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_lr_lollipop(ax: plt.Axes, lr_df: pd.DataFrame) -> None:
    lr_df = lr_df.sort_values("stabilized_score").tail(6).copy()
    lr_df["short_label"] = lr_df["ligand"] + "→" + lr_df["receptor"]
    color_map = {
        "fibroblast_activated_tumor": COLORS["caf"],
        "epithelial_tumor": COLORS["accent"],
        "myeloid_transition_inhba_tumor": COLORS["intermediate"],
    }
    y = np.arange(len(lr_df))
    ax.hlines(y, 0, lr_df["stabilized_score"], color="#CFD5DB", linewidth=2)
    ax.scatter(
        lr_df["stabilized_score"],
        y,
        s=110,
        c=[color_map.get(x, COLORS["grey"]) for x in lr_df["sender_group"]],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(lr_df["short_label"], fontsize=8.0)
    ax.set_xlabel("Stabilized ligand-receptor score")
    add_stat_box(ax, "Dominant edges are CAF-derived", loc="lower right")
    soften_axes(ax)


def build_figure_1() -> None:
    fig, ax = plt.subplots(figsize=(17, 8.8))
    plot_study_schematic(ax)
    save_figure(
        fig,
        "Figure1_study_workflow",
        "Study workflow for dissecting CAF-shaped myeloid suppressive ecology in gastric cancer",
        [
            "results/tables/manuscript_evidence_summary.csv",
            "metadata/cohort_manifest.csv",
        ],
        MAIN_DIR,
    )


def build_figure_1b() -> None:
    myeloid_umap = load_csv("results/myeloid_discovery/GSE206785_myeloid_umap.csv")
    traj = load_csv("results/mechanism_trajectory/GSE206785_trajectory_embedding.csv")
    tissue_sum = load_csv("results/myeloid_validation/GSE163558_myeloid_tissue_summary.csv")
    decile = load_csv("results/mechanism_trajectory/GSE206785_trajectory_decile_summary.csv")

    fig = plt.figure(figsize=(15.8, 8.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.22, 1.0, 0.98], height_ratios=[1.0, 0.88], hspace=0.28, wspace=0.26)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[:, 2])

    plot_discovery_umap(ax_a, myeloid_umap)
    plot_trajectory_umap(ax_b, traj)
    x = np.arange(len(decile))
    ax_c.axvspan(-0.4, 2.4, color=mcolors.to_rgba(COLORS["early"], 0.08))
    ax_c.axvspan(2.6, 5.4, color=mcolors.to_rgba(COLORS["intermediate"], 0.08))
    ax_c.axvspan(5.6, 9.4, color=mcolors.to_rgba(COLORS["late"], 0.08))
    ax_c.plot(x, decile["mean_program_score"], color=COLORS["late"], lw=2.7, marker="o", ms=5.0)
    ax_c.fill_between(x, decile["mean_program_score"], color=mcolors.to_rgba(COLORS["late"], 0.14))
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(decile["trajectory_decile"])
    ax_c.set_xlabel("Trajectory decile")
    ax_c.set_ylabel("Mean suppressive-program score")
    ax_c.text(0.12, 0.92, "Early", transform=ax_c.transAxes, fontsize=8.8, fontweight="bold", color=COLORS["early"])
    ax_c.text(0.45, 0.92, "Bridge", transform=ax_c.transAxes, fontsize=8.8, fontweight="bold", color=COLORS["intermediate"])
    ax_c.text(0.79, 0.92, "Late", transform=ax_c.transAxes, fontsize=8.8, fontweight="bold", color=COLORS["late"])
    add_section_tag(ax_c, "Program accumulation", COLORS["late"])
    soften_axes(ax_c)
    plot_validation_lollipop(ax_d, tissue_sum)
    add_stat_box(ax_d, "External burden rises\nin peritoneal and ovary metastasis", loc="upper left")

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    save_figure(
        fig,
        "Figure2_discovery_validation_axis",
        "Discovery and external validation define the gastric myeloid suppressive trajectory axis",
        [
            "results/myeloid_discovery/GSE206785_myeloid_umap.csv",
            "results/mechanism_trajectory/GSE206785_trajectory_embedding.csv",
            "results/mechanism_trajectory/GSE206785_trajectory_decile_summary.csv",
            "results/myeloid_validation/GSE163558_myeloid_tissue_summary.csv",
        ],
        MAIN_DIR,
    )


def build_figure_1c() -> None:
    patient_metrics = load_csv("results/formation_inputs/GSE206785_formation_patient_metrics.csv")
    ligand_sum = load_csv("results/formation_inputs/GSE206785_formation_ligand_group_summary.csv")
    lr_scores = load_csv("results/cell_crosstalk/GSE206785_top_formative_lr_scores.csv")

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_patient_scatter(
        ax_a,
        patient_metrics,
        "act_vs_quies_fibro_fraction",
        "mean_myeloid_axis_score",
        "Activated CAF dominance",
        "Mean myeloid axis score",
        COLORS["caf"],
        "E01\nrho=0.806\nP=5.90e-06",
    )
    plot_patient_scatter(
        ax_b,
        patient_metrics,
        "fibro_act_INHBA_z",
        "transition_frac_in_trajectory",
        "Fibroblast INHBA z-score",
        "Transition fraction in trajectory",
        COLORS["intermediate"],
        "E02\nrho=0.468\nP=0.0280",
    )
    plot_ligand_heatmap(ax_c, ligand_sum)
    plot_lr_lollipop(ax_d, lr_scores)

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    save_figure(
        fig,
        "Figure3_caf_shaped_formation_mechanism",
        "CAF-shaped inputs define the formation mechanism of the suppressive myeloid state",
        [
            "results/formation_inputs/GSE206785_formation_patient_metrics.csv",
            "results/formation_inputs/GSE206785_formation_ligand_group_summary.csv",
            "results/cell_crosstalk/GSE206785_top_formative_lr_scores.csv",
        ],
        MAIN_DIR,
    )


def select_representative_spatial_sample() -> str:
    niche = load_csv("results/spatial_niche/GSE270678_spatial_niche_sample_summary.csv")
    tls = load_csv("results/spatial_tls_exclusion/GSE270678_tls_exclusion_sample_summary.csv")
    merged = niche.merge(tls[["sample_id", "tls_bcell_fdc_mean"]], on="sample_id", how="left")
    for col in ["suppressive_program_mean", "caf_activation_mean"]:
        merged[col + "_rank"] = merged[col].rank(pct=True)
    merged["tls_bcell_fdc_mean_rank"] = merged["tls_bcell_fdc_mean"].rank(pct=True)
    merged["selection_score"] = merged["suppressive_program_mean_rank"] + merged["caf_activation_mean_rank"] - merged["tls_bcell_fdc_mean_rank"]
    return merged.sort_values("selection_score", ascending=False).iloc[0]["sample_id"]


def plot_spatial_map(
    ax: plt.Axes,
    sample_df: pd.DataFrame,
    metric: str,
    cmap: str,
    title: str,
    focus_window: tuple[float, float, float, float] | None = None,
    show_colorbar: bool = True,
    highlight_barcodes: set[str] | None = None,
) -> None:
    vals = sample_df[metric].to_numpy()
    vmin, vmax = np.nanpercentile(vals, [5, 95])
    ax.scatter(
        sample_df["array_col"],
        -sample_df["array_row"],
        c="#E8EBEF",
        s=18,
        marker="s",
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    sc = ax.scatter(
        sample_df["array_col"],
        -sample_df["array_row"],
        c=sample_df[metric],
        cmap=cmap,
        s=18,
        marker="s",
        linewidths=0,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        zorder=2,
    )
    if highlight_barcodes is not None:
        hi = sample_df[sample_df["barcode"].isin(highlight_barcodes)]
        if not hi.empty:
            ax.scatter(
                hi["array_col"],
                -hi["array_row"],
                facecolors="none",
                edgecolors="#FFFFFF",
                s=34,
                marker="s",
                linewidths=0.7,
                zorder=4,
            )
    if focus_window is not None:
        xmin, xmax, ymin, ymax = focus_window
        ax.add_patch(
            Rectangle(
                (xmin, -ymax),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="#FFFFFF",
                linewidth=1.6,
                linestyle="-",
                zorder=5,
            )
        )
        ax.add_patch(
            Rectangle(
                (xmin, -ymax),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=COLORS["dark"],
                linewidth=0.8,
                linestyle="--",
                zorder=5,
            )
        )
    ax.set_title(title, fontsize=11.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_colorbar:
        cax = inset_axes(ax, width="4.5%", height="54%", loc="lower left", bbox_to_anchor=(1.02, 0.17, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(sc, cax=cax)
        cb.outline.set_edgecolor("#D6D8DB")
        cb.outline.set_linewidth(0.8)


def plot_spatial_zoom(
    ax: plt.Axes,
    sample_df: pd.DataFrame,
    metric: str,
    cmap: str,
    title: str,
    focus_window: tuple[float, float, float, float],
    highlight_barcodes: set[str] | None = None,
    note: str | None = None,
) -> None:
    xmin, xmax, ymin, ymax = focus_window
    zoom = sample_df[
        (sample_df["array_col"] >= xmin)
        & (sample_df["array_col"] <= xmax)
        & (sample_df["array_row"] >= ymin)
        & (sample_df["array_row"] <= ymax)
    ].copy()
    plot_spatial_map(ax, zoom, metric, cmap, title, focus_window=None, show_colorbar=False, highlight_barcodes=highlight_barcodes)
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(-(ymax + 1), -(ymin - 1))
    if note:
        add_stat_box(ax, note, loc="upper left")


def plot_corr_distribution(ax: plt.Axes, primary_corr: pd.DataFrame, tls_corr: pd.DataFrame) -> None:
    subsets = []
    subset_specs = [
        ("CAF activation", primary_corr, "caf_activation_score", COLORS["caf"]),
        ("CAF input", primary_corr, "caf_myeloid_input_score", COLORS["caf_input"]),
        ("TLS B/FDC core", tls_corr, "tls_bcell_fdc_score", COLORS["tls"]),
    ]
    for label, df, metric_y, color in subset_specs:
        sub = df[(df["metric_x"] == "suppressive_program_score") & (df["metric_y"] == metric_y)].copy()
        sub["feature"] = label
        sub["color"] = color
        subsets.append(sub)
    plot_df = pd.concat(subsets, ignore_index=True)
    feature_order = ["CAF activation", "CAF input", "TLS B/FDC core"]
    ymap = {feat: idx for idx, feat in enumerate(feature_order[::-1])}
    ax.axvline(0, color="#9BA5AE", lw=1, ls="--")
    for feature in feature_order:
        sub = plot_df[plot_df["feature"] == feature].copy()
        color = sub["color"].iloc[0]
        yi = ymap[feature]
        q1, med, q3 = np.quantile(sub["rho"], [0.25, 0.5, 0.75])
        ax.hlines(yi, q1, q3, color=color, linewidth=7, alpha=0.35, zorder=1)
        ax.hlines(yi, sub["rho"].min(), sub["rho"].max(), color="#D6DCE3", linewidth=1.4, zorder=0)
        ax.scatter(sub["rho"], yi + np.linspace(-0.22, 0.22, len(sub)), s=25, c="#FFFFFF", edgecolor=color, linewidth=0.9, zorder=3)
        ax.scatter(med, yi, s=95, c=color, edgecolor="white", linewidth=1.2, zorder=4, marker="D")
        ax.text(sub["rho"].max() + 0.03, yi, f"median={med:.2f}", fontsize=8.1, va="center", color=color)
    ax.set_yticks([ymap[f] for f in feature_order])
    ax.set_yticklabels(feature_order, fontsize=9.2)
    ax.set_xlabel("Per-sample Spearman rho")
    ax.set_ylabel("")
    ax.set_xlim(-0.42, 0.62)
    add_section_tag(ax, "Primary spatial consistency", COLORS["caf"])
    soften_axes(ax)
    add_stat_box(ax, "E03: CAF-rich\nE04: TLS-core inverse", loc="upper left")


def plot_neighbor_effects(ax: plt.Axes, niche_summary: pd.DataFrame, tls_summary: pd.DataFrame) -> None:
    combined = pd.concat(
        [
            niche_summary.assign(source="Primary spatial"),
            tls_summary.assign(source="TLS exclusion"),
        ],
        ignore_index=True,
    )
    label_map = {
        "high_caf_activation": "CAF-high neighbors",
        "high_caf_input": "CAF input neighbors",
        "high_tex": "Tex-high neighbors",
        "high_tcell": "T-cell neighbors",
        "high_tls_bcell_fdc": "TLS B/FDC neighbors",
        "high_tls_chemokine": "TLS chemokine neighbors",
    }
    combined["label"] = combined["target_flag"].map(label_map)
    combined = combined.sort_values("median_delta")
    y = np.arange(len(combined))
    colors = [COLORS["tls"] if "TLS" in x else COLORS["caf"] if "CAF" in x else COLORS["grey"] for x in combined["label"]]
    ax.axvline(0, color="#9BA5AE", lw=1, ls="--")
    ax.hlines(y, 0, combined["median_delta"], color="#CDD3DA", linewidth=2.2)
    ax.scatter(combined["median_delta"], y, s=130, c=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for yi, row in zip(y, combined.itertuples()):
        ax.text(row.median_delta, yi + 0.12, format_pvalue(row.pvalue), fontsize=7.5, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(combined["label"], fontsize=8.5)
    ax.set_xlabel("Median neighbor delta (high vs low suppressive spots)")
    add_section_tag(ax, "Neighborhood shift", COLORS["tls"])
    soften_axes(ax)


def build_figure_2() -> None:
    sample_id = select_representative_spatial_sample()
    niche_spots = load_csv("results/spatial_niche/GSE270678_spatial_niche_spot_scores.csv", low_memory=False)
    tls_spots = load_csv("results/spatial_tls_exclusion/GSE270678_tls_exclusion_spot_scores.csv", low_memory=False)

    sample_niche = niche_spots[niche_spots["sample_id"] == sample_id].copy()
    sample_tls = tls_spots[tls_spots["sample_id"] == sample_id].copy()

    focus_window = compute_focus_window(sample_tls, "suppressive_program_score")
    highlight_barcodes = set(sample_tls.loc[sample_tls["suppressive_program_score"] >= sample_tls["suppressive_program_score"].quantile(0.94), "barcode"])

    fig = plt.figure(figsize=(16.2, 8.6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.78], hspace=0.10, wspace=0.16)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    plot_spatial_map(ax_a, sample_niche, "suppressive_program_score", "magma", "Suppressive program", focus_window=focus_window, highlight_barcodes=None)
    plot_spatial_map(ax_b, sample_niche, "caf_activation_score", "rocket", "CAF activation", focus_window=focus_window, highlight_barcodes=None)
    plot_spatial_map(ax_c, sample_tls, "tls_bcell_fdc_score", "crest", "TLS B/FDC core", focus_window=focus_window, highlight_barcodes=None)
    plot_spatial_zoom(ax_d, sample_niche, "suppressive_program_score", "magma", "Hotspot window", focus_window, highlight_barcodes, "High suppressive spots\noutlined in white")
    plot_spatial_zoom(ax_e, sample_niche, "caf_activation_score", "rocket", "CAF-rich neighborhood", focus_window, highlight_barcodes, "Same window shows\nCAF alignment")
    plot_spatial_zoom(ax_f, sample_tls, "tls_bcell_fdc_score", "crest", "TLS-poor context", focus_window, highlight_barcodes, "TLS core remains low\ninside the hotspot")

    ax_a.text(0.02, 1.04, f"Representative section: {sample_id}", transform=ax_a.transAxes, fontsize=9.2, color="#5B6672")
    ax_f.text(
        0.02,
        -0.20,
        "Selected slice ranks high for suppressive burden and CAF activity but low for TLS-core signal.\nMatched zooms emphasize CAF-rich suppressive hotspots embedded in TLS-poor neighborhoods.",
        transform=ax_f.transAxes,
        fontsize=8.9,
        color="#5B6672",
        va="top",
    )

    for label, ax in zip(list("ABCDEF"), [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]):
        add_panel_label(ax, label)

    save_figure(
        fig,
        "Figure4_primary_spatial_maps",
        "Primary spatial maps reveal a CAF-rich and TLS-core-excluded suppressive niche",
        [
            "results/spatial_niche/GSE270678_spatial_niche_spot_scores.csv",
            "results/spatial_tls_exclusion/GSE270678_tls_exclusion_spot_scores.csv",
        ],
        MAIN_DIR,
    )


def build_figure_2b() -> None:
    primary_corr = load_csv("results/spatial_niche/GSE270678_spatial_niche_sample_correlations.csv")
    tls_corr = load_csv("results/spatial_tls_exclusion/GSE270678_tls_exclusion_sample_correlations.csv")
    niche_summary = load_csv("results/spatial_niche/GSE270678_spatial_niche_neighbor_summary.csv")
    tls_summary = load_csv("results/spatial_tls_exclusion/GSE270678_tls_exclusion_neighbor_summary.csv")

    fig = plt.figure(figsize=(13.8, 5.8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.05], wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_corr_distribution(ax_a, primary_corr, tls_corr)
    plot_neighbor_effects(ax_b, niche_summary, tls_summary)

    for label, ax in zip(list("AB"), [ax_a, ax_b]):
        add_panel_label(ax, label)

    save_figure(
        fig,
        "Figure5_primary_spatial_quantification",
        "Primary spatial quantification supports CAF enrichment and TLS-core exclusion",
        [
            "results/spatial_niche/GSE270678_spatial_niche_sample_correlations.csv",
            "results/spatial_niche/GSE270678_spatial_niche_neighbor_summary.csv",
            "results/spatial_tls_exclusion/GSE270678_tls_exclusion_sample_correlations.csv",
            "results/spatial_tls_exclusion/GSE270678_tls_exclusion_neighbor_summary.csv",
        ],
        MAIN_DIR,
    )


def plot_external_spatial_validation(ax: plt.Axes, corr_df: pd.DataFrame) -> None:
    metric_map = {
        "caf_activation_score": ("CAF activation", COLORS["caf"]),
        "caf_myeloid_input_score": ("CAF input", COLORS["caf_input"]),
    }
    sub = corr_df[(corr_df["metric_x"] == "suppressive_program_score") & (corr_df["metric_y"].isin(metric_map))].copy()
    feature_order = ["CAF activation", "CAF input"]
    ymap = {feat: idx for idx, feat in enumerate(feature_order[::-1])}
    for metric_y, (label, color) in metric_map.items():
        cur = sub[sub["metric_y"] == metric_y].copy().sort_values("rho")
        yi = ymap[label]
        q1, med, q3 = np.quantile(cur["rho"], [0.25, 0.5, 0.75])
        ax.hlines(yi, cur["rho"].min(), cur["rho"].max(), color="#D6DCE3", linewidth=1.5, zorder=0)
        ax.hlines(yi, q1, q3, color=color, linewidth=7, alpha=0.32, zorder=1)
        point_colors = np.where(cur["tissue_group"].eq("peritoneal_met"), COLORS["peritoneum"], COLORS["stomach"])
        ax.scatter(cur["rho"], yi + np.linspace(-0.16, 0.16, len(cur)), s=38, c=point_colors, edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(med, yi, s=92, c=color, edgecolor="white", linewidth=1.2, marker="D", zorder=4)
        ax.text(cur["rho"].max() + 0.02, yi, f"median={med:.2f}", fontsize=8.1, va="center", color=color)
    ax.axvline(0, color="#9BA5AE", lw=1, ls="--")
    ax.set_yticks([ymap[f] for f in feature_order])
    ax.set_yticklabels(feature_order, fontsize=9.2)
    ax.set_xlabel("Sample-level rho in GSE251950")
    ax.set_ylabel("")
    ax.set_xlim(-0.05, 0.72)
    add_section_tag(ax, "External ST replication", COLORS["peritoneum"])
    add_stat_box(ax, "CAF-rich coupling repeats\nin an independent ST cohort", loc="upper left")
    soften_axes(ax)


def plot_cosmx_scatter(ax: plt.Axes, sample_summary: pd.DataFrame) -> None:
    x = sample_summary["fibroblast_caf_activation_mean"]
    y = sample_summary["macrophage_suppressive_mean"]
    size = 30 + sample_summary["fibroblast_fraction"].fillna(0).to_numpy() * 110
    sc = ax.scatter(
        x,
        y,
        s=size,
        c=sample_summary["bcell_fraction"],
        cmap="YlGnBu",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, color=COLORS["caf"], lw=2.1)
    ax.axvline(np.median(x), color="#CFD5DB", lw=1.0, ls="--")
    ax.axhline(np.median(y), color="#CFD5DB", lw=1.0, ls="--")
    ax.set_xlabel("Fibroblast CAF activation mean")
    ax.set_ylabel("Macrophage suppressive mean")
    cax = inset_axes(ax, width="4%", height="42%", loc="lower right", borderpad=1.1)
    cb = plt.colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("B-cell fraction", fontsize=7.5)
    add_section_tag(ax, "CosMx sample ecology", COLORS["accent"])
    add_stat_box(ax, "E05c\nrho=0.427\nP=2.05e-04")
    soften_axes(ax)


def plot_cosmx_neighbor(ax: plt.Axes, neighbor_summary: pd.DataFrame) -> None:
    display = neighbor_summary[neighbor_summary["target_flag"].isin(["is_high_caf_fibro", "is_bcell", "is_fibroblast", "is_high_tls_bcell"])].copy()
    display["label"] = display["target_flag"].map(
        {
            "is_high_caf_fibro": "High-CAF fibroblast neighbors",
            "is_bcell": "B-cell neighbors",
            "is_fibroblast": "All fibroblast neighbors",
            "is_high_tls_bcell": "High-TLS B-cell neighbors",
        }
    )
    display = display.sort_values("median_delta")
    y = np.arange(len(display))
    colors = [COLORS["tls"] if "B-cell" in x else COLORS["caf"] for x in display["label"]]
    ax.axvline(0, color="#9BA5AE", lw=1, ls="--")
    ax.hlines(y, 0, display["median_delta"], color="#D0D5DB", linewidth=2)
    ax.scatter(display["median_delta"], y, s=120, c=colors, edgecolor="white", linewidth=1.2, zorder=3)
    for yi, row in zip(y, display.itertuples()):
        ax.text(row.median_delta, yi + 0.12, format_pvalue(row.pvalue), fontsize=7.5, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(display["label"], fontsize=8.5)
    ax.set_xlabel("Median macrophage-neighbor delta")
    add_section_tag(ax, "CosMx neighborhood shift", COLORS["tls"])
    soften_axes(ax)


def plot_receptor_switch_by_state(ax: plt.Axes, mech_df: pd.DataFrame) -> None:
    order = ["early_like", "intermediate_like", "suppressive_like"]
    palette = {"early_like": COLORS["early"], "intermediate_like": COLORS["intermediate"], "suppressive_like": COLORS["late"]}
    sns.violinplot(
        data=mech_df,
        x="mechanism_state",
        y="receptor_switch_score",
        order=order,
        hue="mechanism_state",
        palette=palette,
        legend=False,
        cut=0,
        inner=None,
        linewidth=0.9,
        ax=ax,
    )
    sns.boxplot(
        data=mech_df,
        x="mechanism_state",
        y="receptor_switch_score",
        order=order,
        width=0.18,
        showcaps=False,
        showfliers=False,
        boxprops={"facecolor": "#FFFFFF", "edgecolor": "#C9CED4"},
        whiskerprops={"color": "#C9CED4"},
        medianprops={"color": COLORS["dark"], "lw": 1.4},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Receptor-switch score")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Early-like", "Intermediate-like", "Suppressive-like"])
    add_stat_box(ax, "Cell-level display\nmedian 1.020 vs 0.416\ndescriptive only")
    soften_axes(ax)


def plot_peritoneal_dumbbell(ax: plt.Axes, frac_df: pd.DataFrame) -> None:
    pivot = frac_df[frac_df["mechanism_state"].isin(["early_like", "suppressive_like"])].pivot_table(
        index=["sample_id", "tissue_group"], columns="mechanism_state", values="fraction_within_sample"
    ).reset_index()
    pivot = pivot.sort_values(["tissue_group", "suppressive_like"])
    palette = {"stomach_tumor": COLORS["stomach"], "peritoneum_tumor": COLORS["peritoneum"]}
    y = np.arange(len(pivot))
    for yi, row in zip(y, pivot.itertuples()):
        color = palette.get(row.tissue_group, COLORS["grey"])
        ax.plot([row.early_like, row.suppressive_like], [yi, yi], color=mcolors.to_rgba(color, 0.65), lw=2)
        ax.scatter(row.early_like, yi, s=42, c=COLORS["early"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(row.suppressive_like, yi, s=42, c=COLORS["late"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks([])
    ax.set_xlabel("Within-sample state fraction")
    add_stat_box(ax, "Sample-level fractions\nall P >= 0.130\nterminal/early P=0.610", loc="upper right")
    soften_axes(ax)


def plot_peritoneal_index(ax: plt.Axes, sample_means: pd.DataFrame) -> None:
    order = ["stomach_tumor", "peritoneum_tumor"]
    palette = {"stomach_tumor": COLORS["stomach"], "peritoneum_tumor": COLORS["peritoneum"]}
    sns.violinplot(data=sample_means, x="tissue_group", y="terminal_receptor_switch_index", order=order, hue="tissue_group", palette=palette, legend=False, cut=0, inner=None, ax=ax)
    sns.stripplot(data=sample_means, x="tissue_group", y="terminal_receptor_switch_index", order=order, hue="tissue_group", palette=palette, dodge=False, legend=False, size=6, edgecolor="white", linewidth=0.9, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Terminal receptor-switch index")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Stomach tumor", "Peritoneum tumor"])
    add_stat_box(ax, "Sample-level index\nmedian 0.926 vs -0.139\nP=0.0454; n=3 vs 26")
    soften_axes(ax)


def build_figure_3() -> None:
    gse251950_corr = load_csv("results/external_spatial_gse251950/GSE251950_sample_correlations.csv")
    cosmx_summary = load_csv("results/external_spatial_gse308624/GSE308624_sample_summary.csv")
    cosmx_neighbor = load_csv("results/external_spatial_gse308624/GSE308624_macrophage_neighbor_summary.csv")

    fig = plt.figure(figsize=(16.2, 5.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.05], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_external_spatial_validation(ax_a, gse251950_corr)
    plot_cosmx_scatter(ax_b, cosmx_summary)
    plot_cosmx_neighbor(ax_c, cosmx_neighbor)

    for label, ax in zip(list("ABC"), [ax_a, ax_b, ax_c]):
        add_panel_label(ax, label)

    save_figure(
        fig,
        "Figure6_external_spatial_validation",
        "External spatial validation reinforces CAF-rich suppressive niches",
        [
            "results/external_spatial_gse251950/GSE251950_sample_correlations.csv",
            "results/external_spatial_gse308624/GSE308624_sample_summary.csv",
            "results/external_spatial_gse308624/GSE308624_macrophage_neighbor_summary.csv",
        ],
        MAIN_DIR,
    )


def build_figure_3b() -> None:
    mech_umap = load_csv("results/myeloid_external_mechanism/GSE183904_mechanism_umap_scores.csv")
    peritoneal_frac = load_csv("results/peritoneal_state_bias/GSE183904_peritoneal_sample_state_fractions.csv")
    peritoneal_means = load_csv("results/peritoneal_state_bias/GSE183904_peritoneal_sample_means.csv")

    fig = plt.figure(figsize=(16.2, 5.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_receptor_switch_by_state(ax_a, mech_umap)
    plot_peritoneal_dumbbell(ax_b, peritoneal_frac)
    plot_peritoneal_index(ax_c, peritoneal_means)

    for label, ax in zip(list("ABC"), [ax_a, ax_b, ax_c]):
        add_panel_label(ax, label)

    save_figure(
        fig,
        "Figure7_peritoneal_terminal_ecology",
        "Terminal receptor-switched peritoneal myeloid ecology",
        [
            "results/myeloid_external_mechanism/GSE183904_mechanism_umap_scores.csv",
            "results/peritoneal_state_bias/GSE183904_peritoneal_sample_state_fractions.csv",
            "results/peritoneal_state_bias/GSE183904_peritoneal_sample_means.csv",
        ],
        MAIN_DIR,
    )


def plot_signature_benchmark(ax: plt.Axes, surv_df: pd.DataFrame) -> None:
    surv_df = surv_df.copy()
    signature_order = ["our_suppressive_axis", "trajectory_polarized_axis", "trem2_tam", "spp1_tam", "m2_macrophage", "mdsc_like"]
    label_map = {
        "our_suppressive_axis": "Our axis",
        "trajectory_polarized_axis": "Trajectory axis",
        "trem2_tam": "TREM2 TAM",
        "spp1_tam": "SPP1 TAM",
        "m2_macrophage": "M2-like macrophage",
        "mdsc_like": "MDSC-like",
    }
    dataset_meta = {
        "TCGA_STAD": {"label": "TCGA", "color": COLORS["late"], "marker": "o"},
        "GSE84437": {"label": "84437", "color": COLORS["tls"], "marker": "s"},
    }
    y = np.arange(len(signature_order))[::-1]
    pos_map = {sig: yi for sig, yi in zip(signature_order, y)}
    ax.axvline(1.0, color="#9BA5AE", lw=1.1, ls="--")
    ax.axvspan(1.0, 1.32, color=mcolors.to_rgba(COLORS["late"], 0.06))
    ax.axvspan(0.72, 1.0, color=mcolors.to_rgba(COLORS["early"], 0.05))
    ax.axhspan(pos_map["our_suppressive_axis"] - 0.5, pos_map["our_suppressive_axis"] + 0.5, color=mcolors.to_rgba(COLORS["late"], 0.08), zorder=0)
    for sig in signature_order:
        sub = surv_df[surv_df["signature_name"] == sig].copy()
        if sub.empty:
            continue
        yi = pos_map[sig]
        vals = []
        for ds in ["TCGA_STAD", "GSE84437"]:
            row = sub[sub["dataset_id"] == ds]
            if row.empty:
                continue
            hr = float(row["adjusted_hr"].iloc[0])
            vals.append((ds, hr, float(row["adjusted_p"].iloc[0])))
        if len(vals) == 2:
            ax.plot([vals[0][1], vals[1][1]], [yi, yi], color="#D0D6DD", lw=2.3, zorder=1)
        for ds, hr, pval in vals:
            meta = dataset_meta[ds]
            ax.scatter(hr, yi, s=115, c=meta["color"], marker=meta["marker"], edgecolor="white", linewidth=1.2, zorder=3)
            if pval < 0.05:
                ax.text(hr + 0.015, yi + 0.10, "*", fontsize=12, fontweight="bold", color=meta["color"])
    ax.set_xlim(0.72, 1.32)
    ax.set_yticks(y)
    ax.set_yticklabels([label_map[sig] for sig in signature_order], fontsize=8.6)
    ax.set_xlabel("Adjusted hazard ratio")
    add_section_tag(ax, "Signature benchmark", COLORS["late"])
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker=dataset_meta[k]["marker"], color="w", markerfacecolor=dataset_meta[k]["color"], markeredgecolor="white", markersize=8, label=dataset_meta[k]["label"])
        for k in ["TCGA_STAD", "GSE84437"]
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8.1, loc="lower right")
    soften_axes(ax)


def plot_dual_axis_forest(ax: plt.Axes, cox_df: pd.DataFrame) -> None:
    plot_df = cox_df.copy()
    plot_df["label"] = plot_df["dataset_id"].map({"TCGA_STAD": "TCGA", "GSE84437": "84437"}) + " | " + plot_df["term"].map(
        {"our_axis_std": "Suppressive", "trajectory_axis_std": "Trajectory"}
    )
    plot_df = plot_df.sort_values("hazard_ratio")
    y = np.arange(len(plot_df))
    color_map = {"our_axis_std": COLORS["late"], "trajectory_axis_std": COLORS["early"]}
    ax.axvline(1.0, color="#9BA5AE", lw=1, ls="--")
    for yi, row in zip(y, plot_df.itertuples()):
        ax.plot([row.ci95_lower, row.ci95_upper], [yi, yi], color="#AEB6BF", lw=2.2)
        ax.scatter(row.hazard_ratio, yi, s=120, c=color_map[row.term], edgecolor="white", linewidth=1.1, zorder=3)
        ax.text(row.ci95_upper + 0.03, yi, format_pvalue(row.p_value), va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8.5)
    ax.set_xlabel("Hazard ratio")
    add_section_tag(ax, "Dual-axis Cox", COLORS["late"])
    soften_axes(ax)


def plot_bulk_heatmap(ax: plt.Axes, group_df: pd.DataFrame) -> None:
    plot_df = group_df.copy()
    row_suffix = {
        "burden_high__polar_low": "HS/LAI",
        "burden_high__polar_high": "HS/HAI",
        "burden_low__polar_high": "LS/HAI",
        "burden_low__polar_low": "LS/LAI",
    }
    plot_df["row"] = (
        plot_df["dataset_id"].map({"TCGA_STAD": "TCGA", "GSE84437": "84437"})
        + " | "
        + plot_df["dual_axis_group"].map(row_suffix)
        + " (n="
        + plot_df["n_samples"].astype(int).astype(str)
        + ")"
    )
    mat = plot_df.set_index("row")[
        ["caf_input_score_mean", "tgfb_stroma_score_mean", "teff_score_mean", "ifng_response_score_mean", "tls_b_score_mean", "stromal_adaptive_imbalance_mean"]
    ]
    mat.columns = ["CAF input", "TGFb stroma", "Teff", "IFNg", "TLS-B", "Stromal-adaptive\nimbalance"]
    mat = zscore_frame(mat)
    sns.heatmap(
        mat,
        cmap=sns.diverging_palette(210, 25, as_cmap=True),
        center=0,
        linewidths=0.6,
        linecolor="#F2EFE9",
        ax=ax,
        cbar_kws={"shrink": 0.7, "label": "Within-metric z-score"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8.4)
    ax.tick_params(axis="x", labelsize=8.6)
    ax.hlines([4], *ax.get_xlim(), colors="#FFFFFF", linewidth=2.0)
    add_section_tag(ax, "Ecology context", COLORS["peritoneum"])


def build_figure_4() -> None:
    survival_benchmark = load_csv("results/signature_benchmark/bulk_signature_benchmark_survival.csv")
    dual_axis_cox = load_csv("results/signature_benchmark/dual_axis_bulk_cox.csv")
    ecology_group = load_csv("results/signature_benchmark/bulk_ecology_group_summary.csv")

    fig = plt.figure(figsize=(15.6, 9.8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.90, 1.18], width_ratios=[1.05, 0.95], hspace=0.28, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_signature_benchmark(ax_a, survival_benchmark)
    plot_dual_axis_forest(ax_b, dual_axis_cox)
    plot_bulk_heatmap(ax_c, ecology_group)
    add_stat_box(ax_c, "Highest imbalance:\nhigh-suppressive /\nlow-adaptive-immune", loc="upper right")

    for label, ax in zip(list("ABC"), [ax_a, ax_b, ax_c]):
        add_panel_label(ax, label)

    save_figure(
        fig,
        "Figure8_bulk_ecology_benchmark",
        "Bulk ecology benchmark of the suppressive program",
        [
            "results/signature_benchmark/bulk_signature_benchmark_survival.csv",
            "results/signature_benchmark/dual_axis_bulk_cox.csv",
            "results/signature_benchmark/bulk_ecology_group_summary.csv",
        ],
        MAIN_DIR,
    )


def build_figure_4b() -> None:
    ecology_scores = load_csv("results/signature_benchmark/bulk_ecology_context_scores.csv")

    fig = plt.figure(figsize=(15.5, 6.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_km(
        ax_a,
        ecology_scores,
        "TCGA_STAD",
        ["burden_high__polar_low", "burden_high__polar_high", "burden_low__polar_high", "burden_low__polar_low"],
        "TCGA-STAD",
    )
    add_section_tag(ax_a, "Dual-axis survival", COLORS["late"])
    plot_km(
        ax_b,
        ecology_scores,
        "GSE84437",
        ["burden_high__polar_low", "burden_high__polar_high", "burden_low__polar_high", "burden_low__polar_low"],
        "GSE84437",
    )
    add_section_tag(ax_b, "External survival", COLORS["tls"])
    add_stat_box(ax_a, "E09b\nimbalance P=2.82e-08", loc="upper right")
    add_stat_box(ax_b, "Exploratory group:\nhigh-suppressive /\nlow-adaptive-immune", loc="upper right")

    for label, ax in zip(list("AB"), [ax_a, ax_b]):
        add_panel_label(ax, label)

    save_figure(
        fig,
        "Figure9_bulk_survival_stratification",
        "Bulk dual-axis survival stratification",
        [
            "results/signature_benchmark/bulk_ecology_context_scores.csv",
        ],
        MAIN_DIR,
    )


def plot_working_model(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.text(0.03, 0.94, "CAF-shaped formation and ecological landing of the suppressive myeloid state", transform=ax.transAxes, fontsize=16, fontweight="bold")
    ax.text(0.03, 0.90, "An association-based model linking CAF-related signals, a transitional bridge, spatial context, and clinical transfer.", transform=ax.transAxes, fontsize=10.2, color="#596574")

    # Top mechanistic chain.
    top_nodes = [
        ("Activated CAF", "INHBA / CSF1 / IL34 / TGFB1\nformative pressure", COLORS["caf"], (0.06, 0.63)),
        ("Bridge state", "INHBA-associated\nintermediate", COLORS["intermediate"], (0.37, 0.63)),
        ("Terminal myeloid", "receptor-switched\nsuppressive program", COLORS["late"], (0.69, 0.63)),
    ]
    for title, body, color, (x, y) in top_nodes:
        ax.add_patch(Ellipse((x + 0.095, y + 0.09), 0.19, 0.18, transform=ax.transAxes, facecolor=mcolors.to_rgba(color, 0.11), edgecolor=color, linewidth=1.6))
        for dx, dy in [(-0.04, 0.00), (0.00, 0.03), (0.04, -0.02), (-0.01, -0.04)]:
            ax.add_patch(Circle((x + 0.095 + dx, y + 0.09 + dy), 0.018, transform=ax.transAxes, facecolor=mcolors.to_rgba(color, 0.85), edgecolor="white", linewidth=0.8))
        ax.text(x + 0.095, y + 0.16, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=12.2, fontweight="bold", color=color)
        ax.text(x + 0.095, y + 0.02, body, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.6, color=COLORS["dark"])

    add_arrow_path(ax, [(0.24, 0.72), (0.47, 0.72), (0.80, 0.72)], rad=0.00)

    # Spatial niche panel.
    draw_round_box(ax, (0.05, 0.18, 0.36, 0.28), edgecolor=COLORS["tls"], facecolor="#FBFCFE", linewidth=1.5, rounding=0.025)
    ax.text(0.07, 0.42, "Spatial niche", transform=ax.transAxes, fontsize=12.6, fontweight="bold", color=COLORS["tls"])
    ax.text(0.07, 0.385, "Suppressive hotspots settle in CAF-rich neighborhoods\nwhile remaining separated from TLS-like cores.", transform=ax.transAxes, fontsize=9.5, color=COLORS["dark"])
    ax.add_patch(Circle((0.19, 0.28), 0.055, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["late"], 0.20), edgecolor=COLORS["late"], linewidth=1.2))
    ax.add_patch(Circle((0.31, 0.27), 0.050, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["tls"], 0.16), edgecolor=COLORS["tls"], linewidth=1.2))
    for cx, cy in [(0.14, 0.33), (0.13, 0.24), (0.22, 0.35), (0.25, 0.22)]:
        ax.add_patch(Circle((cx, cy), 0.015, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["caf"], 0.92), edgecolor="white", linewidth=0.6))
    ax.text(0.19, 0.28, "Suppressive\nhotspot", transform=ax.transAxes, ha="center", va="center", fontsize=8.9, color=COLORS["late"], fontweight="bold")
    ax.text(0.31, 0.27, "TLS core", transform=ax.transAxes, ha="center", va="center", fontsize=8.9, color=COLORS["tls"], fontweight="bold")
    ax.text(0.12, 0.205, "CAF-rich margin", transform=ax.transAxes, fontsize=8.7, color=COLORS["caf"])

    # Peritoneal ecology panel.
    draw_round_box(ax, (0.46, 0.18, 0.22, 0.28), edgecolor=COLORS["peritoneum"], facecolor="#FDFCFA", linewidth=1.5, rounding=0.025)
    ax.text(0.48, 0.42, "Peritoneal ecology", transform=ax.transAxes, fontsize=12.4, fontweight="bold", color=COLORS["peritoneum"])
    ax.text(0.48, 0.385, "Peritoneal tumors preferentially amplify\nthe terminal receptor-switch phenotype.", transform=ax.transAxes, fontsize=9.4)
    ax.add_patch(Ellipse((0.57, 0.28), 0.14, 0.12, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["peritoneum"], 0.12), edgecolor=COLORS["peritoneum"], linewidth=1.3))
    for cx, cy in [(0.54, 0.30), (0.60, 0.31), (0.58, 0.25), (0.52, 0.25)]:
        ax.add_patch(Circle((cx, cy), 0.014, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["late"], 0.95), edgecolor="white", linewidth=0.6))
    ax.text(0.57, 0.21, "terminal-state skew", transform=ax.transAxes, fontsize=8.7, ha="center", color=COLORS["late"])

    # Bulk translation panel.
    draw_round_box(ax, (0.73, 0.18, 0.22, 0.28), edgecolor=COLORS["accent"], facecolor="#FBFCFE", linewidth=1.5, rounding=0.025)
    ax.text(0.75, 0.42, "Bulk translation", transform=ax.transAxes, fontsize=12.4, fontweight="bold", color=COLORS["accent"])
    ax.text(0.75, 0.385, "The clinically unfavorable group is\nstromal-high and adaptive-immune-low.", transform=ax.transAxes, fontsize=9.4)
    for x, h, color in [(0.78, 0.12, COLORS["caf"]), (0.83, 0.10, COLORS["peritoneum"]), (0.88, 0.05, COLORS["tls"])]:
        ax.add_patch(Rectangle((x, 0.23), 0.028, h, transform=ax.transAxes, facecolor=mcolors.to_rgba(color, 0.82), edgecolor="white", linewidth=0.7))
    ax.text(0.84, 0.21, "stromal-high / adaptive-low", transform=ax.transAxes, fontsize=8.7, ha="center", color=COLORS["dark"])

    # Output arrows.
    for start, end in [((0.80, 0.63), (0.23, 0.44)), ((0.82, 0.63), (0.58, 0.44)), ((0.84, 0.63), (0.84, 0.44))]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.8,
                color="#7B8794",
            )
        )
def build_figure_5() -> None:
    fig, ax = plt.subplots(figsize=(16.8, 8.6))
    plot_working_model(ax)
    add_panel_label(ax, "A")
    save_figure(
        fig,
        "Figure10_working_model",
        "Working model of CAF-shaped myeloid suppressive ecology in gastric cancer",
        ["results/tables/manuscript_evidence_summary.csv"],
        MAIN_DIR,
    )


def build_supplementary_figure_1() -> None:
    cluster_program = load_csv("results/myeloid_validation/GSE163558_cluster_tissue_program.csv")
    tissue_burden = load_csv("results/myeloid_external_atlas/GSE183904_myeloid_tissue_cluster_burden.csv")
    receptor_switch = load_csv("results/cell_crosstalk/GSE206785_myeloid_receptor_switch.csv")
    regulator = load_csv("results/mechanism_trajectory/GSE206785_curated_regulator_dynamics.csv")

    fig = plt.figure(figsize=(15.8, 9.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    heat_a = cluster_program.pivot_table(index="leiden", columns="tissue_group", values="mean")
    sns.heatmap(zscore_frame(heat_a.fillna(0)), cmap=sns.diverging_palette(215, 25, as_cmap=True), center=0, linewidths=0.5, linecolor="#F2EFE9", ax=ax_a)
    add_section_tag(ax_a, "GSE163558 cluster map", COLORS["accent"])
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Leiden cluster")

    burden_plot = tissue_burden.set_index("tissue_group")[["early_cluster_fraction", "suppressive_cluster_fraction", "median_axis_score"]]
    sns.heatmap(zscore_frame(burden_plot), cmap=sns.diverging_palette(215, 25, as_cmap=True), center=0, linewidths=0.5, linecolor="#F2EFE9", ax=ax_b)
    add_section_tag(ax_b, "GSE183904 tissue burden", COLORS["peritoneum"])
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")

    receptor_plot = receptor_switch.set_index("gene")[["myeloid_early_tumor", "myeloid_transition_inhba_tumor", "myeloid_suppressive_tumor"]].head(10)
    sns.heatmap(zscore_frame(receptor_plot), cmap=sns.color_palette("blend:#FBFAF7,#426F9B,#B24A39", as_cmap=True), linewidths=0.5, linecolor="#F2EFE9", ax=ax_c)
    add_section_tag(ax_c, "Receptor rewiring", COLORS["late"])
    ax_c.set_xlabel("")
    ax_c.set_ylabel("")

    reg_plot = regulator.sort_values("rho_with_axis_score", ascending=False).head(12).sort_values("rho_with_axis_score")
    ax_d.hlines(np.arange(len(reg_plot)), 0, reg_plot["rho_with_axis_score"], color="#CBD2D9", linewidth=2)
    ax_d.scatter(reg_plot["rho_with_axis_score"], np.arange(len(reg_plot)), s=110, c=reg_plot["late_minus_early"], cmap="RdYlBu_r", edgecolor="white", linewidth=1.1)
    ax_d.set_yticks(np.arange(len(reg_plot)))
    ax_d.set_yticklabels(reg_plot["gene"], fontsize=8.5)
    ax_d.set_xlabel("rho with trajectory axis")
    add_section_tag(ax_d, "Late regulators", COLORS["intermediate"])
    soften_axes(ax_d)

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle("Supplementary Figure 1. External validation and receptor-switch details", fontsize=15.5, y=0.985)
    save_figure(
        fig,
        "SupplementaryFigure1_external_validation_details",
        "External validation and receptor-switch details",
        [
            "results/myeloid_validation/GSE163558_cluster_tissue_program.csv",
            "results/myeloid_external_atlas/GSE183904_myeloid_tissue_cluster_burden.csv",
            "results/cell_crosstalk/GSE206785_myeloid_receptor_switch.csv",
            "results/mechanism_trajectory/GSE206785_curated_regulator_dynamics.csv",
        ],
        SUPP_DIR,
    )


def plot_gse251950_metric_summary(ax: plt.Axes, corr_df: pd.DataFrame) -> None:
    metric_map = {
        "caf_activation_score": ("CAF activation", COLORS["caf"]),
        "caf_myeloid_input_score": ("CAF input", COLORS["caf_input"]),
        "tls_chemokine_score": ("TLS chemokines", COLORS["bcell"]),
        "tls_bcell_fdc_score": ("TLS B/FDC core", COLORS["tls"]),
    }
    sub = corr_df[corr_df["metric_x"] == "suppressive_program_score"].copy()
    feature_order = ["CAF activation", "CAF input", "TLS chemokines", "TLS B/FDC core"]
    ymap = {feat: idx for idx, feat in enumerate(feature_order[::-1])}
    ax.axvline(0, color="#9BA5AE", lw=1.0, ls="--")
    for metric_y, (label, color) in metric_map.items():
        cur = sub[sub["metric_y"] == metric_y].copy().sort_values("rho")
        yi = ymap[label]
        ax.hlines(yi, cur["rho"].min(), cur["rho"].max(), color="#D6DCE3", linewidth=1.4, zorder=0)
        q1, med, q3 = np.quantile(cur["rho"], [0.25, 0.5, 0.75])
        ax.hlines(yi, q1, q3, color=color, linewidth=7, alpha=0.30, zorder=1)
        ax.scatter(cur["rho"], yi + np.linspace(-0.17, 0.17, len(cur)), s=28, c="#FFFFFF", edgecolor=color, linewidth=0.9, zorder=3)
        ax.scatter(med, yi, s=88, c=color, edgecolor="white", linewidth=1.1, marker="D", zorder=4)
    ax.set_yticks([ymap[f] for f in feature_order])
    ax.set_yticklabels(feature_order, fontsize=8.8)
    ax.set_xlabel("Sample-level rho in GSE251950")
    ax.set_ylabel("")
    add_section_tag(ax, "Independent ST metrics", COLORS["peritoneum"])
    soften_axes(ax)


def plot_neighbor_summary_full(ax: plt.Axes, neighbor_summary: pd.DataFrame) -> None:
    display = neighbor_summary.copy()
    display["label"] = display["target_flag"].map(
        {
            "is_bcell": "B-cell neighbors",
            "is_fibroblast": "All fibroblast neighbors",
            "is_high_caf_fibro": "High-CAF fibroblasts",
            "is_high_tls_bcell": "High-TLS B cells",
        }
    )
    display = display.sort_values("median_delta")
    y = np.arange(len(display))
    colors = [COLORS["tls"] if "B" in str(x) else COLORS["caf"] for x in display["label"]]
    ax.axvline(0, color="#9BA5AE", lw=1, ls="--")
    ax.hlines(y, 0, display["median_delta"], color="#D0D5DB", linewidth=2.1)
    ax.scatter(display["median_delta"], y, s=116, c=colors, edgecolor="white", linewidth=1.0, zorder=3)
    for yi, row in zip(y, display.itertuples()):
        ax.text(row.median_delta, yi + 0.11, format_pvalue(row.pvalue), fontsize=7.2, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(display["label"], fontsize=8.6)
    ax.set_xlabel("Median macrophage-neighbor delta")
    add_section_tag(ax, "CosMx neighbors", COLORS["accent"])
    soften_axes(ax)


def plot_state_fraction_box(ax: plt.Axes, frac_df: pd.DataFrame) -> None:
    plot_df = frac_df.copy()
    state_map = {
        "early_like": "Early",
        "intermediate_like": "Bridge",
        "suppressive_like": "Late",
    }
    plot_df = plot_df[plot_df["mechanism_state"].isin(state_map)].copy()
    plot_df["State"] = plot_df["mechanism_state"].map(state_map)
    plot_df["Tissue"] = plot_df["tissue_group"].map({"stomach_tumor": "Stomach", "peritoneum_tumor": "Peritoneum"})
    palette = {"Stomach": COLORS["stomach"], "Peritoneum": COLORS["peritoneum"]}
    sns.boxplot(
        data=plot_df,
        x="State",
        y="fraction_within_sample",
        hue="Tissue",
        palette=palette,
        showcaps=False,
        fliersize=0,
        width=0.62,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="State",
        y="fraction_within_sample",
        hue="Tissue",
        palette=palette,
        dodge=True,
        size=3.8,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, fontsize=8.1, loc="upper left")
    ax.set_xlabel("")
    ax.set_ylabel("Within-sample fraction")
    add_section_tag(ax, "Peritoneal state fractions", COLORS["peritoneum"])
    soften_axes(ax)


def plot_peritoneal_feature_compare(ax: plt.Axes, sample_means: pd.DataFrame) -> None:
    plot_df = sample_means[["sample_id", "tissue_group", "terminal_receptor_switch_index", "suppressive_program_score"]].copy()
    melt = plot_df.melt(id_vars=["sample_id", "tissue_group"], var_name="feature", value_name="value")
    feature_map = {
        "terminal_receptor_switch_index": "Terminal switch index",
        "suppressive_program_score": "Suppressive burden",
    }
    melt["Feature"] = melt["feature"].map(feature_map)
    melt["Tissue"] = melt["tissue_group"].map({"stomach_tumor": "Stomach", "peritoneum_tumor": "Peritoneum"})
    palette = {"Stomach": COLORS["stomach"], "Peritoneum": COLORS["peritoneum"]}
    sns.boxplot(data=melt, x="Feature", y="value", hue="Tissue", palette=palette, showcaps=False, fliersize=0, width=0.62, ax=ax)
    sns.stripplot(data=melt, x="Feature", y="value", hue="Tissue", palette=palette, dodge=True, size=4, edgecolor="white", linewidth=0.6, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, fontsize=8.1, loc="upper left")
    ax.set_xlabel("")
    ax.set_ylabel("Sample-level score")
    add_section_tag(ax, "Switch vs burden", COLORS["late"])
    soften_axes(ax)


def build_supplementary_figure_2() -> None:
    kim_scores = load_csv("results/treatment_validation_direct_kim/Kim_direct_icb_sample_scores_avg.csv")
    kim_tests = load_csv("results/treatment_validation_direct_kim/Kim_direct_icb_composite_tests_avg.csv")
    kim_rates = load_csv("results/treatment_validation_direct_kim/Kim_direct_icb_group_response_rates_avg.csv")

    fig = plt.figure(figsize=(15.2, 8.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    order = ["R", "NR"]
    palette = {"R": COLORS["early"], "NR": COLORS["late"]}
    sns.violinplot(data=kim_scores, x="response_label", y="suppressive_ecology_index", order=order, hue="response_label", palette=palette, legend=False, inner=None, cut=0, ax=ax_a)
    sns.stripplot(data=kim_scores, x="response_label", y="suppressive_ecology_index", order=order, hue="response_label", palette=palette, dodge=False, legend=False, size=6, edgecolor="white", linewidth=0.9, ax=ax_a)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Suppressive ecology index")
    add_section_tag(ax_a, "Suppressive ecology", COLORS["late"])
    top_test = kim_tests[kim_tests["metric"] == "suppressive_ecology_index"].iloc[0]
    add_stat_box(ax_a, f"AUC={top_test['response_oriented_auc']:.3f}\n{format_pvalue(top_test['pvalue'])}")
    soften_axes(ax_a)

    sns.violinplot(data=kim_scores, x="response_label", y="formation_ecology_index", order=order, hue="response_label", palette=palette, legend=False, inner=None, cut=0, ax=ax_b)
    sns.stripplot(data=kim_scores, x="response_label", y="formation_ecology_index", order=order, hue="response_label", palette=palette, dodge=False, legend=False, size=6, edgecolor="white", linewidth=0.9, ax=ax_b)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Formation ecology index")
    add_section_tag(ax_b, "Formation ecology", COLORS["caf"])
    soften_axes(ax_b)

    rate_plot = kim_rates[kim_rates["group_definition"] == "median_split"].copy()
    rate_plot["label"] = rate_plot["metric"].str.replace("_", " ").str.replace("index", "").str.title()
    y = np.arange(len(rate_plot))
    ax_c.hlines(y, rate_plot["response_rate_high"], rate_plot["response_rate_low"], color="#CFD5DB", linewidth=2.2)
    ax_c.scatter(rate_plot["response_rate_high"], y, s=130, c=COLORS["late"], edgecolor="white", linewidth=1.2, label="High-score group")
    ax_c.scatter(rate_plot["response_rate_low"], y, s=130, c=COLORS["early"], edgecolor="white", linewidth=1.2, label="Low-score group")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(rate_plot["label"], fontsize=8.5)
    ax_c.set_xlabel("Response rate")
    add_section_tag(ax_c, "Response-rate split", COLORS["early"])
    ax_c.legend(frameon=False, fontsize=8, loc="lower right")
    soften_axes(ax_c)

    ax_d.axis("off")
    ax_d.text(0.0, 0.95, "Appendix positioning", fontsize=13, fontweight="bold", transform=ax_d.transAxes)
    ax_d.text(
        0.0,
        0.72,
        "These Kim cohort plots are retained only as supportive context.\nThey should not be used as the central publication proof layer for the manuscript.",
        fontsize=10.5,
        transform=ax_d.transAxes,
        va="top",
    )
    ax_d.text(
        0.0,
        0.42,
        "Current safe use\n1. Appendix figure\n2. Discussion support\n3. Not for headline gastric ICB-resistance claims",
        fontsize=10.5,
        transform=ax_d.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFFFFF", "edgecolor": "#D6D8DB"},
    )

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle("Supplementary Figure 2. Appendix gastric ICB context from the Kim cohort", fontsize=15.5, y=0.985)
    save_figure(
        fig,
        "SupplementaryFigure2_appendix_kim_icb_context",
        "Appendix gastric ICB context from the Kim cohort",
        [
            "results/treatment_validation_direct_kim/Kim_direct_icb_sample_scores_avg.csv",
            "results/treatment_validation_direct_kim/Kim_direct_icb_composite_tests_avg.csv",
            "results/treatment_validation_direct_kim/Kim_direct_icb_group_response_rates_avg.csv",
        ],
        SUPP_DIR,
    )


def build_supplementary_figure_3() -> None:
    gse251950_corr = load_csv("results/external_spatial_gse251950/GSE251950_sample_correlations.csv")
    cosmx_neighbor = load_csv("results/external_spatial_gse308624/GSE308624_macrophage_neighbor_summary.csv")
    peritoneal_frac = load_csv("results/peritoneal_state_bias/GSE183904_peritoneal_sample_state_fractions.csv")
    peritoneal_means = load_csv("results/peritoneal_state_bias/GSE183904_peritoneal_sample_means.csv")

    fig = plt.figure(figsize=(15.8, 9.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_gse251950_metric_summary(ax_a, gse251950_corr)
    plot_neighbor_summary_full(ax_b, cosmx_neighbor)
    plot_state_fraction_box(ax_c, peritoneal_frac)
    plot_peritoneal_feature_compare(ax_d, peritoneal_means)

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle("Supplementary Figure 3. External spatial and peritoneal ecology details", fontsize=15.5, y=0.985)
    save_figure(
        fig,
        "SupplementaryFigure3_external_spatial_peritoneal_details",
        "External spatial and peritoneal ecology details",
        [
            "results/external_spatial_gse251950/GSE251950_sample_correlations.csv",
            "results/external_spatial_gse308624/GSE308624_macrophage_neighbor_summary.csv",
            "results/peritoneal_state_bias/GSE183904_peritoneal_sample_state_fractions.csv",
            "results/peritoneal_state_bias/GSE183904_peritoneal_sample_means.csv",
        ],
        SUPP_DIR,
    )


def plot_bulk_axis_scatter(ax: plt.Axes, ecology_scores: pd.DataFrame, dataset_id: str) -> None:
    sub = ecology_scores[ecology_scores["dataset_id"] == dataset_id].copy()
    palette = {
        "burden_high__polar_low": COLORS["late"],
        "burden_high__polar_high": COLORS["caf"],
        "burden_low__polar_high": COLORS["early"],
        "burden_low__polar_low": COLORS["grey"],
    }
    label = {"TCGA_STAD": "TCGA-STAD", "GSE84437": "GSE84437"}[dataset_id]
    for group, color in palette.items():
        cur = sub[sub["dual_axis_group"] == group]
        ax.scatter(
            cur["trajectory_polarized_axis"],
            cur["our_suppressive_axis"],
            s=26,
            c=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.72,
            label=group,
        )
    ax.axhline(0, color="#CFD5DB", lw=1.0, ls="--")
    ax.axvline(0, color="#CFD5DB", lw=1.0, ls="--")
    ax.set_xlabel("Trajectory-polarized axis")
    ax.set_ylabel("Suppressive axis")
    add_section_tag(ax, label, COLORS["late"] if dataset_id == "TCGA_STAD" else COLORS["tls"])
    soften_axes(ax)


def plot_bulk_imbalance_box(ax: plt.Axes, ecology_scores: pd.DataFrame) -> None:
    plot_df = ecology_scores.copy()
    plot_df["Group"] = plot_df["dual_axis_group"].map(
        {
            "burden_high__polar_low": "HS/LAI",
            "burden_high__polar_high": "HS/HAI",
            "burden_low__polar_high": "LS/HAI",
            "burden_low__polar_low": "LS/LAI",
        }
    )
    plot_df["Dataset"] = plot_df["dataset_id"].map({"TCGA_STAD": "TCGA", "GSE84437": "84437"})
    order = ["HS/LAI", "HS/HAI", "LS/HAI", "LS/LAI"]
    palette = {"TCGA": COLORS["late"], "84437": COLORS["tls"]}
    sns.boxplot(data=plot_df, x="Group", y="stromal_adaptive_imbalance", hue="Dataset", order=order, palette=palette, showcaps=False, fliersize=0, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Stromal-adaptive imbalance")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    add_section_tag(ax, "Group imbalance", COLORS["accent"])
    soften_axes(ax)


def plot_group_ecology_scatter(ax: plt.Axes, ecology_group: pd.DataFrame) -> None:
    plot_df = ecology_group.copy()
    plot_df["Dataset"] = plot_df["dataset_id"].map({"TCGA_STAD": "TCGA", "GSE84437": "84437"})
    plot_df["Group"] = plot_df["dual_axis_group"].map(
        {
            "burden_high__polar_low": "HS/LAI",
            "burden_high__polar_high": "HS/HAI",
            "burden_low__polar_high": "LS/HAI",
            "burden_low__polar_low": "LS/LAI",
        }
    )
    for dataset, marker in [("TCGA", "o"), ("84437", "s")]:
        cur = plot_df[plot_df["Dataset"] == dataset]
        ax.scatter(
            cur["tls_b_score_mean"],
            cur["caf_input_score_mean"],
            s=85,
            c=cur["stromal_adaptive_imbalance_mean"],
            cmap="coolwarm",
            marker=marker,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for row in cur.itertuples():
            ax.text(row.tls_b_score_mean + 0.005, row.caf_input_score_mean + 0.005, f"{dataset} {row.Group}", fontsize=7.5)
    ax.set_xlabel("TLS-B mean")
    ax.set_ylabel("CAF-input mean")
    add_section_tag(ax, "Group ecology map", COLORS["caf"])
    soften_axes(ax)


def build_supplementary_figure_4() -> None:
    ecology_scores = load_csv("results/signature_benchmark/bulk_ecology_context_scores.csv")
    ecology_group = load_csv("results/signature_benchmark/bulk_ecology_group_summary.csv")

    fig = plt.figure(figsize=(15.8, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_bulk_axis_scatter(ax_a, ecology_scores, "TCGA_STAD")
    plot_bulk_axis_scatter(ax_b, ecology_scores, "GSE84437")
    plot_bulk_imbalance_box(ax_c, ecology_scores)
    plot_group_ecology_scatter(ax_d, ecology_group)

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle("Supplementary Figure 4. Bulk dual-axis ecology details", fontsize=15.5, y=0.985)
    save_figure(
        fig,
        "SupplementaryFigure4_bulk_dual_axis_details",
        "Bulk dual-axis ecology details",
        [
            "results/signature_benchmark/bulk_ecology_context_scores.csv",
            "results/signature_benchmark/bulk_ecology_group_summary.csv",
        ],
        SUPP_DIR,
    )


def plot_study_schematic(ax: plt.Axes) -> None:
    ax.axis("off")
    ink = "#111111"
    grey = "#555555"
    pale = "#F7F7F7"
    line = "#222222"
    soft_line = "#CFCFCF"

    def rect(xywh: tuple[float, float, float, float], *, facecolor: str = "#FFFFFF", linewidth: float = 1.0) -> None:
        x, y, w, h = xywh
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=line,
                linewidth=linewidth,
            )
        )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color=line,
            )
        )

    ax.text(0.04, 0.92, "Study workflow", transform=ax.transAxes, fontsize=19, fontweight="bold", color=ink)
    ax.text(
        0.04,
        0.875,
        "Public gastric cancer cohorts were organized into one reproducible evidence chain.",
        transform=ax.transAxes,
        fontsize=10.5,
        color=grey,
    )
    ax.plot([0.04, 0.96], [0.845, 0.845], transform=ax.transAxes, color=ink, lw=1.1)

    rect((0.04, 0.18, 0.24, 0.58), facecolor=pale, linewidth=1.0)
    ax.text(0.06, 0.72, "Data inputs", transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=ink)
    cohort_rows = [
        ("Single-cell", "GSE206785\nGSE163558\nGSE183904"),
        ("Spatial", "GSE270678\nGSE251950\nGSE308624 CosMx"),
        ("Bulk", "TCGA-STAD\nGSE84437"),
    ]
    y_positions = [0.575, 0.395, 0.235]
    for (heading, body), y in zip(cohort_rows, y_positions):
        ax.text(0.06, y + 0.085, heading, transform=ax.transAxes, fontsize=9.8, fontweight="bold", color=ink)
        ax.text(0.06, y + 0.010, body, transform=ax.transAxes, fontsize=8.8, color=grey, va="bottom", linespacing=1.25)
        ax.plot([0.06, 0.25], [y - 0.015, y - 0.015], transform=ax.transAxes, color=soft_line, lw=0.8)

    step_specs = [
        ("01", "State discovery", "Early, bridge, and terminal\nsuppressive myeloid states", "GSE206785", (0.33, 0.53, 0.145, 0.20)),
        ("02", "Formation input", "CAF-associated ligands and\nINHBA-linked bridge state", "GSE206785", (0.50, 0.53, 0.145, 0.20)),
        ("03", "Spatial niche", "CAF-rich suppressive hotspots\nwith lower TLS-related core signal", "GSE270678 + external", (0.67, 0.53, 0.145, 0.20)),
        ("04", "Clinical transfer", "Exploratory peritoneal\nreceptor panel and\nbulk context", "GSE183904 + bulk", (0.84, 0.53, 0.12, 0.20)),
    ]
    ax.text(0.34, 0.76, "Analysis chain", transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=ink)
    for number, title, body, dataset, xywh in step_specs:
        x, y, w, h = xywh
        rect(xywh, facecolor="#FFFFFF", linewidth=1.15)
        ax.text(x + 0.012, y + h - 0.035, number, transform=ax.transAxes, fontsize=8.6, color=grey, fontweight="bold")
        ax.text(x + 0.012, y + h - 0.072, title, transform=ax.transAxes, fontsize=9.9, color=ink, fontweight="bold")
        ax.text(x + 0.012, y + 0.060, body, transform=ax.transAxes, fontsize=8.1, color=ink, va="bottom", linespacing=1.15)
        ax.text(x + 0.012, y + 0.024, dataset, transform=ax.transAxes, fontsize=7.8, color=grey, va="bottom")

    for start, end in [((0.475, 0.63), (0.50, 0.63)), ((0.645, 0.63), (0.67, 0.63)), ((0.815, 0.63), (0.84, 0.63))]:
        arrow(start, end)

    ax.text(0.34, 0.40, "Anchor readouts", transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=ink)
    readouts = [
        ("E01", "CAF dominance is associated with the myeloid-axis score", "rho=0.806"),
        ("E02", "Fibroblast INHBA marks a candidate bridge state", "rho=0.468"),
        ("E03-E05", "Suppressive hotspots are CAF-rich and TLS-related-core low", "delta=0.095; rho=-0.102"),
        ("E07-E09", "Peritoneal receptor-panel and bulk readouts are exploratory", "index P=0.045; HR=1.254"),
    ]
    for idx, (eid, title, metric) in enumerate(readouts):
        y = 0.335 - idx * 0.055
        ax.text(0.34, y, eid, transform=ax.transAxes, fontsize=8.8, fontweight="bold", color=ink, va="center")
        ax.text(0.42, y, title, transform=ax.transAxes, fontsize=8.8, color=ink, va="center")
        ax.text(0.88, y, metric, transform=ax.transAxes, fontsize=8.4, color=grey, va="center", ha="right")
        ax.plot([0.34, 0.90], [y - 0.027, y - 0.027], transform=ax.transAxes, color=soft_line, lw=0.75)


def plot_working_model(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.text(0.03, 0.94, "CAF-shaped formation and ecological landing of the suppressive myeloid state", transform=ax.transAxes, fontsize=16, fontweight="bold")
    ax.text(0.03, 0.90, "An association-based model linking CAF-related signals, spatial context, exploratory peritoneal receptor-panel observations, and bulk transfer.", transform=ax.transAxes, fontsize=10.2, color="#596574")

    draw_round_box(ax, (0.04, 0.55, 0.92, 0.26), edgecolor="#D7DEE5", facecolor="#FCFDFC", linewidth=1.15, rounding=0.024)
    ax.text(0.06, 0.77, "Formative stromal corridor", transform=ax.transAxes, fontsize=12.5, fontweight="bold")
    ax.text(0.06, 0.728, "Activated CAF inputs are associated with a bridge state and a receptor-panel terminal-like program.", transform=ax.transAxes, fontsize=9.2, color="#5A6674")
    draw_chip(ax, (0.86, 0.76, 0.07, 0.03), "E01-E02", facecolor=COLORS["caf"], fontsize=7.6)

    top_nodes = [
        ("Activated CAF", "INHBA / CSF1 / IL34 / TGFB1", COLORS["caf"], (0.18, 0.64)),
        ("Bridge myeloid", "INHBA-associated intermediate", COLORS["intermediate"], (0.50, 0.64)),
        ("Terminal suppressive", "receptor-switched landing state", COLORS["late"], (0.82, 0.64)),
    ]
    for title, body, color, center in top_nodes:
        draw_cell_cloud(ax, center, color, scale=1.05)
        ax.text(center[0], center[1] + 0.058, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=11.8, fontweight="bold", color=color)
        ax.text(center[0], center[1] - 0.018, body, transform=ax.transAxes, ha="center", va="top", fontsize=9.1, color=COLORS["dark"])

    for start, end, label in [((0.27, 0.64), (0.41, 0.64), "formative pressure"), ((0.59, 0.64), (0.73, 0.64), "polarization + switch")]:
        ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color="#7B8794"))
        ax.text((start[0] + end[0]) / 2, start[1] + 0.032, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.1, color="#6C7784")

    spatial_xywh = (0.05, 0.15, 0.40, 0.28)
    peritoneal_xywh = (0.49, 0.15, 0.22, 0.28)
    bulk_xywh = (0.75, 0.15, 0.20, 0.28)

    draw_round_box(ax, spatial_xywh, edgecolor=COLORS["tls"], facecolor="#FBFCFE", linewidth=1.5, rounding=0.025)
    draw_chip(ax, (0.37, 0.39, 0.07, 0.028), "E03-E05", facecolor=COLORS["tls"], fontsize=7.5)
    ax.text(0.07, 0.39, "Ecological landing", transform=ax.transAxes, fontsize=12.3, fontweight="bold", color=COLORS["tls"])
    ax.text(0.07, 0.355, "Hotspots remain CAF-rich and displaced from TLS-like cores.", transform=ax.transAxes, fontsize=9.0, color=COLORS["dark"])
    ax.add_patch(Rectangle((0.09, 0.205), 0.065, 0.13, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["caf"], 0.12), edgecolor=COLORS["caf"], linewidth=1.0))
    ax.text(0.122, 0.194, "CAF-rich\nmargin", transform=ax.transAxes, ha="center", va="top", fontsize=8.1, color=COLORS["caf"])
    ax.add_patch(Circle((0.23, 0.255), 0.048, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["late"], 0.16), edgecolor=COLORS["late"], linewidth=1.2))
    ax.text(0.23, 0.255, "Suppressive\nhotspot", transform=ax.transAxes, ha="center", va="center", fontsize=8.7, color=COLORS["late"], fontweight="bold")
    ax.add_patch(Circle((0.355, 0.248), 0.042, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["tls"], 0.16), edgecolor=COLORS["tls"], linewidth=1.2))
    ax.text(0.355, 0.248, "TLS-like\ncore", transform=ax.transAxes, ha="center", va="center", fontsize=8.4, color=COLORS["tls"], fontweight="bold")
    for cx, cy in [(0.17, 0.305), (0.16, 0.215), (0.205, 0.325), (0.285, 0.19)]:
        ax.add_patch(Circle((cx, cy), 0.012, transform=ax.transAxes, facecolor=COLORS["caf"], edgecolor="white", linewidth=0.6))
    ax.add_patch(FancyArrowPatch((0.27, 0.255), (0.315, 0.248), transform=ax.transAxes, arrowstyle="-", linestyle="--", linewidth=1.6, color="#7B8794"))
    ax.text(0.292, 0.275, "TLS exclusion", transform=ax.transAxes, fontsize=8.0, color="#6D7783", ha="center")

    draw_round_box(ax, peritoneal_xywh, edgecolor=COLORS["peritoneum"], facecolor="#FDFCFA", linewidth=1.5, rounding=0.025)
    draw_chip(ax, (0.63, 0.39, 0.07, 0.028), "E07-E08", facecolor=COLORS["peritoneum"], fontsize=7.5)
    ax.text(0.51, 0.39, "Peritoneal skew", transform=ax.transAxes, fontsize=12.0, fontweight="bold", color=COLORS["peritoneum"])
    ax.text(0.51, 0.355, "Peritoneal tumors preferentially enrich terminal receptor switching.", transform=ax.transAxes, fontsize=8.9, color=COLORS["dark"])
    ax.add_patch(Ellipse((0.60, 0.255), 0.13, 0.105, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["peritoneum"], 0.10), edgecolor=COLORS["peritoneum"], linewidth=1.2))
    for cx, cy, size in [(0.565, 0.268, 0.014), (0.620, 0.278, 0.013), (0.650, 0.228, 0.015), (0.585, 0.225, 0.013), (0.618, 0.235, 0.012)]:
        ax.add_patch(Circle((cx, cy), size, transform=ax.transAxes, facecolor=COLORS["late"], edgecolor="white", linewidth=0.6))
    ax.text(0.60, 0.195, "terminal-state skew", transform=ax.transAxes, ha="center", fontsize=8.4, color=COLORS["late"])

    draw_round_box(ax, bulk_xywh, edgecolor=COLORS["accent"], facecolor="#FBFCFE", linewidth=1.5, rounding=0.025)
    draw_chip(ax, (0.875, 0.39, 0.06, 0.028), "E09b", facecolor=COLORS["accent"], fontsize=7.2)
    ax.text(0.77, 0.39, "Bulk transfer", transform=ax.transAxes, fontsize=12.0, fontweight="bold", color=COLORS["accent"])
    ax.text(0.77, 0.355, "The worst group is stromal-high and adaptive-immune-low.", transform=ax.transAxes, fontsize=8.8, color=COLORS["dark"])
    ax.plot([0.81, 0.81], [0.205, 0.315], transform=ax.transAxes, color="#97A4B2", lw=1.2)
    ax.plot([0.81, 0.91], [0.205, 0.205], transform=ax.transAxes, color="#97A4B2", lw=1.2)
    ax.text(0.91, 0.188, "Stroma", transform=ax.transAxes, fontsize=7.8, ha="right", color="#6D7884")
    ax.text(0.792, 0.315, "Adaptive", transform=ax.transAxes, fontsize=7.8, rotation=90, va="bottom", color="#6D7884")
    ax.add_patch(Rectangle((0.86, 0.205), 0.05, 0.05, transform=ax.transAxes, facecolor=mcolors.to_rgba(COLORS["late"], 0.18), edgecolor=COLORS["late"], linewidth=1.0))
    ax.text(0.885, 0.23, "worst", transform=ax.transAxes, ha="center", va="center", fontsize=8.2, color=COLORS["late"], fontweight="bold")
    for px, py, color in [(0.835, 0.285, COLORS["tls"]), (0.875, 0.285, COLORS["caf"]), (0.835, 0.225, COLORS["early"]), (0.885, 0.225, COLORS["late"])]:
        ax.add_patch(Circle((px, py), 0.009, transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=0.5))

    ax.add_patch(FancyArrowPatch((0.82, 0.58), (0.82, 0.46), transform=ax.transAxes, arrowstyle="-", linewidth=1.8, color="#7B8794"))
    for end in [(0.25, 0.43), (0.60, 0.43), (0.85, 0.43)]:
        ax.add_patch(FancyArrowPatch((0.82, 0.46), end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=15, linewidth=1.8, color="#7B8794"))


def plot_kim_index_distribution(
    ax: plt.Axes,
    scores: pd.DataFrame,
    tests: pd.DataFrame,
    metric: str,
    ylabel: str,
    section_title: str,
    section_color: str,
) -> None:
    order = ["R", "NR"]
    palette = {"R": COLORS["early"], "NR": COLORS["late"]}
    sns.violinplot(
        data=scores,
        x="response_label",
        y=metric,
        order=order,
        hue="response_label",
        palette=palette,
        legend=False,
        inner=None,
        linewidth=1.3,
        cut=0,
        saturation=1,
        ax=ax,
    )
    sns.boxplot(
        data=scores,
        x="response_label",
        y=metric,
        order=order,
        width=0.22,
        showcaps=False,
        showfliers=False,
        boxprops={"facecolor": "#FFFFFF", "edgecolor": "#5A6674", "linewidth": 1.1},
        whiskerprops={"visible": False},
        medianprops={"color": "#334155", "linewidth": 1.4},
        ax=ax,
    )
    sns.stripplot(
        data=scores,
        x="response_label",
        y=metric,
        order=order,
        hue="response_label",
        palette=palette,
        dodge=False,
        legend=False,
        size=5.4,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.94,
        ax=ax,
    )
    top_test = tests[tests["metric"] == metric].iloc[0]
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    add_section_tag(ax, section_title, section_color)
    add_stat_box(ax, f"AUC={top_test['response_oriented_auc']:.3f}\n{format_pvalue(top_test['pvalue'])}")
    soften_axes(ax)


def plot_kim_response_rate_summary(ax: plt.Axes, kim_rates: pd.DataFrame) -> None:
    plot_df = kim_rates.copy()
    plot_df["metric_label"] = plot_df["metric"].map(
        {
            "suppressive_ecology_index": "Suppressive",
            "formation_ecology_index": "Formation",
        }
    )
    plot_df["split_label"] = plot_df["group_definition"].map(
        {
            "median_split": "Median split",
            "upper_vs_lower_tertile": "Upper vs lower tertile",
        }
    )
    plot_df["label"] = plot_df["metric_label"] + " | " + plot_df["split_label"]
    order = [
        "Suppressive | Median split",
        "Formation | Median split",
        "Suppressive | Upper vs lower tertile",
        "Formation | Upper vs lower tertile",
    ]
    plot_df["label"] = pd.Categorical(plot_df["label"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("label")
    y = np.arange(len(plot_df))
    ax.hlines(y, plot_df["response_rate_high"], plot_df["response_rate_low"], color="#CFD5DB", linewidth=2.0)
    ax.scatter(plot_df["response_rate_high"], y, s=128, c=COLORS["late"], edgecolor="white", linewidth=1.1, label="High-score group", zorder=3)
    ax.scatter(plot_df["response_rate_low"], y, s=128, c=COLORS["early"], edgecolor="white", linewidth=1.1, label="Low-score group", zorder=3)
    for yi, row in zip(y, plot_df.itertuples()):
        ax.text(max(row.response_rate_high, row.response_rate_low) + 0.018, yi, format_pvalue(row.fisher_pvalue), fontsize=7.4, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8.4)
    ax.set_xlabel("Response rate")
    add_section_tag(ax, "Group-response contrast", COLORS["accent"])
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")
    soften_axes(ax)


def plot_kim_appendix_card(ax: plt.Axes, tests: pd.DataFrame, clinical: pd.DataFrame) -> None:
    ax.axis("off")
    n_total = int(len(clinical))
    n_r = int((clinical["response_label"] == "R").sum())
    n_nr = int((clinical["response_label"] == "NR").sum())
    suppressive_auc = float(tests.loc[tests["metric"] == "suppressive_ecology_index", "response_oriented_auc"].iloc[0])
    formation_auc = float(tests.loc[tests["metric"] == "formation_ecology_index", "response_oriented_auc"].iloc[0])

    draw_round_box(ax, (0.03, 0.58, 0.92, 0.31), edgecolor=COLORS["accent"], facecolor="#FBFCFE", linewidth=1.35, rounding=0.024)
    ax.text(0.06, 0.83, "Appendix positioning", transform=ax.transAxes, fontsize=13.0, fontweight="bold")
    ax.text(0.06, 0.75, "This cohort is retained as supportive gastric ICB context rather than headline proof.", transform=ax.transAxes, fontsize=9.7, color=COLORS["dark"])
    draw_chip(ax, (0.06, 0.64, 0.12, 0.07), f"n = {n_total}", facecolor=COLORS["accent"], fontsize=10.0)
    draw_chip(ax, (0.20, 0.64, 0.12, 0.07), f"R = {n_r}", facecolor=COLORS["early"], fontsize=10.0)
    draw_chip(ax, (0.34, 0.64, 0.12, 0.07), f"NR = {n_nr}", facecolor=COLORS["late"], fontsize=10.0)
    ax.text(0.55, 0.68, f"Suppressive AUC {suppressive_auc:.3f}", transform=ax.transAxes, fontsize=9.2, color=COLORS["late"], fontweight="bold")
    ax.text(0.55, 0.62, f"Formation AUC {formation_auc:.3f}", transform=ax.transAxes, fontsize=9.2, color=COLORS["caf"], fontweight="bold")

    draw_round_box(ax, (0.03, 0.14, 0.92, 0.34), edgecolor="#D6DCE3", facecolor="#FFFFFF", linewidth=1.10, rounding=0.024)
    ax.text(0.06, 0.41, "Safe manuscript use", transform=ax.transAxes, fontsize=11.2, fontweight="bold")
    use_lines = [
        "1. Appendix-only gastric ICB context.",
        "2. Supports the stromal-high / adaptive-low ecology interpretation.",
        "3. Does not substitute for a direct custom-signature response matrix.",
    ]
    for idx, text in enumerate(use_lines):
        ax.text(0.06, 0.33 - idx * 0.09, text, transform=ax.transAxes, fontsize=9.5, color=COLORS["dark"])


def build_supplementary_figure_1() -> None:
    cluster_program = load_csv("results/myeloid_validation/GSE163558_cluster_tissue_program.csv")
    tissue_burden = load_csv("results/myeloid_external_atlas/GSE183904_myeloid_tissue_cluster_burden.csv")
    receptor_switch = load_csv("results/cell_crosstalk/GSE206785_myeloid_receptor_switch.csv")
    regulator = load_csv("results/mechanism_trajectory/GSE206785_curated_regulator_dynamics.csv")

    fig = plt.figure(figsize=(15.8, 9.4))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.18, 0.92], hspace=0.30, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    heat_a = cluster_program.pivot_table(index="leiden", columns="tissue_group", values="mean").fillna(0)
    heat_a = zscore_frame(heat_a).rename(
        columns={
            "adjacent_normal": "Adj normal",
            "primary_tumor": "Primary",
            "lymph_node_met": "LN met",
            "liver_met": "Liver met",
            "peritoneal_met": "Peritoneal met",
            "ovary_met": "Ovary met",
        }
    )
    plot_bubble_heatmap(ax_a, heat_a, cmap=sns.diverging_palette(215, 25, as_cmap=True), cbar_label="Cluster z-score", size_scale=520, xtick_rotation=28)
    add_section_tag(ax_a, "GSE163558 cluster map", COLORS["accent"])
    ax_a.set_ylabel("Leiden cluster", fontsize=9.2)

    burden_plot = tissue_burden.set_index("tissue_group")[["early_cluster_fraction", "suppressive_cluster_fraction", "median_axis_score", "median_program_score"]]
    burden_plot = zscore_frame(burden_plot).rename(
        index={
            "peritoneum_tumor": "Peritoneum tumor",
            "stomach_normal": "Stomach normal",
            "stomach_tumor": "Stomach tumor",
            "peritoneum_normal": "Peritoneum normal",
        },
        columns={
            "early_cluster_fraction": "Early frac",
            "suppressive_cluster_fraction": "Late frac",
            "median_axis_score": "Axis",
            "median_program_score": "Program",
        },
    )
    plot_bubble_heatmap(ax_b, burden_plot, cmap=sns.diverging_palette(215, 25, as_cmap=True), cbar_label="Tissue z-score", size_scale=820, xtick_rotation=18)
    add_section_tag(ax_b, "GSE183904 tissue burden", COLORS["peritoneum"])
    add_stat_box(ax_b, "Peritoneal tumors carry\nthe strongest late-state load", loc="lower right")

    receptor_plot = receptor_switch.sort_values("suppressive_vs_early_ratio", ascending=False).head(10).set_index("gene")[
        ["myeloid_early_tumor", "myeloid_transition_inhba_tumor", "myeloid_suppressive_tumor"]
    ]
    receptor_plot = zscore_frame(receptor_plot).rename(
        columns={
            "myeloid_early_tumor": "Early",
            "myeloid_transition_inhba_tumor": "Bridge",
            "myeloid_suppressive_tumor": "Late",
        }
    )
    plot_bubble_heatmap(
        ax_c,
        receptor_plot,
        cmap=sns.color_palette("blend:#F5F2EC,#4F7AA3,#B5533D", as_cmap=True),
        cbar_label="Gene z-score",
        size_scale=620,
        xtick_rotation=0,
    )
    add_section_tag(ax_c, "Receptor rewiring", COLORS["late"])
    add_stat_box(ax_c, "Late-state receptors\nexpand beyond the bridge", loc="lower right")

    reg_plot = regulator.sort_values("rho_with_axis_score", ascending=False).head(12).sort_values("rho_with_axis_score")
    y = np.arange(len(reg_plot))
    ax_d.hlines(y, 0, reg_plot["rho_with_axis_score"], color="#CBD2D9", linewidth=2)
    sc = ax_d.scatter(reg_plot["rho_with_axis_score"], y, s=115, c=reg_plot["late_minus_early"], cmap="RdYlBu_r", edgecolor="white", linewidth=1.0, zorder=3)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(reg_plot["gene"], fontsize=8.5)
    ax_d.set_xlabel("rho with trajectory axis")
    add_section_tag(ax_d, "Late regulators", COLORS["intermediate"])
    soften_axes(ax_d)
    cax = inset_axes(ax_d, width="5%", height="70%", loc="lower left", bbox_to_anchor=(1.02, 0.14, 1, 1), bbox_transform=ax_d.transAxes, borderpad=0)
    cb = plt.colorbar(sc, cax=cax)
    cb.outline.set_edgecolor("#D6DCE3")
    cb.set_label("Late minus early", fontsize=8.1)
    cb.ax.tick_params(labelsize=7.6)

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle("Supplementary Figure 1. External validation and receptor-switch details", fontsize=15.4, y=0.985)
    save_figure(
        fig,
        "SupplementaryFigure1_external_validation_details",
        "External validation and receptor-switch details",
        [
            "results/myeloid_validation/GSE163558_cluster_tissue_program.csv",
            "results/myeloid_external_atlas/GSE183904_myeloid_tissue_cluster_burden.csv",
            "results/cell_crosstalk/GSE206785_myeloid_receptor_switch.csv",
            "results/mechanism_trajectory/GSE206785_curated_regulator_dynamics.csv",
        ],
        SUPP_DIR,
    )


def build_supplementary_figure_2() -> None:
    kim_scores = load_csv("results/treatment_validation_direct_kim/Kim_direct_icb_sample_scores_avg.csv")
    kim_tests = load_csv("results/treatment_validation_direct_kim/Kim_direct_icb_composite_tests_avg.csv")
    kim_rates = load_csv("results/treatment_validation_direct_kim/Kim_direct_icb_group_response_rates_avg.csv")
    kim_clinical = load_csv("results/treatment_validation_direct_kim/Kim_direct_icb_clinical_table.csv")

    fig = plt.figure(figsize=(15.2, 8.8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_kim_index_distribution(
        ax_a,
        kim_scores,
        kim_tests,
        "suppressive_ecology_index",
        "Suppressive ecology index",
        "Suppressive ecology",
        COLORS["late"],
    )
    plot_kim_index_distribution(
        ax_b,
        kim_scores,
        kim_tests,
        "formation_ecology_index",
        "Formation ecology index",
        "Formation ecology",
        COLORS["caf"],
    )
    plot_kim_response_rate_summary(ax_c, kim_rates)
    plot_kim_appendix_card(ax_d, kim_tests, kim_clinical)

    for label, ax in zip(list("ABCD"), [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle("Supplementary Figure 2. Appendix gastric ICB context from the Kim cohort", fontsize=15.4, y=0.985)
    save_figure(
        fig,
        "SupplementaryFigure2_appendix_kim_icb_context",
        "Appendix gastric ICB context from the Kim cohort",
        [
            "results/treatment_validation_direct_kim/Kim_direct_icb_sample_scores_avg.csv",
            "results/treatment_validation_direct_kim/Kim_direct_icb_composite_tests_avg.csv",
            "results/treatment_validation_direct_kim/Kim_direct_icb_group_response_rates_avg.csv",
        ],
        SUPP_DIR,
    )


def write_manifest_tables() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(FIGURE_MANIFEST).copy()
    manifest_df["outputs"] = manifest_df["outputs"].apply(lambda x: "; ".join(x))
    manifest_df["source_files"] = manifest_df["source_files"].apply(lambda x: "; ".join(x))
    manifest_csv = TABLE_DIR / "figure_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    lines = ["# Figure Manifest", "", "| figure_id | title | outputs | source_files |", "| --- | --- | --- | --- |"]
    for row in FIGURE_MANIFEST:
        outputs = "<br>".join(row["outputs"])
        sources = "<br>".join(row["source_files"])
        lines.append(f"| {row['figure_id']} | {row['title']} | {outputs} | {sources} |")
    (TABLE_DIR / "figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(TABLE_DIR / "figure_manifest.json", {"figures": FIGURE_MANIFEST})


def main() -> None:
    setup_style()
    FIGURE_MANIFEST.clear()
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    build_figure_1()
    build_figure_1b()
    build_figure_1c()
    build_figure_2()
    build_figure_2b()
    build_figure_3()
    build_figure_3b()
    build_figure_4()
    build_figure_4b()
    build_figure_5()
    build_supplementary_figure_1()
    build_supplementary_figure_2()
    build_supplementary_figure_3()
    build_supplementary_figure_4()
    write_manifest_tables()
    print(f"[figures] Generated {len(FIGURE_MANIFEST)} figure packages under {TABLE_DIR}")


if __name__ == "__main__":
    main()
