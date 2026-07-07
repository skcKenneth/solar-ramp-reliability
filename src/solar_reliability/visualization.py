from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "native": "#4D4D4D",
    "clean_cqr": "#0072B2",
    "augmented_cqr": "#E69F00",
    "mondrian_cqr": "#56B4E9",
    "hierarchical_cqr": "#009E73",
    "failure": "#D55E00",
    "secondary": "#CC79A7",
}

DISPLAY = {
    "native": "Native ensemble",
    "clean_cqr": "Clean CQR",
    "augmented_cqr": "Augmented CQR",
    "mondrian_cqr": "Condition CQR",
    "hierarchical_cqr": "Risk–mask hierarchical CQR",
    "clean": "No added outage",
    "irradiance_outage": "Irradiance outage",
    "weather_outage": "Weather outage",
    "recent_power_outage": "Recent-power outage",
    "combined_outage": "Combined outage",
    "non_ramp": "Non-ramp",
    "ramp": "Ramp",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.45, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
