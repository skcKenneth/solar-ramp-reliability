from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.visualization import DISPLAY, PALETTE, apply_publication_style, clean_axes, save_figure  # noqa: E402

apply_publication_style()
TABLES = ROOT / "outputs" / "tables"
CACHE = ROOT / "outputs" / "cache"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=9.5, fontweight="semibold", va="bottom")


def protocol_figure() -> None:
    fig, ax = plt.subplots(figsize=(6.3, 2.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.35, 0.14, 0.30, "15-min SCADA\nhistory", "#E6E6E6"),
        (0.22, 0.35, 0.15, 0.30, "Outage\nscenarios", "#FCE5D8"),
        (0.43, 0.35, 0.15, 0.30, "Frozen quantile\nensemble", "#DDEBF7"),
        (0.64, 0.55, 0.16, 0.24, "Source-only\ncalibration", "#E2F0D9"),
        (0.64, 0.17, 0.16, 0.24, "Sealed 2020-H2\ntest targets", "#F2F2F2"),
        (0.85, 0.35, 0.13, 0.30, "Conditional\nreliability audit", "#E8E1F2"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#555555", linewidth=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)
    arrows = [
        ((0.16, 0.50), (0.22, 0.50)),
        ((0.37, 0.50), (0.43, 0.50)),
        ((0.58, 0.50), (0.64, 0.67)),
        ((0.58, 0.50), (0.64, 0.29)),
        ((0.80, 0.67), (0.85, 0.54)),
        ((0.80, 0.29), (0.85, 0.46)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#555555"))
    ax.text(0.72, 0.88, "labels available", ha="center", fontsize=6.6, color="#555555")
    ax.text(0.72, 0.06, "labels used only for evaluation", ha="center", fontsize=6.6, color="#555555")
    save_figure(fig, FIGURES / "fig01_protocol")


def coverage_error_figure(summary: pd.DataFrame) -> None:
    methods = ["clean_cqr", "augmented_cqr", "mondrian_cqr", "hierarchical_cqr"]
    conditions = ["clean", "irradiance_outage", "weather_outage", "recent_power_outage", "combined_outage"]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.0), sharex=True, sharey=True)
    for col, horizon in enumerate([1, 4]):
        for row, ramp in enumerate(["non_ramp", "ramp"]):
            ax = axes[row, col]
            subset = summary[(summary.horizon_steps == horizon) & (summary.ramp_group == ramp)]
            offsets = np.linspace(-0.24, 0.24, len(methods))
            y = np.arange(len(conditions))
            for offset, method in zip(offsets, methods):
                values = [
                    subset[(subset.condition == condition) & (subset.method == method)].coverage_error.iloc[0]
                    for condition in conditions
                ]
                ax.scatter(values, y + offset, label=DISPLAY[method], color=PALETTE[method], s=18, zorder=3, edgecolor="white", linewidth=0.35)
            ax.axvline(0.05, color="#777777", linestyle="--", linewidth=0.8)
            ax.set_yticks(y, [DISPLAY[c] for c in conditions])
            ax.invert_yaxis()
            ax.set_xlim(0, 0.72)
            clean_axes(ax, grid_axis="x")
            ax.set_title(f"{horizon * 15} min · {DISPLAY[ramp]}", pad=5)
            if row == 1:
                ax.set_xlabel("Absolute 90% coverage error")
            panel_label(ax, chr(ord("a") + row * 2 + col))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.16, wspace=0.13, hspace=0.28)
    save_figure(fig, FIGURES / "fig02_conditional_coverage_error")


def pareto_figure(primary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for ax, horizon, label in zip(axes, [1, 4], ["a", "b"]):
        subset = primary[primary.horizon_steps == horizon]
        for _, row in subset.iterrows():
            ax.scatter(
                row.mean_width,
                row.worst_group_coverage_error,
                s=34,
                color=PALETTE[row.method],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
                label=DISPLAY[row.method],
            )
        ax.axhline(0.05, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Mean interval width (MW)")
        ax.set_ylabel("Worst-group coverage error")
        ax.set_title(f"{horizon * 15}-min horizon", pad=5)
        clean_axes(ax, grid_axis="both")
        panel_label(ax, label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.30, bottom=0.24)
    save_figure(fig, FIGURES / "fig03_coverage_width_tradeoff")


def improvement_heatmap(summary: pd.DataFrame) -> None:
    rows = []
    for horizon in [1, 4]:
        for condition in ["clean", "irradiance_outage", "weather_outage", "recent_power_outage", "combined_outage"]:
            for ramp in ["non_ramp", "ramp"]:
                a = summary[(summary.horizon_steps == horizon) & (summary.condition == condition) & (summary.ramp_group == ramp) & (summary.method == "clean_cqr")]
                b = summary[(summary.horizon_steps == horizon) & (summary.condition == condition) & (summary.ramp_group == ramp) & (summary.method == "hierarchical_cqr")]
                rows.append({"horizon": horizon, "condition": condition, "ramp": ramp, "improvement": float(a.coverage_error.iloc[0] - b.coverage_error.iloc[0])})
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), constrained_layout=True)
    vmax = max(abs(data.improvement.min()), abs(data.improvement.max()))
    for ax, horizon, label in zip(axes, [1, 4], ["a", "b"]):
        pivot = data[data.horizon == horizon].pivot(index="condition", columns="ramp", values="improvement").loc[
            ["clean", "irradiance_outage", "weather_outage", "recent_power_outage", "combined_outage"],
            ["non_ramp", "ramp"],
        ]
        im = ax.imshow(pivot.to_numpy(), cmap="PiYG", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(pivot.index)), [DISPLAY[i] for i in pivot.index])
        ax.set_xticks(range(len(pivot.columns)), [DISPLAY[i] for i in pivot.columns])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7, color="black")
        ax.set_title(f"{horizon * 15}-min horizon", pad=5)
        panel_label(ax, label)
    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.03)
    cbar.set_label("Reduction in absolute coverage error")
    save_figure(fig, FIGURES / "fig04_hierarchical_improvement_heatmap")


def example_interval_figure(predictions: pd.DataFrame) -> None:
    subset = predictions[
        (predictions.horizon_steps == 4)
        & (predictions.condition == "clean")
        & (predictions.method == "hierarchical_cqr")
    ].copy()
    day_scores = subset.groupby("date")["ramp_fraction"].max().sort_values(ascending=False)
    chosen = day_scores.index[0]
    day = subset[subset.date == chosen].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.fill_between(day.timestamp, day.lower, day.upper, color=PALETTE["hierarchical_cqr"], alpha=0.20, linewidth=0)
    ax.plot(day.timestamp, day.y, color="#222222", linewidth=1.2, label="Observed power")
    ax.plot(day.timestamp, day["median"], color=PALETTE["hierarchical_cqr"], linewidth=1.1, label="Median forecast")
    ramps = day.ramp_fraction >= 0.10
    ax.scatter(day.loc[ramps, "timestamp"], day.loc[ramps, "y"], color=PALETTE["failure"], s=16, zorder=4, label="≥10% capacity ramp")
    ax.set_ylabel("Power (MW)")
    ax.set_xlabel(f"Time on {pd.Timestamp(chosen).date()}")
    ax.set_ylim(0, 52)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    clean_axes(ax, grid_axis="y")
    save_figure(fig, FIGURES / "fig05_representative_ramp_interval")


def bootstrap_forest(bootstrap: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), sharex=True)
    labels_order = []
    for condition in ["clean", "irradiance_outage", "weather_outage", "recent_power_outage", "combined_outage"]:
        for ramp in ["non_ramp", "ramp"]:
            labels_order.append((condition, ramp))
    for ax, horizon, label in zip(axes, [1, 4], ["a", "b"]):
        sub = bootstrap[bootstrap.horizon_steps == horizon].copy()
        mapping = {(r.condition, r.ramp_group): r for r in sub.itertuples()}
        y = np.arange(len(labels_order))
        values, low, high = [], [], []
        for key in labels_order:
            row = mapping[key]
            values.append(row.difference_a_minus_b)
            low.append(row.ci_low)
            high.append(row.ci_high)
        values = np.asarray(values)
        ax.errorbar(values, y, xerr=[values - np.asarray(low), np.asarray(high) - values], fmt="o", color=PALETTE["hierarchical_cqr"], ecolor="#555555", elinewidth=0.8, capsize=2.0)
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.set_yticks(y, [f"{DISPLAY[c]} · {DISPLAY[r]}" for c, r in labels_order])
        ax.invert_yaxis()
        ax.set_xlabel("Coverage-error change\n(hierarchical − clean CQR)")
        ax.set_title(f"{horizon * 15}-min horizon", pad=5)
        clean_axes(ax, grid_axis="x")
        panel_label(ax, label)
    fig.subplots_adjust(wspace=0.20)
    save_figure(fig, FIGURES / "fig06_day_cluster_bootstrap")


def main() -> None:
    summary = pd.read_csv(TABLES / "conditional_metrics.csv")
    primary = pd.read_csv(TABLES / "method_primary_scores.csv")
    bootstrap = pd.read_csv(TABLES / "bootstrap_hierarchical_vs_clean.csv")
    predictions = pd.read_csv(CACHE / "pilot_predictions.csv.gz", parse_dates=["timestamp", "date"])
    protocol_figure()
    coverage_error_figure(summary)
    pareto_figure(primary)
    improvement_heatmap(summary)
    example_interval_figure(predictions)
    bootstrap_forest(bootstrap)
    print(f"Rendered figures to {FIGURES}")


if __name__ == "__main__":
    main()
