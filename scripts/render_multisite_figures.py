from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.visualization import apply_publication_style, clean_axes  # noqa: E402

COLORS = {
    "baseline": "#5477C4",
    "local": "#71B436",
    "risk": "#CC6F47",
    "rolling": "#7A828F",
    "oracle": "#B8A037",
    "threshold": "#1F2430",
    "secondary": "#8C8C8C",
    "soft_fail": "#FFEDDE",
    "soft_pass": "#D8ECBD",
    "soft_warn": "#FFF4C2",
    "panel": "#FCFCFD",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "ink": "#1F2430",
}

HEATMAP = LinearSegmentedColormap.from_list(
    "undercoverage_pressure",
    ["#F7FBF9", "#D8ECE6", "#F8E5C0", "#E79873", "#B54A53"],
)


def save(fig: plt.Figure, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    stem = figures / name
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def apply_lnai_visual_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.9,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.5,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.06,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def overview_protocol(tables: Path, figures: Path) -> None:
    audit = pd.read_csv(tables / "station_data_audit.csv")
    valid = audit[audit["status"] == "valid"].copy()
    verdict = json.loads((tables / "multisite_verdict.json").read_text(encoding="utf-8"))
    gates = pd.read_csv(tables / "candidate_station_gates.csv")
    local = gates[gates["method"] == "robust_local_cqr"].set_index("horizon_steps")

    fig = plt.figure(figsize=(6.0, 3.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 1.0], height_ratios=[0.64, 1.0], hspace=0.34, wspace=0.30)
    ax_flow = fig.add_subplot(gs[0, :])
    ax_station = fig.add_subplot(gs[1, 0])
    ax_verdict = fig.add_subplot(gs[1, 1])

    ax_flow.axis("off")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    steps = [
        ("Develop", "Site 1 only"),
        ("Freeze", "protocol"),
        ("Screen", "data integrity"),
        ("Confirm", "5 x 4 x 2 chunks"),
        ("Decide", "evaluation paper"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, ((title, body), x) in enumerate(zip(steps, xs)):
        color = COLORS["local"] if i in {1, 3} else COLORS["baseline"] if i == 0 else COLORS["threshold"]
        ax_flow.scatter([x], [0.60], s=176, color=color, edgecolor="white", linewidth=1.3, zorder=3)
        ax_flow.text(x, 0.60, str(i + 1), color="white", ha="center", va="center", fontsize=7.6, fontweight="bold")
        ax_flow.text(x, 0.36, title, ha="center", va="top", fontsize=7.6, fontweight="bold")
        ax_flow.text(x, 0.16, body, ha="center", va="top", fontsize=6.4, color="#4A4A4A")
        if i < len(xs) - 1:
            ax_flow.annotate(
                "",
                xy=(xs[i + 1] - 0.045, 0.58),
                xytext=(x + 0.045, 0.58),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#8E96A3"},
            )
    ax_flow.text(0.0, 0.98, "Frozen confirmation design", ha="left", va="top", fontsize=8.8, fontweight="bold")

    station_order = ["CSGS2", "CSGS5", "CSGS6", "CSGS7", "CSGS8", "CSGS3", "CSGS4"]
    station_rows = audit[audit["site_id"].isin(station_order)].set_index("site_id").loc[station_order].reset_index()
    station_rows = pd.concat(
        [
            station_rows[station_rows["status"] == "valid"],
            station_rows[station_rows["status"] != "valid"],
        ],
        ignore_index=True,
    )
    station_rows["y"] = np.arange(len(station_rows))
    is_valid = station_rows["status"] == "valid"
    valid_count = int(is_valid.sum())
    ax_station.set_title("External-site screen", loc="left", fontweight="bold", pad=6)
    ax_station.axhspan(-0.5, valid_count - 0.5, color=COLORS["soft_pass"], alpha=0.28, zorder=0)
    ax_station.axhspan(valid_count - 0.5, len(station_rows) - 0.5, color=COLORS["soft_fail"], alpha=0.32, zorder=0)
    ax_station.hlines(
        station_rows["y"],
        0,
        station_rows["nominal_capacity_mw"],
        color=np.where(is_valid, COLORS["local"], COLORS["risk"]),
        linewidth=1.3,
        alpha=0.85,
        zorder=1,
    )
    ax_station.scatter(
        station_rows.loc[is_valid, "nominal_capacity_mw"],
        station_rows.loc[is_valid, "y"],
        s=72,
        marker="o",
        color=COLORS["local"],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    excluded = station_rows.loc[~is_valid]
    ax_station.scatter(
        excluded["nominal_capacity_mw"],
        excluded["y"],
        s=84,
        marker="X",
        facecolors="white",
        edgecolors=COLORS["risk"],
        linewidth=1.3,
        zorder=4,
    )
    for row in station_rows.itertuples(index=False):
        color = COLORS["local"] if row.status == "valid" else COLORS["risk"]
        label_x = row.nominal_capacity_mw + 4
        label_ha = "left"
        if row.nominal_capacity_mw >= 125:
            label_x = row.nominal_capacity_mw + 3
        if row.status != "valid" and row.nominal_capacity_mw >= 120:
            label_x = row.nominal_capacity_mw - 18
            label_ha = "center"
        ax_station.text(label_x, row.y, f"{row.nominal_capacity_mw:.0f} MW", ha=label_ha, va="center", fontsize=5.7, color=color, fontweight="bold")
    ax_station.text(134, (valid_count - 1) / 2, "valid after\npre-outcome checks", ha="right", va="center", fontsize=5.9, color=COLORS["local"], fontweight="bold")
    ax_station.text(
        82,
        valid_count + 0.5,
        "2 excluded before scoring",
        ha="center",
        va="center",
        fontsize=5.9,
        color=COLORS["risk"],
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.5},
    )
    ax_station.set_yticks(station_rows["y"], station_rows["site_id"])
    ax_station.set_xlim(0, 142)
    ax_station.set_ylim(len(station_rows) - 0.45, -0.55)
    ax_station.set_xlabel("Nominal capacity (MW)")
    clean_axes(ax_station, "x")
    add_panel_label(ax_station, "A")

    ax_verdict.set_xlim(0, 1)
    ax_verdict.set_ylim(0, 4.6)
    ax_verdict.set_title("Decision ledger", loc="left", fontweight="bold", pad=6)
    ledger = [
        ("PASS", "phenomenon", f"{int(verdict['station_count'])}/5 stations", COLORS["local"]),
        ("PASS", "comparative gain", f"+{local.loc[1, 'mean_improvement']:.3f}/+{local.loc[4, 'mean_improvement']:.3f}", COLORS["local"]),
        ("FAIL", "absolute gate", "no deployable pass", COLORS["risk"]),
        ("LIMIT", "claim", "evaluation, not superiority", COLORS["oracle"]),
    ]
    for i, (status, label, text, color) in enumerate(ledger):
        yy = 3.7 - i * 0.78
        ax_verdict.hlines(yy, 0.18, 0.94, color=COLORS["grid"], linewidth=0.9, zorder=0)
        marker = "o" if status == "PASS" else "X" if status == "FAIL" else "s"
        ax_verdict.scatter([0.13], [yy], s=72, marker=marker, color=color if status != "FAIL" else "white", edgecolor=color, linewidth=1.3, zorder=3)
        ax_verdict.text(0.22, yy + 0.12, status, fontsize=6.0, fontweight="bold", color=color, ha="left", va="center")
        ax_verdict.text(0.22, yy - 0.12, label, fontsize=6.0, color="#555555", ha="left", va="center")
        ax_verdict.text(0.65, yy, text, fontsize=6.4, color=COLORS["ink"], ha="left", va="center")
    ax_verdict.text(
        0.02,
        0.08,
        f"Scored: {int(verdict['prediction_chunks'])} chunks; worst 15-min local-CQR undercoverage "
        f"{local.loc[1, 'max_method_worst_undercoverage']:.3f}.",
        fontsize=6.4,
        color="#555555",
        ha="left",
        va="bottom",
    )
    ax_verdict.axis("off")
    add_panel_label(ax_verdict, "B")
    save(fig, figures, "fig20_confirmation_overview")


def station_undercoverage(tables: Path, figures: Path) -> None:
    effects = pd.read_csv(tables / "station_level_effects.csv")
    local = effects[effects["method"] == "robust_local_cqr"].copy()
    stations = sorted(local["station"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.35), sharey=True)
    y = np.arange(len(stations))
    for ax, horizon, title in zip(axes, [1, 4], ["15-min horizon", "60-min horizon"]):
        sub = local[local["horizon_steps"] == horizon].set_index("station").loc[stations]
        baseline = sub["baseline_worst_undercoverage"].to_numpy()
        method = sub["method_worst_undercoverage"].to_numpy()
        improvement = baseline - method
        ax.axvspan(0.10, 0.78, color=COLORS["soft_fail"], alpha=0.18, zorder=0)
        for yi, base, meth, gain in zip(y, baseline, method, improvement):
            ax.plot([meth, base], [yi, yi], color=COLORS["axis"], linewidth=2.8, solid_capstyle="round", zorder=1)
            ax.annotate(
                "",
                xy=(meth, yi),
                xytext=(base, yi),
                arrowprops={"arrowstyle": "->", "lw": 1.35, "color": COLORS["local"], "shrinkA": 8, "shrinkB": 8},
                zorder=2,
            )
            ax.text(
                min(base, meth) + gain * 0.50,
                yi - 0.31,
                f"-{gain:.2f}",
                ha="center",
                va="center",
                fontsize=6.6,
                color=COLORS["local"],
                fontweight="bold",
            )
        ax.scatter(baseline, y, s=92, marker="o", facecolors="white", edgecolors=COLORS["baseline"], linewidths=1.6, label="Condition CQR", zorder=3)
        ax.scatter(method, y, s=92, marker="o", color=COLORS["local"], edgecolors="white", linewidths=1.0, label="Local CQR", zorder=4)
        ax.axvline(0.10, color=COLORS["threshold"], linestyle="--", linewidth=1.05)
        ax.text(0.107, -0.70, "absolute gate", ha="left", va="center", fontsize=6.4, color=COLORS["threshold"])
        ax.text(0.75, -0.70, "beyond gate", ha="right", va="center", fontsize=6.2, color=COLORS["risk"])
        ax.set_xlim(0, 0.78)
        ax.set_ylim(len(stations) - 0.35, -0.85)
        ax.set_xlabel("Worst ramp-group undercoverage")
        ax.set_title(title, loc="left", fontweight="bold")
        clean_axes(ax, "x")
    axes[0].set_yticks(y, stations)
    axes[0].set_ylabel("Confirmatory station")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS["baseline"], markeredgewidth=1.4, label="Condition CQR"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["local"], markeredgecolor="white", markeredgewidth=0.9, label="Local CQR"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.13, 0.86), ncol=2, borderaxespad=0, handletextpad=0.45, columnspacing=1.1)
    fig.suptitle("Local calibration moves every station left, but not far enough", x=0.03, y=0.99, ha="left", fontsize=8.8, fontweight="bold")
    fig.text(
        0.03,
        0.035,
        "Arrows show the station-level change from condition-CQR to descriptor-local CQR; labels are absolute reductions in worst-group undercoverage.",
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#555555",
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.26, top=0.70, wspace=0.16)
    save(fig, figures, "fig20_station_worst_undercoverage")


def candidate_gates(tables: Path, figures: Path) -> None:
    gates = pd.read_csv(tables / "candidate_station_gates.csv")
    labels = {
        "robust_local_cqr": "Local CQR",
        "robust_risk_mondrian": "Risk-Mondrian",
        "robust_rolling_cqr": "Rolling CQR",
    }
    order = [
        (4, "robust_local_cqr"),
        (4, "robust_risk_mondrian"),
        (4, "robust_rolling_cqr"),
        (1, "robust_local_cqr"),
        (1, "robust_risk_mondrian"),
        (1, "robust_rolling_cqr"),
    ]
    method_colors = {"robust_local_cqr": COLORS["local"], "robust_risk_mondrian": COLORS["risk"], "robust_rolling_cqr": COLORS["rolling"]}
    fig, ax = plt.subplots(figsize=(6.0, 3.20))
    rows = gates.set_index(["horizon_steps", "method"])
    y_positions = np.arange(len(order))
    for y, key in zip(y_positions, order):
        row = rows.loc[key]
        color = method_colors[key[1]]
        ci_low = float(row["station_bootstrap_ci_low"])
        ci_high = float(row["station_bootstrap_ci_high"])
        mean = float(row["mean_improvement"])
        passed = bool(row["comparative_gate_passed"])
        ax.plot([ci_low, ci_high], [y, y], color=color, linewidth=2.3, solid_capstyle="round", zorder=2)
        marker = "o" if passed else "X"
        face = color if passed else "white"
        ax.scatter([mean], [y], s=108, marker=marker, facecolors=face, edgecolors=color, linewidths=1.6, zorder=3)
        value_x = mean + 0.010
        value_ha = "left"
        if not passed and mean > 0.025:
            value_x = mean - 0.014
            value_ha = "right"
        ax.text(value_x, y, f"{mean:.3f}", ha=value_ha, va="center", fontsize=6.5, color=COLORS["ink"], fontweight="bold" if passed else "normal")
        tag_x = -0.078
        tag_color = COLORS["local"] if passed else COLORS["rolling"]
        ax.text(
            tag_x,
            y,
            "PASS" if passed else "FAIL",
            ha="left",
            va="center",
            fontsize=6.0,
            color=tag_color,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.92},
        )
        ax.text(0.228, y, f"{int(row['stations_improved_n'])}/5", ha="right", va="center", fontsize=6.3, color="#333333")
    ax.axvspan(0.05, 0.24, color=COLORS["soft_pass"], alpha=0.48, zorder=0)
    ax.axvline(0.05, color=COLORS["threshold"], linestyle="--", linewidth=1.0)
    ax.axvline(0, color=COLORS["threshold"], linewidth=0.8)
    ax.set_yticks(
        y_positions,
        [f"{key[0] * 15} min  {labels[key[1]]}" for key in order],
    )
    ax.set_xlim(-0.085, 0.255)
    ax.set_ylim(len(order) - 0.35, -0.68)
    ax.set_xlabel("Mean reduction in worst-group undercoverage")
    ax.set_title("Station-aware intervals separate comparative from absolute success", loc="left", fontweight="bold", fontsize=8.8, pad=12)
    ax.text(0.052, -0.58, "comparative gate", fontsize=6.4, ha="left", va="center", color=COLORS["threshold"])
    ax.text(
        0.228,
        -0.62,
        "stations\nimproved",
        fontsize=5.8,
        ha="right",
        va="center",
        color="#555555",
    )
    clean_axes(ax, "x")
    fig.text(
        0.30,
        0.035,
        "Filled circles mark comparative-gate passes; X marks failures. Horizontal segments are station-bootstrap intervals.",
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#555555",
    )
    fig.subplots_adjust(top=0.84, bottom=0.28, left=0.30, right=0.97)
    save(fig, figures, "fig21_station_aware_improvement")


def oracle_gap(tables: Path, figures: Path) -> None:
    gap = pd.read_csv(tables / "oracle_gap_summary.csv")
    risk = pd.read_csv(tables / "ramp_risk_diagnostics.csv")
    fig = plt.figure(figsize=(6.4, 3.48))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.06], wspace=0.30)
    ax_gap = fig.add_subplot(gs[0, 0])
    ax_risk = fig.add_subplot(gs[0, 1])

    decomp = (
        gap.assign(
            local_gain=lambda d: d["baseline_worst_undercoverage"] - d["local_worst_undercoverage"],
            local_to_oracle=lambda d: d["local_worst_undercoverage"] - d["oracle_worst_undercoverage"],
        )
        .groupby("horizon_steps", as_index=False)
        .agg(
            oracle=("oracle_worst_undercoverage", "median"),
            remaining=("local_to_oracle", "median"),
            repaired=("local_gain", "median"),
            baseline=("baseline_worst_undercoverage", "median"),
            local=("local_worst_undercoverage", "median"),
        )
    )
    parts = [
        ("oracle", "Oracle residual", COLORS["oracle"]),
        ("remaining", "Local-to-oracle gap", COLORS["risk"]),
        ("repaired", "Repaired by local CQR", COLORS["local"]),
    ]
    y_pos = np.arange(len(decomp))
    for yi, row in zip(y_pos, decomp.itertuples(index=False)):
        left = 0.0
        for col, label, color in parts:
            value = max(float(getattr(row, col)), 0.0)
            ax_gap.barh(yi, value, left=left, height=0.50, color=color, edgecolor="white", linewidth=0.9, label=label if yi == 0 else None)
            if value >= 0.055:
                ax_gap.text(left + value / 2, yi, f"{value:.2f}", ha="center", va="center", fontsize=6.7, color="white", fontweight="bold")
            left += value
        ax_gap.plot([float(row.local), float(row.local)], [yi - 0.29, yi + 0.29], color=COLORS["threshold"], linewidth=1.0)
        ax_gap.text(float(row.local), yi + 0.35, "local", ha="center", va="bottom", fontsize=6.0, color=COLORS["threshold"])
    ax_gap.set_yticks(y_pos, [f"{int(h) * 15} min" for h in decomp["horizon_steps"]])
    ax_gap.invert_yaxis()
    ax_gap.set_ylim(len(decomp) - 0.20, -1.02)
    ax_gap.set_xlim(0, max(decomp["baseline"]) + 0.08)
    ax_gap.set_xlabel("Median worst-group undercoverage")
    ax_gap.set_title("Failure decomposes into an oracle gap", loc="left", fontweight="bold", fontsize=8.5, pad=8)
    ax_gap.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.98), ncol=1, borderaxespad=0, handlelength=1.2, fontsize=6.4)
    clean_axes(ax_gap, "x")
    ax_gap.text(-0.10, 1.05, "A", transform=ax_gap.transAxes, fontsize=8.8, fontweight="bold", ha="left", va="bottom")

    condition_order = ["clean", "irradiance_outage", "weather_outage", "recent_power_outage", "combined_outage"]
    condition_labels = ["Clean", "Irrad.", "Weather", "Recent\npwr", "Combo"]
    agg = (
        risk[risk["condition"].isin(condition_order)]
        .groupby(["horizon_steps", "condition"], as_index=False)
        .agg(average_precision=("average_precision", "mean"), roc_auc=("roc_auc", "mean"))
    )
    heat = (
        agg.pivot(index="horizon_steps", columns="condition", values="average_precision")
        .loc[[1, 4], condition_order]
        .to_numpy()
    )
    auc = (
        agg.pivot(index="horizon_steps", columns="condition", values="roc_auc")
        .loc[[1, 4], condition_order]
        .to_numpy()
    )
    im = ax_risk.imshow(heat, cmap=HEATMAP, vmin=0.10, vmax=0.55, aspect="auto")
    ax_risk.set_xticks(np.arange(len(condition_order)), condition_labels)
    ax_risk.set_yticks([0, 1], ["15 min", "60 min"])
    ax_risk.set_title("Ramp identification fades under outages", loc="left", fontweight="bold", fontsize=8.5, pad=8)
    ax_risk.set_xlabel("Sensor condition")
    ax_risk.set_ylabel("Forecast horizon")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            color = "white" if heat[i, j] >= 0.32 else COLORS["ink"]
            ax_risk.text(
                j,
                i - 0.08,
                f"{heat[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=6.7,
                color=color,
                fontweight="bold" if j in {0, len(condition_order) - 1} else "normal",
            )
            ax_risk.text(
                j,
                i + 0.17,
                f"AUC {auc[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=4.8,
                color=color,
            )
    ax_risk.tick_params(length=0)
    ax_risk.tick_params(axis="x", labelsize=7.4)
    for spine in ax_risk.spines.values():
        spine.set_visible(False)
    ax_risk.set_xticks(np.arange(-0.5, len(condition_order), 1), minor=True)
    ax_risk.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax_risk.grid(which="minor", color="white", linewidth=1.2)
    ax_risk.tick_params(which="minor", bottom=False, left=False, labelbottom=False, labelleft=False)
    cbar = fig.colorbar(im, ax=ax_risk, fraction=0.046, pad=0.025)
    cbar.set_label("Average precision", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6.0, length=0)
    ax_risk.text(-0.10, 1.05, "B", transform=ax_risk.transAxes, fontsize=8.8, fontweight="bold", ha="left", va="bottom")
    save(fig, figures, "fig22_oracle_conditioning_gap")


def risk_degradation(tables: Path, figures: Path) -> None:
    directional = pd.read_csv(tables / "directional_ramp_metrics.csv")
    method_map = {
        "robust_condition_cqr": "Condition CQR",
        "robust_local_cqr": "Local CQR",
    }
    sub = directional[
        directional["method"].isin(method_map)
        & directional["condition"].isin(["clean", "combined_outage", "weather_outage", "irradiance_outage"])
    ].copy()
    agg = (
        sub.groupby(["horizon_steps", "method", "ramp_direction"], as_index=False)
        .agg(undercoverage=("undercoverage_error", "mean"))
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.85), sharey=True)
    directions = ["down", "up"]
    x = np.arange(len(directions))
    for ax, horizon in zip(axes, [1, 4]):
        h = agg[agg["horizon_steps"] == horizon]
        for offset, method, color in [(-0.17, "robust_condition_cqr", COLORS["baseline"]), (0.17, "robust_local_cqr", COLORS["local"])]:
            vals = [float(h[(h["method"] == method) & (h["ramp_direction"] == d)]["undercoverage"].iloc[0]) for d in directions]
            ax.bar(x + offset, vals, width=0.30, color=color, alpha=0.95, label=method_map[method])
            for xi, val in zip(x + offset, vals):
                label_y = val + 0.018
                if abs(val - 0.10) < 0.025:
                    label_y = val + 0.030
                ax.text(
                    xi,
                    label_y,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#333333",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
                )
        ax.axhline(0.10, color=COLORS["threshold"], linestyle="--", linewidth=0.9)
        ax.set_xticks(x, ["Down-ramp", "Up-ramp"])
        ax.set_title(f"{horizon*15}-min horizon", loc="left", fontweight="bold")
        ax.set_ylim(0, 0.46)
        ax.set_ylabel("Mean ramp undercoverage")
        clean_axes(ax, "y")
    axes[1].set_ylabel("")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, figures, "fig23_ramp_direction_reliability")


def audit(figures: Path, tables: Path) -> None:
    records = []
    expected = [
        "fig20_confirmation_overview.png",
        "fig20_station_worst_undercoverage.png",
        "fig21_station_aware_improvement.png",
        "fig22_oracle_conditioning_gap.png",
        "fig23_ramp_direction_reliability.png",
    ]
    for name in expected:
        png = figures / name
        pdf = png.with_suffix(".pdf")
        svg = png.with_suffix(".svg")
        records.append(
            {
                "figure": png.name,
                "pdf_vector": pdf.exists(),
                "svg_vector": svg.exists(),
                "png_bytes": png.stat().st_size,
                "status": "pass" if pdf.exists() and svg.exists() and png.stat().st_size > 10000 else "review",
            }
        )
    payload = {"figures": records, "all_pass": bool(records) and all(r["status"] == "pass" for r in records)}
    (tables / "multisite_visualization_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multisite_confirmatory.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tables = ROOT / cfg["outputs"]["tables"]
    figures = ROOT / cfg["outputs"]["figures"]
    apply_publication_style()
    apply_lnai_visual_style()
    overview_protocol(tables, figures)
    station_undercoverage(tables, figures)
    candidate_gates(tables, figures)
    oracle_gap(tables, figures)
    risk_degradation(tables, figures)
    audit(figures, tables)


if __name__ == "__main__":
    main()
