from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROOT = ROOT / "outputs" / "tables"

# Okabe--Ito-derived palette. Method identity is also encoded by marker shape,
# fill, and line style so that the figures remain legible in grayscale.
COLORS = {
    "ink": "#202124",
    "muted": "#62666D",
    "grid": "#D9DDE3",
    "baseline": "#666666",
    "local": "#0072B2",
    "risk": "#D55E00",
    "rolling": "#009E73",
    "valid": "#0072B2",
    "excluded": "#D55E00",
    "soft_blue": "#E7F1F7",
    "soft_orange": "#FCECE5",
    "soft_gray": "#F2F3F5",
}

METHODS = {
    "robust_condition_cqr": {
        "label": "Condition CQR",
        "color": COLORS["baseline"],
        "marker": "o",
        "linestyle": "--",
        "filled": False,
    },
    "robust_local_cqr": {
        "label": "Descriptor CQR",
        "color": COLORS["local"],
        "marker": "D",
        "linestyle": "-",
        "filled": True,
    },
    "robust_risk_mondrian": {
        "label": "Risk-Mondrian CQR",
        "color": COLORS["risk"],
        "marker": "^",
        "linestyle": "-.",
        "filled": True,
    },
    "robust_rolling_cqr": {
        "label": "Rolling CQR",
        "color": COLORS["rolling"],
        "marker": "s",
        "linestyle": ":",
        "filled": True,
    },
}

CORE_FIGURES = (
    "fig01_protocol_station_screening",
    "fig02_station_worst_undercoverage",
    "fig03_width_score_tradeoff",
)
ALT_TEXT_FILE = "FIGURE_ALT_TEXT.md"
MIN_SOURCE_FIGURE_TEXT_PT = 7.5
MIN_FINAL_FIGURE_TEXT_PT = 6.0
TARGET_TEXT_WIDTH_BP = 122.0 / 25.4 * 72.0
FIGURE_PLACEMENT_FRACTIONS = {
    CORE_FIGURES[0]: 0.90,
    CORE_FIGURES[1]: 0.88,
    CORE_FIGURES[2]: 0.88,
}
SOURCE_TEXT_MINIMA: dict[str, float] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required figure input is unavailable: {path}")
    return path


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def finite_numeric(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite values in {columns}")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.2,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            # Core figures are evaluated at 0.88--0.90 of the target text width.
            # A 7.5-pt source minimum retains headroom above a 6-pt
            # lower bound after measured tight-bbox scaling.
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)


def display_exclusion_reason(reason: str, daylight_threshold: float) -> str:
    """Translate audit codes into concise wording suitable for a figure."""
    schema_prefix = "schema validation failed: Missing required columns:"
    if reason.startswith(schema_prefix):
        encoded_fields = reason.removeprefix(schema_prefix).strip()
        try:
            fields = ast.literal_eval(encoded_fields)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Cannot parse excluded-field record: {reason}") from exc
        if not isinstance(fields, (list, tuple)) or not fields:
            raise ValueError(f"Invalid excluded-field record: {reason}")
        clean_fields = [str(field).strip() for field in fields]
        if any(not field for field in clean_fields):
            raise ValueError(f"Invalid excluded-field name: {reason}")
        if len(clean_fields) == 1:
            return f"missing required {clean_fields[0]} field"
        return f"missing required fields: {', '.join(clean_fields)}"
    if reason == "daylight target missingness above threshold":
        return (
            "daylight target missingness exceeded the prespecified "
            f"{daylight_threshold:.0%} threshold"
        )
    raise ValueError(f"No display wording defined for exclusion reason: {reason}")


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["ink"],
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / name
    fig.canvas.draw()
    text_sizes = [
        float(artist.get_fontsize())
        for artist in fig.findobj(match=Text)
        if str(artist.get_text()).strip()
    ]
    if not text_sizes:
        raise ValueError(f"Figure contains no auditable text: {name}")
    minimum_source_text = min(text_sizes)
    if minimum_source_text + 1e-9 < MIN_SOURCE_FIGURE_TEXT_PT:
        raise ValueError(
            f"Figure source text is too small in {name}: "
            f"{minimum_source_text:.3f} pt < {MIN_SOURCE_FIGURE_TEXT_PT:.3f} pt"
        )
    SOURCE_TEXT_MINIMA[name] = minimum_source_text
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={
            "Title": title,
            "Subject": "Solar-ramp reliability analysis figure",
            "Keywords": "solar forecasting, prediction intervals, reliability",
            "Creator": "scripts/render_figures.py",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={"Software": "scripts/render_figures.py"},
    )
    plt.close(fig)


def validated_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[Path]]:
    tables = ROOT / cfg["outputs"]["tables"]
    audit_path = require_file(tables / "station_data_audit.csv")
    effects_path = require_file(tables / "station_level_effects.csv")
    verdict_path = require_file(tables / "multisite_verdict.json")
    tradeoff_path = require_file(CAMERA_ROOT / "method_tradeoffs.csv")

    audit = pd.read_csv(audit_path)
    effects = pd.read_csv(effects_path)
    tradeoffs = pd.read_csv(tradeoff_path)
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))

    require_columns(
        audit,
        {"site_id", "nominal_capacity_mw", "status", "exclusion_reason"},
        audit_path,
    )
    require_columns(
        effects,
        {
            "station",
            "horizon_steps",
            "method",
            "baseline_worst_undercoverage",
            "method_worst_undercoverage",
        },
        effects_path,
    )
    require_columns(
        tradeoffs,
        {
            "horizon_steps",
            "method",
            "stratum",
            "stations",
            "aggregation_unit",
            "mean_normalized_width",
            "mean_normalized_interval_score",
            "normalized_width_change_vs_baseline",
            "normalized_interval_score_change_vs_baseline",
            "mean_station_prediction_availability",
        },
        tradeoff_path,
    )

    development = set(map(str, cfg["project"].get("development_sites", [])))
    if not development:
        raise ValueError("The development-site set is unavailable; protocol figure cannot be inferred")
    primary_unit = str(
        cfg.get("metrics", {}).get("statistical_replication_units", {}).get("primary", "")
    )
    if primary_unit != "station":
        raise ValueError("The frozen primary statistical replication unit is not station")
    decision_rules = cfg.get("decision_rules", {})
    if not isinstance(decision_rules.get("comparative_gate"), dict) or not isinstance(
        decision_rules.get("absolute_reliability_gate"), dict
    ):
        raise ValueError("The frozen comparative and absolute decision gates are unavailable")
    external = audit[~audit["site_id"].astype(str).isin(development)].copy()
    if external.empty:
        raise ValueError("No external station audit rows are available")
    if external["site_id"].astype(str).duplicated().any():
        raise ValueError("External station audit contains duplicate station IDs")
    preferred_external = sorted(map(str, cfg["project"].get("confirmatory_sites_preferred", [])))
    observed_external = sorted(external["site_id"].astype(str))
    if observed_external != preferred_external:
        raise ValueError(
            "Station audit and frozen external-station list disagree: "
            f"audit={observed_external}, config={preferred_external}"
        )
    finite_numeric(external, ["nominal_capacity_mw"], audit_path)
    if bool((external["nominal_capacity_mw"] <= 0).any()):
        raise ValueError("Station capacities must be positive")
    allowed_status = {"valid", "excluded"}
    observed_status = set(external["status"].astype(str))
    if not observed_status.issubset(allowed_status):
        raise ValueError(f"Unexpected station audit status values: {sorted(observed_status - allowed_status)}")
    excluded = external[external["status"].astype(str) == "excluded"]
    if excluded["exclusion_reason"].fillna("").str.strip().eq("").any():
        raise ValueError("An excluded station lacks a documented exclusion reason")

    valid_external = sorted(
        external.loc[external["status"].astype(str) == "valid", "site_id"].astype(str)
    )
    verdict_stations = sorted(map(str, verdict.get("valid_confirmatory_stations", [])))
    if valid_external != verdict_stations:
        raise ValueError(
            "Station audit and confirmatory verdict disagree: "
            f"audit={valid_external}, verdict={verdict_stations}"
        )
    fold_count = len(cfg["data"].get("site1_calendar_folds", []))
    horizon_count = len(cfg["forecast"].get("horizons_steps", []))
    expected_chunks = len(valid_external) * fold_count * horizon_count
    if int(verdict.get("prediction_chunks", -1)) != expected_chunks:
        raise ValueError(
            f"Verdict reports {verdict.get('prediction_chunks')} chunks; expected {expected_chunks}"
        )

    local = effects[effects["method"].astype(str) == "robust_local_cqr"].copy()
    expected_cells = {
        (station, int(horizon))
        for station in valid_external
        for horizon in cfg["forecast"]["horizons_steps"]
    }
    observed_cells = {
        (str(row.station), int(row.horizon_steps)) for row in local.itertuples(index=False)
    }
    if observed_cells != expected_cells or len(local) != len(expected_cells):
        raise ValueError("Station-level local-CQR effects do not form the exact station x horizon grid")
    finite_numeric(
        local,
        ["baseline_worst_undercoverage", "method_worst_undercoverage"],
        effects_path,
    )
    reliability_values = local[
        ["baseline_worst_undercoverage", "method_worst_undercoverage"]
    ].to_numpy(dtype=float)
    if bool(((reliability_values < 0.0) | (reliability_values > 1.0)).any()):
        raise ValueError("Worst-group undercoverage values must lie in [0, 1]")

    method_order = list(METHODS)
    trade = tradeoffs[
        (tradeoffs["stratum"].astype(str) == "all")
        & tradeoffs["method"].astype(str).isin(method_order)
    ].copy()
    expected_trade_cells = {
        (int(horizon), method)
        for horizon in cfg["forecast"]["horizons_steps"]
        for method in method_order
    }
    observed_trade_cells = {
        (int(row.horizon_steps), str(row.method)) for row in trade.itertuples(index=False)
    }
    if observed_trade_cells != expected_trade_cells or len(trade) != len(expected_trade_cells):
        raise ValueError("Tradeoff input does not form the exact horizon x method grid")
    finite_numeric(
        trade,
        [
            "stations",
            "mean_normalized_width",
            "mean_normalized_interval_score",
            "normalized_width_change_vs_baseline",
            "normalized_interval_score_change_vs_baseline",
            "mean_station_prediction_availability",
        ],
        tradeoff_path,
    )
    if not (trade["stations"].astype(int) == len(valid_external)).all():
        raise ValueError("Tradeoff rows do not all use the full valid-station set")
    if not (trade["aggregation_unit"].astype(str) == "station").all():
        raise ValueError("Tradeoff rows are not station-first aggregates")
    if not np.allclose(trade["mean_station_prediction_availability"], 1.0):
        raise ValueError("A core comparison method has incomplete prediction availability")

    for horizon in map(int, cfg["forecast"]["horizons_steps"]):
        horizon_rows = trade[trade["horizon_steps"].astype(int) == horizon].set_index("method")
        baseline = horizon_rows.loc["robust_condition_cqr"]
        if not np.allclose(
            [
                float(baseline["normalized_width_change_vs_baseline"]),
                float(baseline["normalized_interval_score_change_vs_baseline"]),
            ],
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Baseline tradeoff deltas are not zero at horizon {horizon}")
        for method, row in horizon_rows.iterrows():
            expected_width_delta = (
                float(row["mean_normalized_width"])
                - float(baseline["mean_normalized_width"])
            )
            expected_score_delta = (
                float(row["mean_normalized_interval_score"])
                - float(baseline["mean_normalized_interval_score"])
            )
            if not np.isclose(
                float(row["normalized_width_change_vs_baseline"]),
                expected_width_delta,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Width delta is inconsistent with source means for {method}, horizon {horizon}"
                )
            if not np.isclose(
                float(row["normalized_interval_score_change_vs_baseline"]),
                expected_score_delta,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Interval-score delta is inconsistent with source means for {method}, horizon {horizon}"
                )

    return external, local, trade, verdict, [audit_path, effects_path, tradeoff_path, verdict_path]


def protocol_station_screening(
    external: pd.DataFrame,
    verdict: dict,
    cfg: dict,
    output_dir: Path,
) -> None:
    valid = external[external["status"].astype(str) == "valid"].copy()
    excluded = external[external["status"].astype(str) == "excluded"].copy()
    folds = len(cfg["data"]["site1_calendar_folds"])
    horizons = len(cfg["forecast"]["horizons_steps"])
    chunks = int(verdict["prediction_chunks"])

    fig = plt.figure(figsize=(4.78, 2.78))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.22], hspace=0.28)
    ax_flow = fig.add_subplot(grid[0])
    ax_station = fig.add_subplot(grid[1])

    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    steps = [
        ("Develop", f"{', '.join(map(str, cfg['project']['development_sites']))}\nonly"),
        ("Pre-specify", "splits,\nconditions,\nseeds"),
        ("Screen", f"{len(external)} external\nstations"),
        (
            "Confirm",
            f"${len(valid)} \\times {folds} \\times {horizons}$\n{chunks} units",
        ),
        ("Assess", "station-level\ncriteria"),
    ]
    # Keep the rounded boxes inside the axes so their borders are not clipped in
    # the tightly cropped PDF/PNG exports.
    box_pad = 0.008
    xs = [0.015, 0.205, 0.40, 0.595, 0.79]
    widths = [0.16, 0.165, 0.165, 0.165, 0.195]
    if any(x - box_pad < 0 or x + width + box_pad > 1 for x, width in zip(xs, widths)):
        raise ValueError("Protocol flow boxes must remain inside the axes after padding")
    fills = [
        COLORS["soft_gray"],
        COLORS["soft_blue"],
        COLORS["soft_orange"],
        COLORS["soft_blue"],
        COLORS["soft_gray"],
    ]
    edges = [
        COLORS["muted"],
        COLORS["valid"],
        COLORS["excluded"],
        COLORS["valid"],
        COLORS["ink"],
    ]
    for idx, ((title, body), x, width, fill, edge) in enumerate(zip(steps, xs, widths, fills, edges)):
        box = FancyBboxPatch(
            (x, 0.22),
            width,
            0.58,
            boxstyle=f"round,pad={box_pad},rounding_size=0.025",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=fill,
        )
        ax_flow.add_patch(box)
        ax_flow.text(x + width / 2, 0.65, title, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax_flow.text(
            x + width / 2,
            0.42,
            body,
            ha="center",
            va="center",
            fontsize=7.5,
            linespacing=1.0,
            color=COLORS["muted"],
        )
        if idx < len(steps) - 1:
            ax_flow.annotate(
                "",
                xy=(xs[idx + 1] - box_pad, 0.51),
                xytext=(x + width + box_pad, 0.51),
                arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": COLORS["muted"], "mutation_scale": 8},
            )
    ax_flow.text(
        0,
        0.99,
        "A  Prespecified confirmation protocol",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    station = external.sort_values(
        ["status", "nominal_capacity_mw", "site_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    y = np.arange(len(station))
    is_valid = station["status"].astype(str).eq("valid")
    ax_station.set_title("B  Pre-outcome station screening", loc="left", pad=5)
    ax_station.hlines(y, 0, station["nominal_capacity_mw"], color=COLORS["grid"], linewidth=1.0, zorder=1)
    ax_station.scatter(
        station.loc[is_valid, "nominal_capacity_mw"],
        y[is_valid.to_numpy()],
        marker="o",
        s=38,
        facecolor=COLORS["valid"],
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
        label=rf"Included ($n={len(valid)}$)",
    )
    ax_station.scatter(
        station.loc[~is_valid, "nominal_capacity_mw"],
        y[(~is_valid).to_numpy()],
        marker="X",
        s=44,
        facecolor="white",
        edgecolor=COLORS["excluded"],
        linewidth=1.1,
        zorder=4,
        label=rf"Excluded ($n={len(excluded)}$)",
    )
    for yi, row in enumerate(station.itertuples(index=False)):
        ax_station.text(
            float(row.nominal_capacity_mw) + 2.2,
            yi,
            f"{float(row.nominal_capacity_mw):.0f}",
            ha="left",
            va="center",
            fontsize=7.5,
            color=COLORS["ink"],
        )
    max_capacity = float(station["nominal_capacity_mw"].max())
    ax_station.set_xlim(0, max_capacity * 1.16)
    ax_station.set_yticks(y, station["site_id"].astype(str))
    ax_station.set_ylim(len(station) - 0.45, -0.55)
    ax_station.set_xlabel("Nominal capacity (MW)")
    ax_station.legend(
        loc="lower right",
        frameon=False,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0.2,
    )
    clean_axes(ax_station, "x")
    # Exclusion reasons remain in the adjacent screened-station table and the
    # accessibility artifact, where they inherit document typography, rather
    # than in an undersized artwork footnote.
    fig.subplots_adjust(left=0.105, right=0.985, top=0.985, bottom=0.16)
    save_figure(
        fig,
        output_dir,
        CORE_FIGURES[0],
        "Prespecified confirmation protocol and pre-outcome station screening",
    )


def station_worst_undercoverage(local: pd.DataFrame, cfg: dict, output_dir: Path) -> None:
    stations = sorted(local["station"].astype(str).unique())
    horizons = list(map(int, cfg["forecast"]["horizons_steps"]))
    sampling_minutes = int(cfg["data"]["sampling_minutes"])
    tolerance = float(
        cfg["decision_rules"]["absolute_reliability_gate"]
        ["max_worst_group_undercoverage_each_horizon"]
    )
    if horizons != [1, 4]:
        raise ValueError(f"Expected the frozen horizons [1, 4], found {horizons}")
    if sampling_minutes <= 0:
        raise ValueError("Sampling interval must be positive")
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
        raise ValueError("Absolute reliability tolerance must lie in [0, 1]")

    fig, axes = plt.subplots(1, 2, figsize=(4.78, 2.66), sharey=True)
    y = np.arange(len(stations))
    for panel, (ax, horizon) in enumerate(zip(axes, horizons)):
        subset = (
            local[local["horizon_steps"].astype(int) == horizon]
            .assign(station=lambda frame: frame["station"].astype(str))
            .set_index("station")
            .loc[stations]
        )
        baseline = subset["baseline_worst_undercoverage"].to_numpy(dtype=float)
        candidate = subset["method_worst_undercoverage"].to_numpy(dtype=float)
        for yi, base, method in zip(y, baseline, candidate):
            ax.annotate(
                "",
                xy=(method, yi),
                xytext=(base, yi),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 1.0,
                    "color": COLORS["local"],
                    "mutation_scale": 7,
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
                zorder=2,
            )
        ax.scatter(
            baseline,
            y,
            marker="o",
            s=34,
            facecolor="white",
            edgecolor=COLORS["baseline"],
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            candidate,
            y,
            marker="D",
            s=31,
            facecolor=COLORS["local"],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax.axvline(tolerance, color=COLORS["ink"], linestyle=(0, (3, 2)), linewidth=0.85)
        ax.text(
            tolerance + 0.005,
            -0.60,
            rf"$\delta={tolerance:.2f}$",
            ha="left",
            va="center",
            fontsize=7.5,
            color=COLORS["ink"],
        )
        ax.set_xlim(0, 0.59)
        ax.set_ylim(len(stations) - 0.45, -0.72)
        ax.set_xlabel("Worst-group undercoverage")
        ax.set_title(f"{horizon * sampling_minutes}-min horizon", loc="left", pad=5)
        clean_axes(ax, "x")
        panel_label(ax, chr(ord("A") + panel), x=-0.12 if panel == 0 else -0.07)
    axes[0].set_yticks(y, stations)
    axes[0].set_ylabel("Confirmatory station")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["baseline"],
            markeredgewidth=1.0,
            label=METHODS["robust_condition_cqr"]["label"],
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=COLORS["local"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=METHODS["robust_local_cqr"]["label"],
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.93),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.suptitle(
        "Station-level worst-group undercoverage",
        x=0.012,
        y=0.995,
        ha="left",
        fontsize=8.4,
        fontweight="semibold",
    )
    # Direction and tolerance are defined in the figure specification.
    fig.subplots_adjust(left=0.13, right=0.99, top=0.73, bottom=0.19, wspace=0.13)
    save_figure(
        fig,
        output_dir,
        CORE_FIGURES[1],
        "Station-level worst-group undercoverage comparison",
    )


def width_score_tradeoff(trade: pd.DataFrame, cfg: dict, output_dir: Path) -> None:
    horizons = list(map(int, cfg["forecast"]["horizons_steps"]))
    sampling_minutes = int(cfg["data"]["sampling_minutes"])
    candidate_methods = ["robust_local_cqr", "robust_risk_mondrian", "robust_rolling_cqr"]
    baseline_method = "robust_condition_cqr"

    fig, axes = plt.subplots(1, 2, figsize=(4.78, 2.62), sharex=True, sharey=True)
    all_dx = trade["normalized_width_change_vs_baseline"].to_numpy(dtype=float)
    all_dy = trade["normalized_interval_score_change_vs_baseline"].to_numpy(dtype=float)
    x_limit = max(0.034, math.ceil(float(np.max(np.abs(all_dx))) * 1000) / 1000 + 0.004)
    y_lower = min(-0.056, math.floor(float(np.min(all_dy)) * 1000) / 1000 - 0.005)
    y_upper = max(0.012, math.ceil(float(np.max(all_dy)) * 1000) / 1000 + 0.004)

    for panel, (ax, horizon) in enumerate(zip(axes, horizons)):
        subset = (
            trade[trade["horizon_steps"].astype(int) == horizon]
            .assign(method=lambda frame: frame["method"].astype(str))
            .set_index("method")
        )
        ax.axhline(0, color=COLORS["grid"], linewidth=0.7)
        ax.axvline(0, color=COLORS["grid"], linewidth=0.7)
        for method in candidate_methods:
            row = subset.loc[method]
            dx = float(row["normalized_width_change_vs_baseline"])
            dy = float(row["normalized_interval_score_change_vs_baseline"])
            style = METHODS[method]
            ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(0, 0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 1.05,
                    "linestyle": style["linestyle"],
                    "color": style["color"],
                    "mutation_scale": 7,
                    "shrinkA": 4,
                    "shrinkB": 4,
                },
                zorder=2,
            )
            ax.scatter(
                [dx],
                [dy],
                marker=style["marker"],
                s=36,
                color=style["color"],
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
        ax.scatter(
            [0],
            [0],
            marker=METHODS[baseline_method]["marker"],
            s=34,
            facecolor="white",
            edgecolor=COLORS["baseline"],
            linewidth=1.0,
            zorder=4,
        )
        ax.set_xlim(-0.004, x_limit)
        ax.set_ylim(y_lower, y_upper)
        ax.set_xlabel(r"$\Delta$ normalized width")
        ax.set_title(f"{horizon * sampling_minutes}-min horizon", loc="left", pad=5)
        clean_axes(ax, None)
        ax.grid(color=COLORS["grid"], linewidth=0.4, alpha=0.55)
        panel_label(ax, chr(ord("A") + panel), x=-0.14 if panel == 0 else -0.08)
    axes[0].set_ylabel(r"$\Delta$ normalized interval score")
    handles = []
    for method in [baseline_method, *candidate_methods]:
        style = METHODS[method]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color=style["color"] if method != baseline_method else "none",
                linestyle="none",
                markerfacecolor=style["color"] if style["filled"] else "white",
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                label=style["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.89),
        ncol=4,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.65,
    )
    fig.suptitle(
        "Station-first width–score trade-off relative to condition CQR",
        x=0.012,
        y=0.995,
        ha="left",
        fontsize=8.4,
        fontweight="semibold",
    )
    # Delta direction and availability are defined in the formal caption.
    fig.subplots_adjust(left=0.13, right=0.99, top=0.70, bottom=0.19, wspace=0.17)
    save_figure(
        fig,
        output_dir,
        CORE_FIGURES[2],
        "Station-first normalized-width and interval-score tradeoff",
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def write_accessibility_artifact(
    external: pd.DataFrame,
    local: pd.DataFrame,
    trade: pd.DataFrame,
    verdict: dict,
    cfg: dict,
    output_dir: Path,
) -> tuple[Path, dict]:
    """Write deterministic alt text and an exact source-to-mark provenance ledger."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / ALT_TEXT_FILE
    sampling_minutes = int(cfg["data"]["sampling_minutes"])
    horizons = list(map(int, cfg["forecast"]["horizons_steps"]))
    tolerance = float(
        cfg["decision_rules"]["absolute_reliability_gate"]
        ["max_worst_group_undercoverage_each_horizon"]
    )
    daylight_threshold = float(
        cfg["data"]["integrity_exclusion_rules"]["max_target_missing_rate_daylight"]
    )
    development = list(map(str, cfg["project"]["development_sites"]))

    station_rows = external.sort_values("site_id").copy()
    valid_rows = station_rows[station_rows["status"].astype(str) == "valid"]
    excluded_rows = station_rows[station_rows["status"].astype(str) == "excluded"]
    folds = len(cfg["data"]["site1_calendar_folds"])
    chunks = int(verdict["prediction_chunks"])

    included_summary = ", ".join(
        f"{row.site_id} ({float(row.nominal_capacity_mw):.0f} MW)"
        for row in valid_rows.itertuples(index=False)
    )
    excluded_summary = "; ".join(
        f"{row.site_id} ({float(row.nominal_capacity_mw):.0f} MW; "
        f"{display_exclusion_reason(str(row.exclusion_reason), daylight_threshold)})"
        for row in excluded_rows.itertuples(index=False)
    )

    reliability_rows = local.assign(
        station=lambda frame: frame["station"].astype(str),
        horizon_steps=lambda frame: frame["horizon_steps"].astype(int),
    ).sort_values(["horizon_steps", "station"])
    improved_count = int(
        (
            reliability_rows["method_worst_undercoverage"]
            < reliability_rows["baseline_worst_undercoverage"]
        ).sum()
    )
    at_tolerance_count = int(
        (reliability_rows["method_worst_undercoverage"] <= tolerance).sum()
    )
    if at_tolerance_count == 0:
        tolerance_summary = "No Descriptor CQR estimate satisfies it."
    else:
        tolerance_summary = (
            f"{at_tolerance_count} Descriptor CQR estimates satisfy it."
        )
    reliability_ranges = []
    for horizon in horizons:
        rows = reliability_rows[reliability_rows["horizon_steps"] == horizon]
        reliability_ranges.append(
            f"{horizon * sampling_minutes} minutes: Descriptor CQR "
            f"{rows['method_worst_undercoverage'].min():.3f} to {rows['method_worst_undercoverage'].max():.3f} "
            f"versus condition CQR {rows['baseline_worst_undercoverage'].min():.3f} to "
            f"{rows['baseline_worst_undercoverage'].max():.3f}"
        )

    ordered_methods = list(METHODS)
    trade_rows = trade.assign(
        method=lambda frame: frame["method"].astype(str),
        horizon_steps=lambda frame: frame["horizon_steps"].astype(int),
    )
    trade_rows["method_order"] = pd.Categorical(
        trade_rows["method"], categories=ordered_methods, ordered=True
    )
    trade_rows = trade_rows.sort_values(["horizon_steps", "method_order"])
    candidate_rows = trade_rows[trade_rows["method"] != "robust_condition_cqr"]
    wider_count = int((candidate_rows["normalized_width_change_vs_baseline"] > 0).sum())
    better_score_count = int(
        (candidate_rows["normalized_interval_score_change_vs_baseline"] < 0).sum()
    )
    if wider_count == len(candidate_rows) and better_score_count == len(candidate_rows):
        candidate_direction_summary = (
            f"All six candidate estimates have positive width differences "
            "and negative interval-score differences relative to condition CQR."
        )
    else:
        candidate_direction_summary = (
            f"Of the {len(candidate_rows)} candidate estimates, {wider_count} have positive "
            f"width differences and {better_score_count} have negative interval-score differences."
        )
    trade_summary_parts = []
    for row in candidate_rows.itertuples(index=False):
        trade_summary_parts.append(
            f"{int(row.horizon_steps) * sampling_minutes}-minute "
            f"{METHODS[str(row.method)]['label']}: "
            f"width {float(row.normalized_width_change_vs_baseline):+.3f}, "
            f"score {float(row.normalized_interval_score_change_vs_baseline):+.3f}"
        )
    baseline_summary_parts = []
    for row in trade_rows[
        trade_rows["method"] == "robust_condition_cqr"
    ].itertuples(index=False):
        baseline_summary_parts.append(
            f"{int(row.horizon_steps) * sampling_minutes} minutes: normalized width "
            f"{float(row.mean_normalized_width):.3f} and interval score "
            f"{float(row.mean_normalized_interval_score):.3f}"
        )

    lines = [
        "# Figure alt text and evidence mapping",
        "",
        "This deterministic accessibility artifact covers exactly the three corrected core figures. "
        "Alt-text values are rounded to three decimals for readability; plotting and validation use "
        "the full-precision source values. In the companion GitHub code archive, repository-relative "
        "source paths and exact input hashes are recorded in "
        "`outputs/tables/visualization_audit.json`.",
        "",
        "## Figure 1 - Prespecified confirmation protocol and pre-outcome station screening",
        "",
        "**Alt text.** Five protocol stages run from development on "
        f"{', '.join(development)} only, through prespecified splits, sensor conditions, and seeds, "
        "to screening "
        f"{len(station_rows)} external station records and confirming {len(valid_rows)} valid stations "
        f"over {folds} folds and {len(horizons)} horizons ({chunks} station-fold-horizon evaluation units), "
        "followed by prespecified station-level criteria. "
        f"The included stations and nominal capacities are {included_summary}. "
        f"The excluded records are {excluded_summary}. Station codes are dataset identifiers, not "
        "geographic locations.",
        "",
        "**Source and claim mapping.** Protocol settings come from "
        "`configs/experiment.yaml` (`project.development_sites`, `project.seed`, "
        "`data.site1_calendar_folds`, `conditions`, `forecast.horizons_steps`, "
        "`metrics.statistical_replication_units.primary`, and `decision_rules`). Screening marks, "
        "capacities, statuses, and recorded reasons come directly from "
        "`outputs/tables/station_data_audit.csv` (`site_id`, "
        "`nominal_capacity_mw`, `status`, `exclusion_reason`). The included-station set and total of "
        "40 evaluations are cross-checked against "
        "`outputs/tables/multisite_verdict.json`.",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Station", "Plotted capacity (MW)", "Plotted status", "Displayed exclusion criterion"],
            [
                [
                    str(row.site_id),
                    f"{float(row.nominal_capacity_mw):.0f}",
                    "included" if str(row.status) == "valid" else "excluded",
                    display_exclusion_reason(str(row.exclusion_reason), daylight_threshold)
                    if str(row.status) == "excluded"
                    else "-",
                ]
                for row in station_rows.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Figure 2 - Station-level worst-group undercoverage",
            "",
            "**Alt text.** Two panels compare condition CQR (open circles) with "
            "Descriptor CQR (filled diamonds) at 15- and 60-minute horizons for "
            f"five confirmatory stations. Arrows run from condition CQR to the Descriptor CQR "
            f"value, which is lower in {improved_count} of {len(reliability_rows)} station-horizon "
            f"comparisons. {'; '.join(reliability_ranges)}. A dashed line marks the prespecified "
            f"undercoverage tolerance of {tolerance:.2f}. {tolerance_summary}",
            "",
            "**Source and claim mapping.** Every circle, diamond, and arrow endpoint comes from "
            "`outputs/tables/station_level_effects.csv`, filtered to "
            "`method=robust_local_cqr`, using `station`, `horizon_steps`, "
            "`baseline_worst_undercoverage`, and `method_worst_undercoverage`. Horizon labels use "
            "`data.sampling_minutes` and `forecast.horizons_steps`; the dashed tolerance uses "
            "`decision_rules.absolute_reliability_gate.max_worst_group_undercoverage_each_horizon` "
            "from `configs/experiment.yaml`.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Horizon", "Station", "Condition CQR", "Descriptor CQR"],
            [
                [
                    f"{int(row.horizon_steps) * sampling_minutes} min",
                    str(row.station),
                    f"{float(row.baseline_worst_undercoverage):.3f}",
                    f"{float(row.method_worst_undercoverage):.3f}",
                ]
                for row in reliability_rows.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Figure 3 - Station-first width–score trade-off",
            "",
            "**Alt text.** Two panels plot change in normalized interval width horizontally and "
            "change in normalized interval score vertically, relative to condition CQR at 15 and "
            "60 minutes. The two condition-CQR baselines are at the origin; their absolute values "
            f"are {'; '.join(baseline_summary_parts)}. {candidate_direction_summary} "
            f"The plotted candidate coordinates are {'; '.join(trade_summary_parts)}. "
            "Every displayed method and horizon aggregates five stations and has 100% mean "
            "station-level prediction availability.",
            "",
            "**Source and claim mapping.** Every point, baseline reference value, and availability claim "
            "comes from `outputs/tables/method_tradeoffs.csv`, filtered to `stratum=all` and "
            "the four displayed methods. The mapped fields are `horizon_steps`, `method`, `stations`, "
            "`aggregation_unit`, `mean_normalized_width`, `mean_normalized_interval_score`, "
            "`normalized_width_change_vs_baseline`, "
            "`normalized_interval_score_change_vs_baseline`, and "
            "`mean_station_prediction_availability`. Each delta is recomputed from its method and "
            "condition-CQR source means before rendering.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Horizon",
                "Method",
                "Mean normalized width",
                "Mean normalized interval score",
                "Width change",
                "Interval-score change",
                "Availability",
            ],
            [
                [
                    f"{int(row.horizon_steps) * sampling_minutes} min",
                    METHODS[str(row.method)]["label"],
                    f"{float(row.mean_normalized_width):.3f}",
                    f"{float(row.mean_normalized_interval_score):.3f}",
                    f"{float(row.normalized_width_change_vs_baseline):+.3f}",
                    f"{float(row.normalized_interval_score_change_vs_baseline):+.3f}",
                    f"{100.0 * float(row.mean_station_prediction_availability):.0f}%",
                ]
                for row in trade_rows.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Validation criteria",
            "",
            "Validation fails if the frozen external-station list does not match the audit; any capacity, "
            "status, or exclusion reason is missing; the audit and verdict station sets differ; the "
            "station-by-horizon or method-by-horizon grids are incomplete; a plotted metric is "
            "non-finite; an undercoverage value lies outside [0, 1]; a tradeoff row is not a "
            "five-station station-first aggregate with complete availability; a condition-CQR delta "
            "is nonzero; or a reported tradeoff delta differs from the corresponding difference of "
            "full-precision source means. A violation prevents figure generation.",
            "",
        ]
    )
    artifact.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    claim_validation = {
        CORE_FIGURES[0]: {
            "protocol_stages_checked": 5,
            "external_station_rows": len(station_rows),
            "capacity_status_marks_checked": len(station_rows),
            "exclusion_reasons_checked": len(excluded_rows),
            "valid_station_set_matches_verdict": True,
            "chunk_count_matches_station_fold_horizon_product": True,
        },
        CORE_FIGURES[1]: {
            "station_horizon_rows": len(reliability_rows),
            "metric_coordinates_checked": 2 * len(reliability_rows),
            "tolerance_loaded_from_config": tolerance,
            "complete_station_horizon_grid": True,
        },
        CORE_FIGURES[2]: {
            "horizon_method_rows": len(trade_rows),
            "plotted_points_checked": len(trade_rows),
            "candidate_delta_coordinates_checked": 2 * len(candidate_rows),
            "baseline_reference_values_checked": 2 * len(horizons),
            "deltas_recomputed_from_source_means": True,
            "complete_station_availability_checked": len(trade_rows),
        },
    }
    return artifact, claim_validation


def audit_outputs(
    output_dir: Path,
    inputs: list[Path],
    accessibility_artifact: Path,
    claim_validation: dict,
) -> dict:
    require_file(accessibility_artifact)
    records: list[dict] = []
    for stem_name in CORE_FIGURES:
        pdf = require_file(output_dir / f"{stem_name}.pdf")
        png = require_file(output_dir / f"{stem_name}.png")
        with Image.open(png) as image:
            width, height = image.size
            dpi = image.info.get("dpi")
        pdf_bytes = pdf.read_bytes()
        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError(f"Figure is not a PDF: {pdf}")
        media_box = re.search(
            rb"/MediaBox\s*\[\s*([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*\]",
            pdf_bytes,
        )
        if media_box is None:
            raise ValueError(f"Figure PDF has no readable MediaBox: {pdf}")
        x0, _y0, x1, _y1 = (float(value) for value in media_box.groups())
        page_width_bp = x1 - x0
        if not math.isfinite(page_width_bp) or page_width_bp <= 0:
            raise ValueError(f"Figure PDF has an invalid page width: {pdf}")
        source_minimum = SOURCE_TEXT_MINIMA.get(stem_name)
        if source_minimum is None:
            raise ValueError(f"Figure source typography was not audited: {stem_name}")
        placement_fraction = FIGURE_PLACEMENT_FRACTIONS[stem_name]
        placement_scale = placement_fraction * TARGET_TEXT_WIDTH_BP / page_width_bp
        estimated_final_minimum = source_minimum * placement_scale
        if estimated_final_minimum + 1e-9 < MIN_FINAL_FIGURE_TEXT_PT:
            raise ValueError(
                f"Estimated final figure text is too small in {stem_name}: "
                f"{estimated_final_minimum:.3f} pt < {MIN_FINAL_FIGURE_TEXT_PT:.3f} pt"
            )
        if width < 1600 or height < 700:
            raise ValueError(f"High-resolution PNG check failed for {png}: {width} x {height}")
        records.append(
            {
                "stem": stem_name,
                "pdf": {
                    "path": pdf.relative_to(ROOT).as_posix(),
                    "bytes": pdf.stat().st_size,
                    "sha256": sha256(pdf),
                    "vector_only": b"/Subtype /Image" not in pdf_bytes,
                },
                "png": {
                    "path": png.relative_to(ROOT).as_posix(),
                    "bytes": png.stat().st_size,
                    "sha256": sha256(png),
                    "width_px": width,
                    "height_px": height,
                    "dpi": list(dpi) if dpi else None,
                },
                "typography": {
                    "source_minimum_pt": source_minimum,
                    "target_text_width_mm": 122.0,
                    "placement_width_fraction": placement_fraction,
                    "pdf_page_width_bp": page_width_bp,
                    "placement_scale": placement_scale,
                    "estimated_final_minimum_pt": estimated_final_minimum,
                    "required_final_minimum_pt": MIN_FINAL_FIGURE_TEXT_PT,
                    "status": "PASS",
                },
                "status": "PASS",
            }
        )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "core_figure_count": len(records),
        "deterministic_renderer": "scripts/render_figures.py",
        "design_width_inches": 4.78,
        "minimum_source_figure_text_pt": MIN_SOURCE_FIGURE_TEXT_PT,
        "minimum_final_figure_text_pt": MIN_FINAL_FIGURE_TEXT_PT,
        "png_export_dpi": 450,
        "accessibility": {
            "palette": "Okabe-Ito-derived",
            "redundant_encoding": "marker shape, fill, and line style",
            "geographic_claims": False,
            "alt_text": {
                "path": accessibility_artifact.relative_to(ROOT).as_posix(),
                "bytes": accessibility_artifact.stat().st_size,
                "sha256": sha256(accessibility_artifact),
                "figure_count": len(CORE_FIGURES),
            },
            "claim_validation": claim_validation,
        },
        "inputs": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(inputs)
        ],
        "figures": records,
    }
    manifest = output_dir.parent / "tables" / "visualization_audit.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the three core reliability figures.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = require_file(ROOT / args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / cfg["outputs"]["figures"]
    configure_style()
    external, local, trade, verdict, inputs = validated_inputs(cfg)
    protocol_station_screening(external, verdict, cfg, output_dir)
    station_worst_undercoverage(local, cfg, output_dir)
    width_score_tradeoff(trade, cfg, output_dir)
    accessibility_artifact, claim_validation = write_accessibility_artifact(
        external,
        local,
        trade,
        verdict,
        cfg,
        output_dir,
    )
    payload = audit_outputs(
        output_dir,
        [config_path, Path(__file__), *inputs],
        accessibility_artifact,
        claim_validation,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
