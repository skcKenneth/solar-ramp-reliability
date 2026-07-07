from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.visualization import apply_publication_style, clean_axes  # noqa: E402

apply_publication_style()
TABLES = ROOT / "outputs" / "advanced" / "tables"
CACHE = ROOT / "outputs" / "advanced" / "cache"
FIGURES = ROOT / "outputs" / "advanced" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "persistence_condition_cqr": "#7F7F7F",
    "lgbm_condition_cqr": "#56B4E9",
    "clean_condition_cqr": "#D55E00",
    "robust_condition_cqr": "#0072B2",
    "robust_risk_mondrian": "#CC79A7",
    "robust_local_cqr": "#009E73",
    "robust_rolling_cqr": "#4D4D4D",
    "oracle_ramp_mondrian": "#E69F00",
}
DISPLAY = {
    "persistence_condition_cqr": "Persistence + CQR",
    "lgbm_condition_cqr": "LightGBM quantile + CQR",
    "clean_condition_cqr": "Clean-trained ensemble + CQR",
    "robust_condition_cqr": "Mask-augmented ensemble + CQR",
    "robust_risk_mondrian": "Risk-Mondrian CQR",
    "robust_local_cqr": "Descriptor-local CQR",
    "robust_rolling_cqr": "Rolling CQR",
    "oracle_ramp_mondrian": "Oracle ramp Mondrian",
    "clean": "No added outage",
    "irradiance_outage": "Irradiance outage",
    "weather_outage": "Weather outage",
    "recent_power_outage": "Recent-power outage",
    "combined_outage": "Combined outage",
}
CONDITIONS = [
    "clean",
    "irradiance_outage",
    "weather_outage",
    "recent_power_outage",
    "combined_outage",
]


def save(fig: plt.Figure, name: str) -> None:
    stem = FIGURES / name
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=9.5, fontweight="semibold", va="bottom")


def protocol_figure() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x0, y0, w, h = 0.04, 0.67, 0.91, 0.13
    ax.text(x0, y0 + 0.20, "Four seasonal rolling-origin replications", fontsize=8.2, fontweight="semibold")
    colors = {"train": "#D9D9D9", "cal": "#B8D6EB", "test": "#A8D8C5"}
    segments = [
        (0.00, 0.36, "Train", "train"),
        (0.36, 0.49, "Cal Q4-2019", "cal"),
        (0.49, 0.61, "Test Q1", "test"),
        (0.61, 0.74, "Test Q2", "test"),
        (0.74, 0.87, "Test Q3", "test"),
        (0.87, 1.00, "Test Q4", "test"),
    ]
    for left, right, text, kind in segments:
        ax.add_patch(Rectangle((x0 + left * w, y0), (right - left) * w, h, facecolor=colors[kind], edgecolor="white", linewidth=0.7))
        ax.text(x0 + (left + right) * w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=6.7)
    boxes = [
        (0.04, 0.18, 0.14, 0.25, "Observed 15-min\nSCADA context", "#E6E6E6"),
        (0.23, 0.18, 0.15, 0.25, "Source-only\nmask augmentation", "#FCE5D8"),
        (0.43, 0.18, 0.15, 0.25, "Quantile models\n+ ramp-risk score", "#DDEBF7"),
        (0.63, 0.18, 0.15, 0.25, "Condition / risk /\nlocal calibration", "#E2F0D9"),
        (0.83, 0.18, 0.13, 0.25, "Conditional\nreliability audit", "#E8E1F2"),
    ]
    for x, y, bw, bh, text, color in boxes:
        ax.add_patch(Rectangle((x, y), bw, bh, facecolor=color, edgecolor="#555555", linewidth=0.7))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center", fontsize=7.0)
    for start, end in [((0.18, .305), (.23, .305)), ((.38, .305), (.43, .305)), ((.58, .305), (.63, .305)), ((.78, .305), (.83, .305))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#555555"))
    ax.text(0.50, 0.06, "Targets remain sealed; inference is clustered by test day and temporal fold", ha="center", fontsize=6.8, color="#444444")
    save(fig, "fig07_advanced_protocol")


def worst_group_figure(aggregate: pd.DataFrame) -> None:
    order = [
        "persistence_condition_cqr",
        "clean_condition_cqr",
        "lgbm_condition_cqr",
        "robust_condition_cqr",
        "robust_rolling_cqr",
        "robust_risk_mondrian",
        "robust_local_cqr",
        "oracle_ramp_mondrian",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.35), sharey=True)
    for ax, horizon, label in zip(axes, [1, 4], ["a", "b"]):
        sub = aggregate[aggregate.horizon_steps == horizon].set_index("method").loc[order]
        y = np.arange(len(order))
        ax.scatter(sub.worst_group_undercoverage, y, s=30, c=[COLORS[m] for m in order], edgecolor="white", linewidth=0.45, zorder=3)
        ax.axvline(0.05, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_yticks(y, [DISPLAY[m] for m in order])
        ax.invert_yaxis()
        ax.set_xlim(-0.01, 0.94)
        ax.set_xlabel("Worst-group undercoverage")
        ax.set_title(f"{horizon * 15}-min horizon", pad=5)
        clean_axes(ax, "x")
        panel(ax, label)
    fig.subplots_adjust(wspace=0.15)
    save(fig, "fig08_worst_group_undercoverage")


def coverage_heatmap(metrics: pd.DataFrame) -> None:
    sub = metrics[metrics.method == "robust_local_cqr"].copy()
    sub["covered_n"] = sub.coverage * sub.n
    agg = sub.groupby(["horizon_steps", "condition", "ramp_group"], as_index=False).agg(covered_n=("covered_n", "sum"), n=("n", "sum"))
    agg["coverage_delta"] = agg.covered_n / agg.n - 0.90
    vmax = max(0.35, float(np.abs(agg.coverage_delta).max()))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), constrained_layout=True)
    for ax, horizon, label in zip(axes, [1, 4], ["a", "b"]):
        pivot = agg[agg.horizon_steps == horizon].pivot(index="condition", columns="ramp_group", values="coverage_delta").loc[CONDITIONS, ["non_ramp", "ramp"]]
        im = ax.imshow(pivot.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_yticks(range(len(CONDITIONS)), [DISPLAY[x] for x in CONDITIONS])
        ax.set_xticks([0, 1], ["Non-ramp", "Ramp"])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7.2, color="black")
        ax.set_title(f"Descriptor-local CQR · {horizon * 15} min", pad=5)
        panel(ax, label)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.025)
    cbar.set_label("Coverage minus 0.90")
    save(fig, "fig09_local_cqr_coverage_heatmap")


def paired_fold_figure(fold_scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharey=True)
    for ax, horizon, label in zip(axes, [1, 4], ["a", "b"]):
        sub = fold_scores[(fold_scores.horizon_steps == horizon) & fold_scores.method.isin(["robust_condition_cqr", "robust_local_cqr"])]
        pivot = sub.pivot(index="fold", columns="method", values="worst_undercoverage")
        for i, (fold, row) in enumerate(pivot.iterrows()):
            ax.plot([0, 1], [row.robust_condition_cqr, row.robust_local_cqr], color="#A0A0A0", linewidth=0.8, zorder=1)
            ax.scatter(0, row.robust_condition_cqr, color=COLORS["robust_condition_cqr"], s=26, edgecolor="white", linewidth=0.4, zorder=3)
            ax.scatter(1, row.robust_local_cqr, color=COLORS["robust_local_cqr"], s=26, edgecolor="white", linewidth=0.4, zorder=3)
            ax.text(1.04, row.robust_local_cqr, fold.replace("_2020", ""), fontsize=6.3, va="center")
        ax.set_xticks([0, 1], ["Condition CQR", "Local CQR"])
        ax.set_xlim(-0.2, 1.35)
        ax.set_ylabel("Worst conditional undercoverage")
        ax.set_title(f"{horizon * 15}-min horizon", pad=5)
        clean_axes(ax, "y")
        panel(ax, label)
    save(fig, "fig10_foldwise_paired_improvement")


def risk_diagnostic_figure(risk: pd.DataFrame) -> None:
    summary = risk.groupby(["horizon_steps", "condition"], as_index=False).agg(
        auc=("roc_auc", "mean"), ap=("average_precision", "mean"), recall=("top_decile_ramp_recall", "mean")
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
    y = np.arange(len(CONDITIONS))
    for ax, horizon, label in zip(axes, [1, 4], ["a", "b"]):
        sub = summary[summary.horizon_steps == horizon].set_index("condition").loc[CONDITIONS]
        ax.scatter(sub.auc, y - 0.16, label="ROC AUC", color="#0072B2", s=26, edgecolor="white", linewidth=0.4)
        ax.scatter(sub.ap, y + 0.16, label="Average precision", color="#D55E00", marker="s", s=24, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y, [DISPLAY[x] for x in CONDITIONS])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Ramp-risk discrimination")
        ax.set_title(f"{horizon * 15}-min horizon", pad=5)
        clean_axes(ax, "x")
        panel(ax, label)
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, "fig11_ramp_risk_diagnostics")


def oracle_gap_figure(gap: pd.DataFrame) -> None:
    methods = [
        ("baseline_worst_undercoverage", "Condition CQR", COLORS["robust_condition_cqr"]),
        ("local_worst_undercoverage", "Descriptor-local CQR", COLORS["robust_local_cqr"]),
        ("oracle_worst_undercoverage", "Oracle ramp groups", COLORS["oracle_ramp_mondrian"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65), sharex=True, sharey=True)
    y = np.arange(len(methods))
    for ax, (_, row), label in zip(axes, gap.sort_values("horizon_steps").iterrows(), ["a", "b"]):
        values = np.array([float(row[column]) for column, _, _ in methods])
        ax.plot(values, y, color="#A0A0A0", linewidth=1.0, zorder=1)
        for yi, (value, (_, _, color)) in enumerate(zip(values, methods)):
            ax.scatter(value, yi, s=38, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(value + 0.012, yi, f"{value:.3f}", va="center", fontsize=6.8)
        ax.axvline(0.05, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_yticks(y, [name for _, name, _ in methods])
        ax.invert_yaxis()
        ax.set_xlim(-0.01, 0.46)
        ax.set_xlabel("Worst-group undercoverage")
        ax.set_title(f"{int(row.horizon_steps) * 15}-min horizon", pad=5)
        clean_axes(ax, "x")
        panel(ax, label)
    fig.subplots_adjust(wspace=0.16)
    save(fig, "fig12_oracle_conditioning_gap")


def example_interval_figure(pred: pd.DataFrame) -> None:
    base = pred[(pred.horizon_steps == 1) & (pred.condition.astype(str) == "clean") & (pred.method.astype(str) == "robust_condition_cqr")].copy()
    local = pred[(pred.horizon_steps == 1) & (pred.condition.astype(str) == "clean") & (pred.method.astype(str) == "robust_local_cqr")].copy()
    day_rank = base.groupby("date").agg(ramp_count=("ramp_fraction", lambda x: int((x >= 0.10).sum())), max_ramp=("ramp_fraction", "max")).sort_values(["ramp_count", "max_ramp"], ascending=False)
    chosen = day_rank.index[0]
    b = base[base.date == chosen].sort_values("timestamp")
    l = local[local.date == chosen].sort_values("timestamp")
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.1), sharex=True, sharey=True)
    for ax, data, method, label in [
        (axes[0], b, "robust_condition_cqr", "a"),
        (axes[1], l, "robust_local_cqr", "b"),
    ]:
        color = COLORS[method]
        ax.fill_between(data.timestamp, data.lower, data.upper, color=color, alpha=0.20, linewidth=0)
        ax.plot(data.timestamp, data.y, color="#222222", linewidth=1.0, label="Observed")
        ax.plot(data.timestamp, data["median"], color=color, linewidth=1.0, label="Median")
        ramps = data.ramp_fraction >= 0.10
        ax.scatter(data.loc[ramps, "timestamp"], data.loc[ramps, "y"], color="#D55E00", s=15, zorder=4, label="Ramp target")
        missed = ramps & (~data.covered.astype(bool))
        ax.scatter(data.loc[missed, "timestamp"], data.loc[missed, "y"], facecolors="none", edgecolors="#000000", s=28, linewidth=0.8, zorder=5, label="Missed ramp")
        ax.set_ylabel("Power (MW)")
        ax.set_title(DISPLAY[method], pad=4)
        clean_axes(ax, "y")
        panel(ax, label)
    axes[1].set_xlabel(f"Forecast origin on {pd.Timestamp(chosen).date()}")
    axes[0].legend(frameon=False, ncol=4, fontsize=6.7, loc="upper center")
    save(fig, "fig13_representative_ramp_intervals")


def directional_figure(direction: pd.DataFrame) -> None:
    methods = ["robust_condition_cqr", "robust_local_cqr"]
    agg = direction[direction.method.isin(methods)].groupby(["horizon_steps", "method", "ramp_direction"], as_index=False).agg(
        mean_under=("undercoverage_error", "mean"),
        min_under=("undercoverage_error", "min"),
        max_under=("undercoverage_error", "max"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75), sharey=True)
    for ax, horizon, label in zip(axes, [1, 4], ["a", "b"]):
        sub = agg[agg.horizon_steps == horizon]
        xbase = {"down": 0, "up": 1}
        for j, method in enumerate(methods):
            m = sub[sub.method == method].set_index("ramp_direction")
            xs = np.array([xbase[d] for d in ["down", "up"]], dtype=float) + (-0.10 if j == 0 else 0.10)
            vals = m.loc[["down", "up"], "mean_under"].to_numpy()
            lo = m.loc[["down", "up"], "min_under"].to_numpy()
            hi = m.loc[["down", "up"], "max_under"].to_numpy()
            ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="o", color=COLORS[method], ecolor=COLORS[method], capsize=2, label=DISPLAY[method])
        ax.set_xticks([0, 1], ["Down ramps", "Up ramps"])
        ax.set_ylabel("Undercoverage across fold-condition cells")
        ax.set_title(f"{horizon * 15}-min horizon", pad=5)
        clean_axes(ax, "y")
        panel(ax, label)
    axes[0].legend(frameon=False, fontsize=6.8)
    save(fig, "fig14_directional_ramp_reliability")


def main() -> None:
    aggregate = pd.read_csv(TABLES / "method_aggregate.csv")
    metrics = pd.read_csv(TABLES / "conditional_metrics.csv")
    fold_scores = pd.read_csv(TABLES / "fold_method_scores.csv")
    risk = pd.read_csv(TABLES / "ramp_risk_diagnostics.csv")
    gap = pd.read_csv(TABLES / "oracle_gap_summary.csv")
    direction = pd.read_csv(TABLES / "directional_ramp_metrics.csv")
    pred = pd.read_pickle(CACHE / "advanced_predictions.pkl.gz", compression="gzip")
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    pred["date"] = pd.to_datetime(pred["date"])

    protocol_figure()
    worst_group_figure(aggregate)
    coverage_heatmap(metrics)
    paired_fold_figure(fold_scores)
    risk_diagnostic_figure(risk)
    oracle_gap_figure(gap)
    example_interval_figure(pred)
    directional_figure(direction)
    print(f"Rendered advanced figures to {FIGURES}")


if __name__ == "__main__":
    main()
