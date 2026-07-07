from __future__ import annotations

import numpy as np
import pandas as pd


def interval_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    below = (lower - y) * (y < lower)
    above = (y - upper) * (y > upper)
    interval_score = width + (2 / alpha) * below + (2 / alpha) * above
    return {
        "n": int(y.size),
        "coverage": float(np.mean(covered)),
        "coverage_error": float(abs(np.mean(covered) - (1 - alpha))),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "interval_score": float(np.mean(interval_score)),
    }


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    error = y - pred
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def summarize_predictions(predictions: pd.DataFrame, alpha: float, min_group_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["horizon_steps", "method", "condition", "ramp_group"]
    for key, group in predictions.groupby(group_cols, sort=True):
        if len(group) < min_group_n:
            continue
        metrics = interval_metrics(group["y"], group["lower"], group["upper"], alpha)
        metrics.update(point_metrics(group["y"], group["median"]))
        rows.append(dict(zip(group_cols, key)) | metrics)
    return pd.DataFrame(rows)


def method_primary_scores(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["horizon_steps", "method"], as_index=False)
        .agg(
            worst_group_coverage_error=("coverage_error", "max"),
            mean_group_coverage_error=("coverage_error", "mean"),
            mean_width=("mean_width", "mean"),
            mean_interval_score=("interval_score", "mean"),
            groups=("coverage_error", "size"),
        )
    )
