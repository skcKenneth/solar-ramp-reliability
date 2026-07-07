from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def cqr_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.maximum.reduce([lower - y, y - upper, np.zeros_like(y)])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("No finite calibration scores")
    rank = math.ceil((scores.size + 1) * (1 - alpha))
    rank = min(max(rank, 1), scores.size)
    return float(np.partition(scores, rank - 1)[rank - 1])


def adjust_interval(
    lower: np.ndarray,
    upper: np.ndarray,
    qhat: float | np.ndarray,
    capacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.clip(np.asarray(lower) - qhat, 0.0, capacity)
    hi = np.clip(np.asarray(upper) + qhat, 0.0, capacity)
    return np.minimum(lo, hi), np.maximum(lo, hi)


@dataclass(frozen=True)
class CalibrationTables:
    global_q: float
    condition_q: dict[str, float]
    condition_volatility_q: dict[tuple[str, str], float]
    group_counts: dict[tuple[str, str], int]
    volatility_threshold: float


def fit_calibration_tables(
    score_frame: pd.DataFrame,
    alpha: float,
    volatility_quantile: float,
) -> CalibrationTables:
    required = {"score", "condition", "volatility"}
    if not required.issubset(score_frame.columns):
        raise ValueError(f"Missing columns: {sorted(required.difference(score_frame.columns))}")
    threshold = float(score_frame.loc[score_frame["condition"] == "clean", "volatility"].quantile(volatility_quantile))
    sf = score_frame.copy()
    sf["volatility_group"] = np.where(sf["volatility"] >= threshold, "high", "low")
    global_q = conformal_quantile(sf["score"].to_numpy(), alpha)
    condition_q = {
        str(condition): conformal_quantile(group["score"].to_numpy(), alpha)
        for condition, group in sf.groupby("condition", sort=True)
    }
    cv_q: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for key, group in sf.groupby(["condition", "volatility_group"], sort=True):
        tuple_key = (str(key[0]), str(key[1]))
        cv_q[tuple_key] = conformal_quantile(group["score"].to_numpy(), alpha)
        counts[tuple_key] = int(len(group))
    return CalibrationTables(global_q, condition_q, cv_q, counts, threshold)


def qhat_for_method(
    method: str,
    condition: str,
    volatility: np.ndarray,
    tables: CalibrationTables,
    shrinkage_k: float,
) -> np.ndarray:
    volatility = np.asarray(volatility, dtype=float)
    if method == "native":
        return np.zeros_like(volatility)
    if method == "clean_cqr":
        return np.full_like(volatility, tables.condition_q["clean"])
    if method == "augmented_cqr":
        return np.full_like(volatility, tables.global_q)
    if method == "mondrian_cqr":
        return np.full_like(volatility, tables.condition_q.get(condition, tables.global_q))
    if method == "hierarchical_cqr":
        out = np.empty_like(volatility)
        group = np.where(volatility >= tables.volatility_threshold, "high", "low")
        condition_base = tables.condition_q.get(condition, tables.global_q)
        for label in ["low", "high"]:
            mask = group == label
            key = (condition, label)
            local = tables.condition_volatility_q.get(key, condition_base)
            n = tables.group_counts.get(key, 0)
            weight = n / (n + shrinkage_k) if n > 0 else 0.0
            out[mask] = weight * local + (1 - weight) * condition_base
        return out
    raise ValueError(f"Unknown method: {method}")
