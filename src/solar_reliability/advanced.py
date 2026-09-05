from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .conformal import conformal_quantile
from .features import apply_condition

DESCRIPTOR_CONTEXT_COLUMNS = (
    "recent_power_missing_fraction",
    "irradiance_missing_fraction",
    "weather_missing_fraction",
    "observed_context_fraction",
    "hour_sin",
    "hour_cos",
    "transition_risk",
)
LOCAL_SCORE_OBJECTIVE = "quantile"
LOCAL_SCORE_NUM_LEAVES = 15
LOCAL_SCORE_REG_LAMBDA = 2.0
LOCAL_SCORE_N_JOBS = 2


def augment_training_set(
    x: pd.DataFrame,
    y: pd.Series,
    ramp_labels: np.ndarray,
    conditions: list[str],
    condition_fraction: float,
    ramp_sample_weight: float,
    degraded_sample_weight: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series]:
    if not 0 < condition_fraction <= 1:
        raise ValueError("condition_fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    frames = [x.copy()]
    targets = [y.copy()]
    weights = [np.where(ramp_labels, ramp_sample_weight, 1.0)]
    labels = [pd.Series("clean", index=x.index)]

    degraded = [c for c in conditions if c != "clean"]
    n_total = max(len(degraded), int(round(len(x) * condition_fraction)))
    permutation = rng.permutation(len(x))
    chunks = np.array_split(permutation[: min(n_total, len(x))], len(degraded))
    for condition, chunk in zip(degraded, chunks):
        take = np.sort(chunk)
        x_part = apply_condition(x.iloc[take], condition)
        y_part = y.iloc[take]
        ramp_part = np.asarray(ramp_labels, dtype=bool)[take]
        w = np.where(ramp_part, ramp_sample_weight, 1.0) * degraded_sample_weight
        frames.append(x_part)
        targets.append(y_part)
        weights.append(w)
        labels.append(pd.Series(condition, index=x_part.index))

    x_aug = pd.concat(frames, axis=0)
    y_aug = pd.concat(targets, axis=0)
    weight_aug = np.concatenate(weights).astype(float)
    condition_aug = pd.concat(labels, axis=0)
    return x_aug, y_aug, weight_aug, condition_aug


def condition_quantiles(score_frame: pd.DataFrame, alpha: float) -> dict[str, float]:
    if not {"score", "condition"}.issubset(score_frame.columns):
        raise ValueError("score_frame requires score and condition")
    return {
        str(condition): conformal_quantile(group["score"].to_numpy(), alpha)
        for condition, group in score_frame.groupby("condition", sort=True)
    }


@dataclass(frozen=True)
class RiskMondrianTables:
    condition_q: dict[str, float]
    local_q: dict[tuple[str, int], float]
    local_n: dict[tuple[str, int], int]
    edges: np.ndarray
    shrinkage_k: float

    def bins(self, risk: np.ndarray) -> np.ndarray:
        return np.digitize(np.asarray(risk, dtype=float), self.edges[1:-1], right=True)

    def qhat(self, condition: str, risk: np.ndarray) -> np.ndarray:
        bins = self.bins(risk)
        base = self.condition_q[condition]
        out = np.full(len(bins), base, dtype=float)
        for label in np.unique(bins):
            key = (condition, int(label))
            local = self.local_q.get(key, base)
            n = self.local_n.get(key, 0)
            w = n / (n + self.shrinkage_k) if n else 0.0
            out[bins == label] = w * local + (1 - w) * base
        return out


def fit_risk_mondrian(
    score_frame: pd.DataFrame,
    alpha: float,
    bins: int,
    shrinkage_k: float = 100.0,
) -> RiskMondrianTables:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    pooled_risk = score_frame["risk"].to_numpy(dtype=float)
    edges = np.quantile(pooled_risk, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    sf = score_frame.copy()
    sf["risk_bin"] = np.digitize(sf["risk"], edges[1:-1], right=True)
    condition_q = condition_quantiles(sf, alpha)
    local_q: dict[tuple[str, int], float] = {}
    local_n: dict[tuple[str, int], int] = {}
    for (condition, risk_bin), group in sf.groupby(["condition", "risk_bin"], sort=True):
        key = (str(condition), int(risk_bin))
        local_q[key] = conformal_quantile(group["score"].to_numpy(), alpha)
        local_n[key] = int(len(group))
    return RiskMondrianTables(condition_q, local_q, local_n, edges, shrinkage_k)


def descriptor_frame(
    x: pd.DataFrame,
    condition: str,
    risk: np.ndarray,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    capacity: float,
    conditions: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(index=x.index)
    frame["risk"] = np.asarray(risk, dtype=float)
    frame["native_width"] = (np.asarray(upper) - np.asarray(lower)) / capacity
    frame["median_level"] = np.asarray(median) / capacity
    for col in DESCRIPTOR_CONTEXT_COLUMNS:
        frame[col] = x[col].to_numpy(dtype=float)
    for label in conditions:
        frame[f"condition__{label}"] = float(condition == label)
    return frame


def chronological_fit_adjust_masks(
    timestamps: pd.Series | pd.DatetimeIndex | np.ndarray,
    fit_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split complete timestamps into an earlier fit block and later adjustment block."""

    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("fit_fraction must lie strictly between 0 and 1")
    parsed = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    if parsed.isna().any():
        raise ValueError("timestamps must all be valid")
    unique_times = np.array(sorted(pd.unique(parsed)))
    if len(unique_times) < 2:
        raise ValueError("at least two unique timestamps are required")
    cut = int(np.floor(len(unique_times) * float(fit_fraction)))
    cut = min(max(cut, 1), len(unique_times) - 1)
    fit_times = set(unique_times[:cut])
    fit_mask = parsed.isin(fit_times).to_numpy(dtype=bool)
    return fit_mask, ~fit_mask


@dataclass
class LocalScoreCalibrator:
    model: LGBMRegressor
    correction: float
    columns: list[str]

    def predict(self, descriptors: pd.DataFrame) -> np.ndarray:
        q = self.model.predict(descriptors[self.columns]) + self.correction
        return np.maximum(q, 0.0)


def fit_local_score_calibrator(
    calibration_frame: pd.DataFrame,
    descriptor_columns: list[str],
    alpha: float,
    fit_fraction: float,
    min_samples_leaf: int,
    max_iter: int,
    learning_rate: float,
    seed: int,
) -> LocalScoreCalibrator:
    fit_mask, adjust_mask = chronological_fit_adjust_masks(
        calibration_frame["timestamp"],
        fit_fraction,
    )
    fit = calibration_frame.loc[fit_mask]
    adjust = calibration_frame.loc[adjust_mask]
    model = LGBMRegressor(
        objective=LOCAL_SCORE_OBJECTIVE,
        alpha=1 - alpha,
        learning_rate=learning_rate,
        n_estimators=max_iter,
        num_leaves=LOCAL_SCORE_NUM_LEAVES,
        min_child_samples=min_samples_leaf,
        reg_lambda=LOCAL_SCORE_REG_LAMBDA,
        verbosity=-1,
        n_jobs=LOCAL_SCORE_N_JOBS,
        random_state=seed,
    )
    model.fit(fit[descriptor_columns], fit["score"])
    raw = model.predict(adjust[descriptor_columns])
    residual = adjust["score"].to_numpy() - raw
    correction = max(0.0, conformal_quantile(residual, alpha))
    return LocalScoreCalibrator(model, correction, descriptor_columns)


def oracle_ramp_quantiles(score_frame: pd.DataFrame, alpha: float) -> dict[tuple[str, str], float]:
    return {
        (str(condition), str(ramp_group)): conformal_quantile(group["score"].to_numpy(), alpha)
        for (condition, ramp_group), group in score_frame.groupby(["condition", "ramp_group"], sort=True)
    }


def rolling_qhat(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    target_timestamps: pd.DatetimeIndex,
    alpha: float,
    window: int,
    min_history: int,
) -> np.ndarray:
    history: deque[float] = deque(maxlen=window)
    for score in np.asarray(calibration_scores, dtype=float)[-window:]:
        if np.isfinite(score):
            history.append(float(score))
    if len(history) < min_history:
        raise ValueError("Insufficient calibration history for rolling conformal")

    pending: deque[tuple[pd.Timestamp, float]] = deque()
    out = np.empty(len(test_scores), dtype=float)
    for i, (origin, target_time, score) in enumerate(zip(timestamps, target_timestamps, test_scores)):
        origin = pd.Timestamp(origin)
        while pending and pending[0][0] <= origin:
            _, released_score = pending.popleft()
            if np.isfinite(released_score):
                history.append(float(released_score))
        out[i] = conformal_quantile(np.asarray(history, dtype=float), alpha)
        pending.append((pd.Timestamp(target_time), float(score)))
    return out
