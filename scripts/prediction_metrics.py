from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


PREDICTION_VALUE_COLUMNS = ("median", "lower", "upper")
_INTERNAL_COLUMNS = (
    "_prediction_usable",
    "_covered_if_issued",
    "_interval_width",
    "_interval_score",
    "_absolute_error",
    "_squared_error",
)


def prepare_prediction_metric_frame(
    predictions: pd.DataFrame,
    *,
    alpha: float,
    additional_finite_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Validate prediction rows and add explicit issued-prediction metrics.

    A prediction is usable only when its median and both interval endpoints are
    finite.  A completely missing prediction tuple is retained as an unavailable
    service attempt.  Infinities, non-numeric values, and partially missing tuples
    are invalid evidence and fail closed.
    """

    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0, 1)")
    required = ["y", *PREDICTION_VALUE_COLUMNS, *additional_finite_columns]
    missing = sorted(set(required).difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction chunk is missing columns: {missing}")

    frame = predictions.copy()
    raw = frame[required]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    coercion_failed = raw.notna() & numeric.isna()
    if bool(coercion_failed.any().any()):
        columns = sorted(coercion_failed.columns[coercion_failed.any()].tolist())
        raise ValueError(f"Prediction chunk contains non-numeric metric inputs: {columns}")

    always_finite = ["y", *additional_finite_columns]
    if not np.isfinite(numeric[always_finite].to_numpy(dtype=float)).all():
        raise ValueError(
            "Prediction chunk contains non-finite required metric inputs: "
            f"{always_finite}"
        )

    values = numeric[list(PREDICTION_VALUE_COLUMNS)]
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("Prediction chunk contains infinite prediction values")
    missing_values = values.isna()
    missing_count = missing_values.sum(axis=1)
    partially_missing = ~missing_count.isin([0, len(PREDICTION_VALUE_COLUMNS)])
    if bool(partially_missing.any()):
        raise ValueError(
            "Prediction chunk contains partially missing prediction tuples; "
            "median, lower, and upper must be all finite or all missing"
        )

    usable = missing_count.eq(0)
    if bool((values.loc[usable, "upper"] < values.loc[usable, "lower"]).any()):
        raise ValueError("Prediction chunk contains an interval with upper < lower")

    frame[required] = numeric
    frame["_prediction_usable"] = usable.to_numpy(dtype=bool)
    frame["_covered_if_issued"] = False
    frame["_interval_width"] = np.nan
    frame["_interval_score"] = np.nan
    frame["_absolute_error"] = np.nan
    frame["_squared_error"] = np.nan

    if bool(usable.any()):
        usable_index = usable[usable].index
        y = numeric.loc[usable_index, "y"].to_numpy(dtype=float)
        median = numeric.loc[usable_index, "median"].to_numpy(dtype=float)
        lower = numeric.loc[usable_index, "lower"].to_numpy(dtype=float)
        upper = numeric.loc[usable_index, "upper"].to_numpy(dtype=float)
        covered = (y >= lower) & (y <= upper)
        width = upper - lower
        score = (
            width
            + (2.0 / float(alpha)) * (lower - y) * (y < lower)
            + (2.0 / float(alpha)) * (y - upper) * (y > upper)
        )
        error = y - median
        frame.loc[usable_index, "_covered_if_issued"] = covered
        frame.loc[usable_index, "_interval_width"] = width
        frame.loc[usable_index, "_interval_score"] = score
        frame.loc[usable_index, "_absolute_error"] = np.abs(error)
        frame.loc[usable_index, "_squared_error"] = error**2

    return frame


def aggregate_prediction_metrics(
    prepared: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    alpha: float,
    capacity: float,
) -> pd.DataFrame:
    """Aggregate with usable predictions as metric denominator.

    ``service_coverage`` retains unavailable attempts in its denominator, while
    ``coverage`` and ``issued_coverage`` describe only rows where an interval was
    issued.  ``n`` remains as a compatibility alias for ``usable_n``.
    """

    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not np.isfinite(float(capacity)) or float(capacity) <= 0:
        raise ValueError("capacity must be finite and positive")
    missing = sorted(
        set([*group_columns, *_INTERNAL_COLUMNS]).difference(prepared.columns)
    )
    if missing:
        raise ValueError(f"Prepared prediction frame is missing columns: {missing}")

    grouped = (
        prepared.groupby(list(group_columns), observed=True, sort=True)
        .agg(
            attempted_n=("_prediction_usable", "size"),
            usable_n=("_prediction_usable", "sum"),
            covered_n=("_covered_if_issued", "sum"),
            mean_width=("_interval_width", "mean"),
            median_width=("_interval_width", "median"),
            interval_score=("_interval_score", "mean"),
            mae=("_absolute_error", "mean"),
            mean_squared_error=("_squared_error", "mean"),
        )
        .reset_index()
    )
    count_columns = ["attempted_n", "usable_n", "covered_n"]
    grouped[count_columns] = grouped[count_columns].astype(int)
    grouped["n"] = grouped["usable_n"]
    grouped["unavailable_n"] = grouped["attempted_n"] - grouped["usable_n"]
    grouped["availability_rate"] = grouped["usable_n"] / grouped["attempted_n"]
    denominator = grouped["usable_n"].replace(0, np.nan)
    grouped["coverage"] = grouped["covered_n"] / denominator
    grouped["issued_coverage"] = grouped["coverage"]
    grouped["service_coverage"] = grouped["covered_n"] / grouped["attempted_n"]
    target = 1.0 - float(alpha)
    grouped["coverage_error"] = (grouped["coverage"] - target).abs()
    grouped["undercoverage_error"] = np.maximum(0.0, target - grouped["coverage"])
    grouped["overcoverage_error"] = np.maximum(0.0, grouped["coverage"] - target)
    grouped["service_coverage_error"] = (grouped["service_coverage"] - target).abs()
    grouped["service_undercoverage_error"] = np.maximum(
        0.0, target - grouped["service_coverage"]
    )
    grouped["service_overcoverage_error"] = np.maximum(
        0.0, grouped["service_coverage"] - target
    )
    grouped["normalized_width"] = grouped["mean_width"] / float(capacity)
    grouped["normalized_interval_score"] = grouped["interval_score"] / float(capacity)
    grouped["rmse"] = np.sqrt(grouped.pop("mean_squared_error"))
    grouped["metric_denominator"] = "usable_predictions"
    grouped["service_denominator"] = "attempted_predictions"
    return grouped


def require_full_availability(
    prepared: pd.DataFrame,
    methods: Iterable[str],
    *,
    context: str,
) -> None:
    """Fail closed when a decision-bearing method did not issue every prediction."""

    required = set(map(str, methods))
    if not required:
        return
    observed = set(prepared["method"].astype(str)) if "method" in prepared else set()
    missing_methods = sorted(required - observed)
    if missing_methods:
        raise RuntimeError(
            f"{context}: decision method rows are missing: {missing_methods}"
        )
    selected = prepared[prepared["method"].astype(str).isin(required)]
    unavailable = selected[~selected["_prediction_usable"].astype(bool)]
    if unavailable.empty:
        return
    counts = (
        unavailable.groupby("method", observed=True, sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    raise RuntimeError(
        f"{context}: decision methods contain unavailable predictions: {counts}"
    )


def require_complete_method_condition_rows(
    predictions: pd.DataFrame,
    *,
    expected_methods: Iterable[str],
    expected_conditions: Iterable[str],
    baseline_method: str,
    context: str,
) -> None:
    """Require an exact method-condition grid with one row per baseline attempt."""

    required_columns = {"method", "condition"}
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise RuntimeError(
            f"{context}: cannot audit method-condition rows; missing {missing_columns}"
        )
    methods = set(map(str, expected_methods))
    conditions = set(map(str, expected_conditions))
    baseline = str(baseline_method)
    if baseline not in methods:
        raise ValueError("baseline_method must be included in expected_methods")

    observed_methods = set(predictions["method"].astype(str))
    observed_conditions = set(predictions["condition"].astype(str))
    if observed_methods != methods or observed_conditions != conditions:
        raise RuntimeError(
            f"{context}: method/condition set mismatch; "
            f"expected_methods={sorted(methods)}, observed_methods={sorted(observed_methods)}, "
            f"expected_conditions={sorted(conditions)}, "
            f"observed_conditions={sorted(observed_conditions)}"
        )

    counts = (
        predictions.assign(
            _method=predictions["method"].astype(str),
            _condition=predictions["condition"].astype(str),
        )
        .groupby(["_method", "_condition"], observed=True, sort=True)
        .size()
    )
    expected_index = pd.MultiIndex.from_product(
        [sorted(methods), sorted(conditions)], names=["_method", "_condition"]
    )
    counts = counts.reindex(expected_index)
    if counts.isna().any():
        missing_cells = [f"{method}/{condition}" for method, condition in counts[counts.isna()].index]
        raise RuntimeError(
            f"{context}: missing method-condition cells: {missing_cells}"
        )
    baseline_counts = counts.loc[baseline]
    reference_n = int(baseline_counts.iloc[0])
    if reference_n <= 0 or not (baseline_counts == reference_n).all():
        raise RuntimeError(
            f"{context}: primary-baseline condition row counts are inconsistent: "
            f"{baseline_counts.astype(int).to_dict()}"
        )
    mismatched = counts[counts != reference_n]
    if not mismatched.empty:
        details = {
            f"{method}/{condition}": int(value)
            for (method, condition), value in mismatched.items()
        }
        raise RuntimeError(
            f"{context}: method-condition row counts differ from baseline n={reference_n}: "
            f"{details}"
        )
