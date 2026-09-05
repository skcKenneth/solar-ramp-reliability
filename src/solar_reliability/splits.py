from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldMasks:
    train: pd.Series
    calibration: pd.Series
    test: pd.Series

    def counts(self) -> dict[str, int]:
        return {
            "train": int(self.train.sum()),
            "calibration": int(self.calibration.sum()),
            "test": int(self.test.sum()),
        }


def _fold_boundaries(fold: Mapping[str, object]) -> dict[str, pd.Timestamp]:
    required = (
        "train_end",
        "calibration_start",
        "calibration_end",
        "test_start",
        "test_end",
    )
    missing = [key for key in required if key not in fold]
    if missing:
        raise ValueError(f"Fold is missing boundaries: {missing}")
    boundaries = {key: pd.Timestamp(fold[key]) for key in required}
    if any(pd.isna(value) for value in boundaries.values()):
        raise ValueError("Fold boundaries must be valid timestamps")
    if not (
        boundaries["train_end"] < boundaries["calibration_start"]
        <= boundaries["calibration_end"] < boundaries["test_start"]
        <= boundaries["test_end"]
    ):
        raise ValueError("Fold periods must be strictly chronological and non-overlapping")
    return boundaries


def leakage_safe_fold_masks(
    index: pd.Index,
    target_timestamps: pd.Series | pd.Index | np.ndarray,
    fold: Mapping[str, object],
    *,
    horizon_steps: int | None = None,
    sampling_minutes: int = 15,
    require_nonempty: bool = True,
) -> FoldMasks:
    """Build chronological masks whose labels remain inside their own period."""

    origins = pd.DatetimeIndex(index)
    targets = pd.DatetimeIndex(pd.to_datetime(np.asarray(target_timestamps)))
    if len(origins) != len(targets):
        raise ValueError("origin and target timestamps must have equal length")
    if origins.has_duplicates or not origins.is_monotonic_increasing:
        raise ValueError("origin timestamps must be unique and monotonic increasing")
    if targets.hasnans:
        raise ValueError("target timestamps must not contain missing values")
    if not np.all(targets > origins):
        raise ValueError("every target timestamp must be later than its forecast origin")
    if horizon_steps is not None:
        if int(horizon_steps) <= 0:
            raise ValueError("horizon_steps must be positive")
        expected = pd.Timedelta(minutes=int(horizon_steps) * int(sampling_minutes))
        if not np.all((targets - origins) == expected):
            raise ValueError("target timestamps do not match the configured fixed horizon")

    bounds = _fold_boundaries(fold)
    train = pd.Series(
        (origins <= bounds["train_end"]) & (targets <= bounds["train_end"]),
        index=index,
        dtype=bool,
    )
    calibration = pd.Series(
        (origins >= bounds["calibration_start"])
        & (origins <= bounds["calibration_end"])
        & (targets <= bounds["calibration_end"]),
        index=index,
        dtype=bool,
    )
    test = pd.Series(
        (origins >= bounds["test_start"])
        & (origins <= bounds["test_end"])
        & (targets <= bounds["test_end"]),
        index=index,
        dtype=bool,
    )
    masks = FoldMasks(train=train, calibration=calibration, test=test)
    assert_leakage_safe_masks(origins, targets, masks, bounds, require_nonempty=require_nonempty)
    return masks


def assert_leakage_safe_masks(
    origins: pd.DatetimeIndex,
    targets: pd.DatetimeIndex,
    masks: FoldMasks,
    bounds: Mapping[str, pd.Timestamp],
    *,
    require_nonempty: bool,
) -> None:
    overlap = (
        masks.train.astype(int)
        + masks.calibration.astype(int)
        + masks.test.astype(int)
    ) > 1
    if bool(overlap.any()):
        raise AssertionError("chronological split masks overlap")
    if require_nonempty and any(count == 0 for count in masks.counts().values()):
        raise ValueError(f"one or more chronological splits are empty: {masks.counts()}")

    checks = (
        (masks.train.to_numpy(), None, bounds["train_end"], "training"),
        (
            masks.calibration.to_numpy(),
            bounds["calibration_start"],
            bounds["calibration_end"],
            "calibration",
        ),
        (masks.test.to_numpy(), bounds["test_start"], bounds["test_end"], "test"),
    )
    for mask, start, end, label in checks:
        if not mask.any():
            continue
        selected_origins = origins[mask]
        selected_targets = targets[mask]
        if start is not None and selected_origins.min() < start:
            raise AssertionError(f"{label} origin precedes its period")
        if selected_origins.max() > end or selected_targets.max() > end:
            raise AssertionError(f"{label} origin or target crosses its period end")
        if start is not None and selected_targets.min() < start:
            raise AssertionError(f"{label} target precedes its period")


def split_audit_summary(
    index: pd.Index,
    target_timestamps: pd.Series | pd.Index | np.ndarray,
    masks: FoldMasks,
) -> dict[str, dict[str, object]]:
    origins = pd.DatetimeIndex(index)
    targets = pd.DatetimeIndex(pd.to_datetime(np.asarray(target_timestamps)))
    summary: dict[str, dict[str, object]] = {}
    for label, series in (
        ("train", masks.train),
        ("calibration", masks.calibration),
        ("test", masks.test),
    ):
        mask = series.to_numpy()
        summary[label] = {
            "rows": int(mask.sum()),
            "origin_start": str(origins[mask].min()) if mask.any() else None,
            "origin_end": str(origins[mask].max()) if mask.any() else None,
            "target_start": str(targets[mask].min()) if mask.any() else None,
            "target_end": str(targets[mask].max()) if mask.any() else None,
        }
    return summary

