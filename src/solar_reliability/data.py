from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .splits import leakage_safe_fold_masks

COLUMN_MAP = {
    "Time(year-month-day h:m:s)": "timestamp",
    "date": "timestamp",
    "Total solar irradiance (W/m2)": "tsi",
    "Direct normal irradiance (W/m2)": "dni",
    "Global horizontal irradiance (W/m2)": "ghi",
    "Air temperature  (°C) ": "temperature",
    "Atmosphere (hpa)": "pressure",
    "Relative humidity (%)": "humidity",
    "Power (MW)": "power",
}

SENSOR_COLUMNS = ["tsi", "dni", "ghi", "temperature", "pressure", "humidity"]


def load_solar_data(path: str | Path, nominal_capacity_mw: float = 50.0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path).rename(columns=COLUMN_MAP)
    else:
        frame = pd.read_excel(path).rename(columns=COLUMN_MAP)
    required = {"timestamp", "power", "tsi", "dni", "ghi", "temperature", "pressure"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "humidity" not in frame.columns:
        frame["humidity"] = np.nan

    frame = frame[["timestamp", *SENSOR_COLUMNS, "power"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.set_index("timestamp")

    numeric = ["power", *SENSOR_COLUMNS]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame[SENSOR_COLUMNS] = frame[SENSOR_COLUMNS].replace(-99, np.nan)

    frame.loc[frame["tsi"] < 0, "tsi"] = np.nan
    frame.loc[frame["dni"] < 0, "dni"] = np.nan
    frame.loc[frame["ghi"] < 0, "ghi"] = np.nan
    frame.loc[~frame["temperature"].between(-60, 60), "temperature"] = np.nan
    frame.loc[~frame["pressure"].between(700, 1100), "pressure"] = np.nan
    frame.loc[~frame["humidity"].between(0, 100), "humidity"] = np.nan
    frame.loc[~frame["power"].between(0, nominal_capacity_mw), "power"] = np.nan

    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="15min")
    if not frame.index.equals(expected):
        frame = frame.reindex(expected)
        frame.index.name = "timestamp"
    return frame


def split_frame(
    frame: pd.DataFrame,
    train_end: str,
    calibration_start: str,
    calibration_end: str,
    test_start: str,
    test_end: str,
    *,
    target_timestamps: pd.Series | pd.Index | np.ndarray,
    horizon_steps: int,
    sampling_minutes: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split forecast-origin rows without allowing labels to cross a boundary."""

    fold = {
        "train_end": train_end,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "test_start": test_start,
        "test_end": test_end,
    }
    masks = leakage_safe_fold_masks(
        frame.index,
        target_timestamps,
        fold,
        horizon_steps=horizon_steps,
        sampling_minutes=sampling_minutes,
    )
    return (
        frame.loc[masks.train].copy(),
        frame.loc[masks.calibration].copy(),
        frame.loc[masks.test].copy(),
    )


def missingness_summary(frame: pd.DataFrame, columns: Iterable[str] = SENSOR_COLUMNS) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for column in columns:
        mask = frame[column].isna()
        records.append(
            {
                "variable": column,
                "missing_n": int(mask.sum()),
                "missing_rate": float(mask.mean()),
            }
        )
    return pd.DataFrame(records)
