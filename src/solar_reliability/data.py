from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[:pd.Timestamp(train_end)].copy()
    calibration = frame.loc[pd.Timestamp(calibration_start):pd.Timestamp(calibration_end)].copy()
    test = frame.loc[pd.Timestamp(test_start):pd.Timestamp(test_end)].copy()
    if train.empty or calibration.empty or test.empty:
        raise ValueError("One or more chronological splits are empty")
    if train.index.max() >= calibration.index.min():
        raise ValueError("Training and calibration periods overlap")
    if calibration.index.max() >= test.index.min():
        raise ValueError("Calibration and test periods overlap")
    return train, calibration, test


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
