from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data import SENSOR_COLUMNS

POWER_LAGS = [1, 2, 4, 8, 12, 24, 96]
SENSOR_LAGS = [0, 1, 4]
ROLLING_WINDOWS = [4, 12, 24]
IRRADIANCE_PREFIXES = ("tsi", "dni", "ghi")
WEATHER_PREFIXES = ("temperature", "pressure", "humidity")
RECENT_POWER_COLUMNS = ("power_current", "power_lag_1", "power_lag_2", "power_lag_4")
RECENT_POWER_ROLLING_WINDOWS = (4, 12)
OUTAGE_CONDITIONS = (
    "clean",
    "irradiance_outage",
    "weather_outage",
    "recent_power_outage",
    "combined_outage",
)


def _safe_fraction_missing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=frame.index)
    return frame[columns].isna().mean(axis=1)


def _refresh_missingness_descriptors(out: pd.DataFrame) -> pd.DataFrame:
    recent_power_cols = [
        c
        for c in ["power_current", "power_lag_1", "power_lag_2", "power_lag_4", "power_lag_8"]
        if c in out
    ]
    irradiance_cols = [
        c for c in out if c.startswith(("tsi", "dni", "ghi")) and "missing_fraction" not in c
    ]
    weather_cols = [
        c
        for c in out
        if c.startswith(("temperature", "pressure", "humidity")) and "missing_fraction" not in c
    ]
    out["recent_power_missing_fraction"] = _safe_fraction_missing(out, recent_power_cols)
    out["irradiance_missing_fraction"] = _safe_fraction_missing(out, irradiance_cols)
    out["weather_missing_fraction"] = _safe_fraction_missing(out, weather_cols)
    out["observed_context_fraction"] = 1.0 - out[
        ["recent_power_missing_fraction", "irradiance_missing_fraction", "weather_missing_fraction"]
    ].mean(axis=1)
    return out


def make_supervised(
    frame: pd.DataFrame,
    horizon_steps: int,
    lag_steps: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    daylight_hours: tuple[int, int] = (5, 20),
    nominal_capacity_mw: float = 50.0,
    include_current_power: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    lag_steps = lag_steps or POWER_LAGS
    rolling_windows = rolling_windows or ROLLING_WINDOWS

    x = pd.DataFrame(index=frame.index)
    if include_current_power:
        x["power_current"] = frame["power"]
    for lag in lag_steps:
        x[f"power_lag_{lag}"] = frame["power"].shift(lag)

    for sensor in SENSOR_COLUMNS:
        for lag in SENSOR_LAGS:
            name = sensor if lag == 0 else f"{sensor}_lag_{lag}"
            x[name] = frame[sensor].shift(lag)

    for window in rolling_windows:
        history = frame["power"].rolling(window, min_periods=max(2, window // 2))
        x[f"power_mean_{window}"] = history.mean()
        x[f"power_std_{window}"] = history.std()
        x[f"power_range_{window}"] = history.max() - history.min()
        x[f"power_absdiff_mean_{window}"] = (
            frame["power"].diff().abs().rolling(window, min_periods=max(2, window // 2)).mean()
        )

    hour = frame.index.hour + frame.index.minute / 60.0
    day = frame.index.dayofyear
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["doy_sin"] = np.sin(2 * np.pi * day / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * day / 365.25)
    x["transition_risk"] = ((hour < 9.0) | (hour > 16.0)).astype(float)
    x = _refresh_missingness_descriptors(x)

    target = frame["power"].shift(-horizon_steps)
    current_power = frame["power"]
    signed_ramp_fraction = (target - current_power) / nominal_capacity_mw
    ramp_fraction = signed_ramp_fraction.abs()

    meta = pd.DataFrame(index=frame.index)
    meta["target_timestamp"] = frame.index + pd.to_timedelta(horizon_steps * 15, unit="min")
    meta["current_power"] = current_power
    meta["signed_ramp_fraction"] = signed_ramp_fraction
    meta["ramp_fraction"] = ramp_fraction
    meta["ramp_direction"] = np.select(
        [signed_ramp_fraction > 0, signed_ramp_fraction < 0], ["up", "down"], default="flat"
    )
    meta["date"] = frame.index.normalize()
    meta["hour"] = frame.index.hour
    meta["base_daylight"] = frame.index.hour.to_series(index=frame.index).between(*daylight_hours)

    valid = target.notna() & current_power.notna() & meta["base_daylight"]
    x = x.loc[valid]
    target = target.loc[valid]
    meta = meta.loc[valid]

    exact_delta = meta["target_timestamp"] - meta.index.to_series(index=meta.index)
    expected_delta = pd.Timedelta(minutes=15 * horizon_steps)
    if not (exact_delta == expected_delta).all():
        raise RuntimeError("Fixed-time horizon invariant failed")
    return x, target, meta


def condition_mask_columns(columns: Iterable[str], condition: str) -> list[str]:
    """Return the exact, ordered feature mask used for a degradation condition."""

    if condition not in OUTAGE_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")

    ordered_columns = list(columns)
    irr = [
        c
        for c in ordered_columns
        if c.startswith(IRRADIANCE_PREFIXES) and "missing_fraction" not in c
    ]
    weather = [
        c
        for c in ordered_columns
        if c.startswith(WEATHER_PREFIXES) and "missing_fraction" not in c
    ]
    power_recent = [c for c in RECENT_POWER_COLUMNS if c in ordered_columns]
    short_power_rolls = [
        c
        for c in ordered_columns
        if c.startswith("power_")
        and c not in power_recent
        and any(c.endswith(f"_{w}") for w in RECENT_POWER_ROLLING_WINDOWS)
    ]

    if condition == "clean":
        return []
    if condition == "irradiance_outage":
        return irr
    if condition == "weather_outage":
        return weather
    if condition == "recent_power_outage":
        return power_recent + short_power_rolls
    return irr + weather + power_recent + short_power_rolls


def apply_condition(x: pd.DataFrame, condition: str) -> pd.DataFrame:
    out = x.copy()
    masked_columns = condition_mask_columns(out.columns, condition)
    if masked_columns:
        out[masked_columns] = np.nan

    return _refresh_missingness_descriptors(out)


def latest_available_power(x: pd.DataFrame) -> np.ndarray:
    candidates = [
        c for c in ["power_current", "power_lag_1", "power_lag_2", "power_lag_4", "power_lag_8", "power_lag_12"] if c in x
    ]
    if not candidates:
        raise ValueError("No power-history columns available")
    return x[candidates].bfill(axis=1).iloc[:, 0].to_numpy(dtype=float)
