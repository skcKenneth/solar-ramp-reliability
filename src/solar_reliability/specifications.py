from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .advanced import DESCRIPTOR_CONTEXT_COLUMNS, LOCAL_SCORE_N_JOBS, LOCAL_SCORE_NUM_LEAVES, LOCAL_SCORE_OBJECTIVE, LOCAL_SCORE_REG_LAMBDA, descriptor_frame
from .data import COLUMN_MAP, SENSOR_COLUMNS
from .features import condition_mask_columns, make_supervised

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
REQUIRED_RAW_COLUMNS = {"timestamp", "power", "tsi", "dni", "ghi", "temperature", "pressure"}
RAW_CHANNEL_FAMILIES = {
    "power": "recent_power", "tsi": "irradiance", "dni": "irradiance", "ghi": "irradiance",
    "temperature": "weather", "pressure": "weather", "humidity": "weather",
}
CONDITION_FAMILIES = {
    "irradiance_outage": ["irradiance"], "weather_outage": ["weather"],
    "recent_power_outage": ["recent_power"],
    "combined_outage": ["irradiance", "weather", "recent_power"],
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _probe_feature_columns(config: dict[str, Any]) -> list[str]:
    forecast = config["forecast"]
    lag_steps = [int(value) for value in forecast["lag_steps"]]
    rolling_windows = [int(value) for value in forecast["rolling_windows"]]
    horizons = [int(value) for value in forecast["horizons_steps"]]
    history = max([*lag_steps, *rolling_windows, 1])
    periods = history + max(horizons) + 32
    index = pd.date_range("2020-06-01 00:00:00", periods=periods, freq="15min")
    probe = pd.DataFrame(index=index)
    probe["power"] = np.linspace(10.0, 20.0, periods)
    probe["tsi"] = 500.0
    probe["dni"] = 400.0
    probe["ghi"] = 450.0
    probe["temperature"] = 25.0
    probe["pressure"] = 1000.0
    probe["humidity"] = 50.0
    x, _, _ = make_supervised(
        probe,
        horizon_steps=min(horizons),
        lag_steps=lag_steps,
        rolling_windows=rolling_windows,
        daylight_hours=(0, 23),
        nominal_capacity_mw=100.0,
    )
    if x.empty:
        raise RuntimeError("Feature-schema probe produced no eligible rows")
    return list(x.columns)


def _descriptor_records(feature_columns: list[str], conditions: list[str]) -> list[dict[str, Any]]:
    x = pd.DataFrame([{column: 0.25 for column in feature_columns}])
    descriptors = descriptor_frame(
        x,
        condition="clean",
        risk=np.array([0.2]),
        lower=np.array([10.0]),
        median=np.array([15.0]),
        upper=np.array([20.0]),
        capacity=100.0,
        conditions=conditions,
    )
    definitions = {
        "risk": "Forecast-time output of the training-fitted ramp-risk model.",
        "native_width": "Base upper-minus-lower interval width divided by nominal capacity.",
        "median_level": "Base median forecast divided by nominal capacity.",
        "recent_power_missing_fraction": "Missing fraction across recent power context after the requested mask.",
        "irradiance_missing_fraction": "Missing fraction across irradiance features after the requested mask.",
        "weather_missing_fraction": "Missing fraction across weather features after the requested mask.",
        "observed_context_fraction": "One minus the mean of the three family missing fractions.",
        "hour_sin": "Sine encoding of forecast-origin hour.",
        "hour_cos": "Cosine encoding of forecast-origin hour.",
        "transition_risk": "Indicator for forecast origins before 09:00 or after 16:00.",
    }
    records: list[dict[str, Any]] = []
    for column in descriptors.columns:
        if column.startswith("condition__"):
            definition = "One-hot indicator for the known evaluation degradation condition."
            source = "evaluation_condition"
        elif column in {"risk", "native_width", "median_level"}:
            definition = definitions[column]
            source = "forecast_model_output"
        else:
            definition = definitions[column]
            source = "forecast_origin_features"
        records.append(
            {
                "name": column,
                "definition": definition,
                "source": source,
                "available_at_forecast_time": True,
                "uses_future_target_or_ramp_label": False,
            }
        )
    expected_context = set(DESCRIPTOR_CONTEXT_COLUMNS)
    if not expected_context.issubset(descriptors.columns):
        raise RuntimeError("Descriptor probe disagrees with the live descriptor context constants")
    return records


def _provenance(root: Path, config_path: Path) -> dict[str, Any]:
    sources = [
        config_path,
        root / "scripts" / "generate_specifications.py",
        root / "src" / "solar_reliability" / "advanced.py",
        root / "src" / "solar_reliability" / "conformal.py",
        root / "src" / "solar_reliability" / "data.py",
        root / "src" / "solar_reliability" / "features.py",
        root / "src" / "solar_reliability" / "specifications.py",
    ]
    return {
        "generator": "scripts/generate_specifications.py",
        "generated_at": None,
        "deterministic": True,
        "sources": [
            {"path": _relative(path, root), "sha256": _sha256(path)}
            for path in sources
        ],
    }


def build_method_specification(
    config: dict[str, Any],
    feature_columns: list[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    alpha = float(config["project"]["alpha"])
    calibration = config["calibration"]
    conditions = [str(value) for value in config["conditions"]]
    fit_fraction = float(calibration["local_fit_fraction"])
    descriptor_records = _descriptor_records(feature_columns, conditions)
    return {
        "schema_version": "1.0.0",
        "artifact": "descriptor_conditioned_score_cqr_method_specification",
        "method_key": "robust_local_cqr",
        "display_name": "Descriptor-conditioned score CQR",
        "status": "machine_readable_method_definition_only",
        "provenance": provenance,
        "target_miscoverage_alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "nonconformity_score": {
            "formula": "max(lower - y, y - upper, 0)",
            "nonnegative": True,
            "semantics": "expansion-only CQR score",
        },
        "conditioning_model": {
            "type": "LightGBM quantile regressor for the nonconformity score",
            "explicit_distance_metric": None,
            "k_nearest_neighbors": None,
            "uses_distance_or_k_neighborhood": False,
            "objective": LOCAL_SCORE_OBJECTIVE,
            "quantile_alpha": 1.0 - alpha,
            "n_estimators": int(calibration["local_max_iter"]),
            "num_leaves": int(LOCAL_SCORE_NUM_LEAVES),
            "min_child_samples": int(calibration["local_min_samples_leaf"]),
            "learning_rate": float(calibration["local_learning_rate"]),
            "regularization": {"reg_lambda_l2": float(LOCAL_SCORE_REG_LAMBDA)},
            "n_jobs": int(LOCAL_SCORE_N_JOBS),
            "project_base_seed": int(config["project"]["seed"]),
            "random_state": "caller-supplied deterministic per-run seed",
        },
        "forecast_time_descriptor_schema": descriptor_records,
        "descriptor_column_order": [record["name"] for record in descriptor_records],
        "chronological_score_split": {
            "ordering_unit": "unique calibration forecast-origin timestamp",
            "sort_order": "ascending",
            "model_fit_fraction": fit_fraction,
            "finite_sample_adjustment_fraction": 1.0 - fit_fraction,
            "model_fit_block": "earliest timestamps",
            "finite_sample_adjustment_block": "latest timestamps",
            "timestamp_groups_kept_intact": True,
            "boundary_rule": "floor(n_unique_timestamps * model_fit_fraction), clamped to [1, n-1]",
            "minimum_unique_timestamps": 2,
        },
        "finite_sample_correction": {
            "residual": "observed_score - score_quantile_model_prediction",
            "sample": "latest 40% chronological adjustment block",
            "quantile_rank": "ceil((n + 1) * (1 - alpha)), bounded to [1, n]",
            "quantile_method": "higher order statistic after sorting finite residuals",
            "correction": "max(0, conformal_quantile(residual, alpha))",
            "test_score_quantile": "max(model_prediction + correction, 0)",
        },
        "interval_construction": {
            "base_interval_clipped_before_scoring": True,
            "lower": "base_lower - test_score_quantile",
            "upper": "base_upper + test_score_quantile",
            "clip_lower_to": 0.0,
            "clip_upper_to": "station nominal capacity MW",
            "post_clip_ordering": "lower=min(clipped_lower, clipped_upper); upper=max(clipped_lower, clipped_upper)",
        },
        "fallback_policy": {
            "alternative_calibrator": None,
            "silent_substitution": False,
            "behavior": "fail explicitly if the chronological split, model fit, or finite residual quantile is unavailable",
        },
        "future_information_exclusions": [
            "future target y",
            "future ramp magnitude",
            "future ramp indicator or direction",
            "oracle ramp-group label",
            "test nonconformity score",
        ],
    }


def _column_details(column: str) -> dict[str, Any]:
    sensor_match = re.fullmatch(r"(tsi|dni|ghi|temperature|pressure|humidity)(?:_lag_(\d+))?", column)
    if sensor_match:
        channel, lag = sensor_match.groups()
        return {
            "column": column,
            "family": RAW_CHANNEL_FAMILIES[channel],
            "base_channel": channel,
            "role": "current" if lag is None else "lag",
            "lag_steps": 0 if lag is None else int(lag),
        }
    if column == "power_current":
        return {
            "column": column,
            "family": "recent_power",
            "base_channel": "power",
            "role": "current",
            "lag_steps": 0,
        }
    power_lag = re.fullmatch(r"power_lag_(\d+)", column)
    if power_lag:
        return {
            "column": column,
            "family": "recent_power",
            "base_channel": "power",
            "role": "lag",
            "lag_steps": int(power_lag.group(1)),
        }
    rolling = re.fullmatch(r"power_(mean|std|range|absdiff_mean)_(\d+)", column)
    if rolling:
        return {
            "column": column,
            "family": "recent_power",
            "base_channel": "power",
            "role": "rolling_summary",
            "statistic": rolling.group(1),
            "window_steps": int(rolling.group(2)),
        }
    return {"column": column, "family": "other", "role": "other"}


def build_outage_specification(
    config: dict[str, Any],
    feature_columns: list[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    condition_records: list[dict[str, Any]] = []
    for condition in [str(value) for value in config["conditions"]]:
        masked = condition_mask_columns(feature_columns, condition)
        details = [_column_details(column) for column in masked]
        condition_records.append(
            {
                "condition": condition,
                "masked_feature_count": len(masked),
                "masked_columns": masked,
                "masked_column_details": details,
                "current_columns": [item["column"] for item in details if item["role"] == "current"],
                "lagged_columns": [item["column"] for item in details if item["role"] == "lag"],
                "rolling_summary_columns": [
                    item["column"] for item in details if item["role"] == "rolling_summary"
                ],
                "descriptor_update": "family missingness descriptors are recomputed after masking",
            }
        )
    return {
        "schema_version": "1.0.0",
        "artifact": "structured_sensor_outage_specification",
        "status": "machine_readable_ablation_definition_only",
        "provenance": provenance,
        "forecast_feature_schema": feature_columns,
        "conditions": condition_records,
        "target_invariant": {
            "target_name": str(config["data"]["target_column"]),
            "target_is_separate_from_feature_matrix": True,
            "target_is_masked": False,
            "target_is_interpolated": False,
            "configured_no_target_interpolation": bool(config["data"]["no_target_interpolation"]),
            "mask_function_scope": "forecast-origin feature matrix only",
        },
        "structured_ablation_limitations": [
            "These conditions are controlled deterministic feature-family ablation stress tests, not stochastic models of outage duration, dependence, or recovery.",
            "They do not model channel noise, bias, or drift, and they do not model all naturally occurring missingness.",
            "They do not establish that a real sensor outage would have the same joint missingness, persistence, or recovery process.",
            "The recent-power mask follows the live suffix rule exactly; it includes power_lag_12 and power rolling summaries with 4- and 12-step windows.",
            "Missingness descriptors remain available and are recomputed after each synthetic mask.",
            "Natural-missingness eligibility is audited separately and does not create performance claims.",
        ],
    }


def _read_raw_for_audit(path: Path, nominal_capacity_mw: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        return pd.DataFrame(), {
            "raw_file_exists": False,
            "raw_schema_complete": False,
            "raw_schema_missing_columns": sorted(REQUIRED_RAW_COLUMNS),
            "raw_channel_presence": {},
            "invalid_timestamp_rows": 0,
        }
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raw = pd.read_excel(path)
    renamed = raw.rename(columns=COLUMN_MAP)
    present = set(renamed.columns)
    missing_required = sorted(REQUIRED_RAW_COLUMNS.difference(present))
    channel_presence = {channel: channel in present for channel in ["power", *SENSOR_COLUMNS]}
    for column in ["timestamp", "power", *SENSOR_COLUMNS]:
        if column not in renamed:
            renamed[column] = np.nan
    parsed_timestamp = pd.to_datetime(renamed["timestamp"], errors="coerce")
    invalid_timestamp_rows = int(parsed_timestamp.isna().sum())
    frame = renamed[["timestamp", *SENSOR_COLUMNS, "power"]].copy()
    frame["timestamp"] = parsed_timestamp
    frame = frame.loc[frame["timestamp"].notna()]
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
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
    if not frame.empty:
        expected = pd.date_range(frame.index.min(), frame.index.max(), freq="15min")
        frame = frame.reindex(expected)
        frame.index.name = "timestamp"
    return frame, {
        "raw_file_exists": True,
        "raw_schema_complete": not missing_required,
        "raw_schema_missing_columns": missing_required,
        "raw_channel_presence": channel_presence,
        "invalid_timestamp_rows": invalid_timestamp_rows,
    }


def _clean_csv_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _station_audit_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Station audit must run before natural missingness audit: {path}")
    frame = pd.read_csv(path)
    required = {"site_id", "status", "exclusion_reason"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Station audit lacks columns: {sorted(missing)}")
    return {str(row["site_id"]): row.to_dict() for _, row in frame.iterrows()}


def _analysis_role(site_id: str, config: dict[str, Any]) -> str:
    project = config["project"]
    if site_id in {str(value) for value in project["development_sites"]}:
        return "development"
    if site_id in {str(value) for value in project["confirmatory_sites_preferred"]}:
        return "confirmatory_preferred"
    return "unassigned"


def _eligibility(
    station_status: str,
    station_reason: str,
    missing_n: int,
    model_evaluable_n: int,
    minimum_group_n: int,
) -> tuple[bool, str]:
    if station_status != "valid":
        suffix = f": {station_reason}" if station_reason else ""
        return False, f"station_{station_status}{suffix}"
    if model_evaluable_n == 0:
        return False, "no_model_evaluable_rows_with_future_target"
    if missing_n < minimum_group_n:
        return False, f"natural_missing_group_below_minimum_{minimum_group_n}"
    return True, "eligible_natural_missingness_group"


def build_natural_missingness_audit(
    root: Path,
    config: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    manifest_path = root / str(config["data"]["manifest"])
    raw_dir = root / str(config["data"]["raw_dir"])
    station_audit_path = root / str(config["data"]["station_audit_table"])
    manifest = pd.read_csv(manifest_path)
    station_audit = _station_audit_lookup(station_audit_path)
    horizons = [int(value) for value in config["forecast"]["horizons_steps"]]
    lag_steps = [int(value) for value in config["forecast"]["lag_steps"]]
    rolling_windows = [int(value) for value in config["forecast"]["rolling_windows"]]
    daylight_hours = tuple(int(value) for value in config["data"]["daylight_hours"])
    sampling_minutes = int(config["data"]["sampling_minutes"])
    minimum_group_n = int(config["calibration"]["min_group_n"])
    optional_channels = set(config["data"]["integrity_exclusion_rules"]["optional_sensor_columns"])
    condition_masks = {
        condition: condition_mask_columns(feature_columns, condition)
        for condition in CONDITION_FAMILIES
    }
    rows: list[dict[str, Any]] = []

    for manifest_row in manifest.sort_values("site_id").to_dict(orient="records"):
        site_id = str(manifest_row["site_id"])
        capacity = float(manifest_row["nominal_capacity_mw"])
        raw_path = raw_dir / str(manifest_row["filename"])
        station = station_audit.get(site_id, {})
        station_status = _clean_csv_cell(station.get("status")) or "not_audited"
        station_reason = _clean_csv_cell(station.get("exclusion_reason"))
        frame, raw_meta = _read_raw_for_audit(raw_path, capacity)
        role = _analysis_role(site_id, config)
        base = {
            "audit_schema_version": "1.0.0",
            "site_id": site_id,
            "analysis_role": role,
            "station_status": station_status,
            "station_exclusion_reason": station_reason,
            "nominal_capacity_mw": capacity,
            "raw_file": _relative(raw_path, root),
            "raw_file_exists": bool(raw_meta["raw_file_exists"]),
            "raw_file_sha256": _sha256(raw_path) if raw_path.exists() else "",
            "raw_schema_complete": bool(raw_meta["raw_schema_complete"]),
            "raw_schema_missing_columns": "|".join(raw_meta["raw_schema_missing_columns"]),
            "invalid_timestamp_rows": int(raw_meta["invalid_timestamp_rows"]),
            "target_masked": False,
            "target_interpolated": False,
            "minimum_group_n": minimum_group_n,
        }
        for horizon in horizons:
            if frame.empty:
                target_available = pd.Series(False, index=frame.index)
                model_evaluable = target_available.copy()
                x = pd.DataFrame(columns=feature_columns, index=frame.index)
            else:
                daylight = pd.Series(
                    frame.index.hour,
                    index=frame.index,
                ).between(*daylight_hours)
                future_target = frame["power"].shift(-horizon)
                target_available = daylight & future_target.notna()
                model_evaluable = target_available & frame["power"].notna()
                x, _, _ = make_supervised(
                    frame,
                    horizon_steps=horizon,
                    lag_steps=lag_steps,
                    rolling_windows=rolling_windows,
                    daylight_hours=daylight_hours,
                    nominal_capacity_mw=capacity,
                )
                if list(x.columns) != feature_columns:
                    raise RuntimeError(f"Feature schema drifted while auditing {site_id}")
                if not x.index.equals(frame.index[model_evaluable]):
                    raise RuntimeError(
                        f"Model-evaluable target mask disagrees with make_supervised for {site_id}, h={horizon}"
                    )
            target_n = int(target_available.sum())
            model_n = int(model_evaluable.sum())
            common = {
                **base,
                "horizon_steps": horizon,
                "horizon_minutes": horizon * sampling_minutes,
                "target_available_daylight_n": target_n,
                "model_evaluable_n": model_n,
            }

            for channel in ["power", *SENSOR_COLUMNS]:
                presence = bool(raw_meta["raw_channel_presence"].get(channel, False))
                values = frame[channel] if channel in frame else pd.Series(np.nan, index=frame.index)
                missing_target = values.isna() & target_available
                missing_model = values.isna() & model_evaluable
                rows.append(
                    {
                        **common,
                        "aggregation_level": "raw_channel",
                        "sensor_family": RAW_CHANNEL_FAMILIES[channel],
                        "group": channel,
                        "source_columns": channel,
                        "raw_source_columns_present": presence,
                        "structurally_absent_optional_channels": (
                            channel if channel in optional_channels and not presence else ""
                        ),
                        "missingness_definition": "raw current-time channel is missing after validity normalization",
                        "missing_n_among_target_available": int(missing_target.sum()),
                        "missing_rate_among_target_available": (
                            float(missing_target.sum() / target_n) if target_n else np.nan
                        ),
                        "missing_n_among_model_evaluable": int(missing_model.sum()),
                        "missing_rate_among_model_evaluable": (
                            float(missing_model.sum() / model_n) if model_n else np.nan
                        ),
                        "nonmissing_n_among_model_evaluable": int(model_n - missing_model.sum()),
                        "performance_eligible": False,
                        "confirmatory_performance_eligible": False,
                        "eligibility_reason": "descriptive_channel_rate_only",
                    }
                )

            for condition, exact_columns in condition_masks.items():
                absent_optional = sorted(
                    {
                        str(_column_details(column).get("base_channel"))
                        for column in exact_columns
                        if _column_details(column).get("base_channel") in optional_channels
                        and not raw_meta["raw_channel_presence"].get(
                            str(_column_details(column).get("base_channel")), False
                        )
                    }
                )
                natural_columns = [
                    column
                    for column in exact_columns
                    if str(_column_details(column).get("base_channel")) not in absent_optional
                ]
                natural_base_channels = {
                    str(_column_details(column).get("base_channel"))
                    for column in natural_columns
                }
                all_natural_sources_present = all(
                    bool(raw_meta["raw_channel_presence"].get(channel, False))
                    for channel in natural_base_channels
                )
                if x.empty or not natural_columns:
                    group_missing = pd.Series(False, index=x.index)
                else:
                    group_missing = x[natural_columns].isna().any(axis=1)
                missing_n = int(group_missing.sum())
                eligible, reason = _eligibility(
                    station_status,
                    station_reason,
                    missing_n,
                    model_n,
                    minimum_group_n,
                )
                rows.append(
                    {
                        **common,
                        "aggregation_level": "forecast_feature_family",
                        "sensor_family": "+".join(CONDITION_FAMILIES[condition]),
                        "group": condition.removesuffix("_outage"),
                        "source_columns": "|".join(natural_columns),
                        "raw_source_columns_present": all_natural_sources_present,
                        "structurally_absent_optional_channels": "|".join(absent_optional),
                        "missingness_definition": (
                            "any naturally missing forecast feature in the exact synthetic-outage mask; "
                            "structurally absent optional channels are excluded"
                        ),
                        "missing_n_among_target_available": np.nan,
                        "missing_rate_among_target_available": np.nan,
                        "missing_n_among_model_evaluable": missing_n,
                        "missing_rate_among_model_evaluable": (
                            float(missing_n / model_n) if model_n else np.nan
                        ),
                        "nonmissing_n_among_model_evaluable": model_n - missing_n,
                        "performance_eligible": eligible,
                        "confirmatory_performance_eligible": eligible and role == "confirmatory_preferred",
                        "eligibility_reason": reason,
                    }
                )

    columns = [
        "audit_schema_version",
        "site_id",
        "analysis_role",
        "station_status",
        "station_exclusion_reason",
        "nominal_capacity_mw",
        "raw_file",
        "raw_file_exists",
        "raw_file_sha256",
        "raw_schema_complete",
        "raw_schema_missing_columns",
        "invalid_timestamp_rows",
        "horizon_steps",
        "horizon_minutes",
        "aggregation_level",
        "sensor_family",
        "group",
        "source_columns",
        "raw_source_columns_present",
        "structurally_absent_optional_channels",
        "missingness_definition",
        "target_available_daylight_n",
        "model_evaluable_n",
        "missing_n_among_target_available",
        "missing_rate_among_target_available",
        "missing_n_among_model_evaluable",
        "missing_rate_among_model_evaluable",
        "nonmissing_n_among_model_evaluable",
        "minimum_group_n",
        "performance_eligible",
        "confirmatory_performance_eligible",
        "eligibility_reason",
        "target_masked",
        "target_interpolated",
    ]
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(
        ["site_id", "horizon_steps", "aggregation_level", "sensor_family", "group"],
        kind="stable",
    ).reset_index(drop=True)

def generate_specifications(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    root = PROJECT_ROOT if config_path == DEFAULT_CONFIG_PATH.resolve() else config_path.parent.parent
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    config = _load_yaml(config_path)
    if not bool(config["data"]["no_target_interpolation"]):
        raise ValueError("Specifications require no_target_interpolation: true")
    feature_columns = _probe_feature_columns(config)
    provenance = _provenance(root, config_path)
    method = build_method_specification(config, feature_columns, provenance)
    outage = build_outage_specification(config, feature_columns, provenance)
    natural = build_natural_missingness_audit(root, config, feature_columns)
    paths = {
        "method_json": output_dir / "method_specification.json",
        "outage_json": output_dir / "outage_specification.json",
        "natural_missingness_csv": output_dir / "natural_missingness_audit.csv",
    }
    _atomic_write_text(paths["method_json"], json.dumps(method, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(paths["outage_json"], json.dumps(outage, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(paths["natural_missingness_csv"], natural.to_csv(index=False, lineterminator="\n", float_format="%.12g"))
    return paths
