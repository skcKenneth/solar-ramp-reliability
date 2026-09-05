from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from solar_reliability.data import load_solar_data
from solar_reliability.features import condition_mask_columns, make_supervised
from solar_reliability.reproducibility import file_sha256, file_matches_sha256_receipt, job_seed_map, model_execution_source_sha256_candidates, runtime_environment
from solar_reliability.splits import leakage_safe_fold_masks
from prediction_metrics import aggregate_prediction_metrics, prepare_prediction_metric_frame, require_complete_method_condition_rows, require_full_availability

COMPLETION_MARKER_SCHEMA_VERSION = 2
NATURAL_MISSINGNESS_SCHEMA_VERSION = "1.0.0"
NATURAL_MISSINGNESS_STATUS = "post_run_descriptive_sensitivity"
NATURAL_MISSINGNESS_GROUPS = ("irradiance", "weather", "recent_power", "combined")
NATURAL_MISSINGNESS_RAMP_GROUPS = ("non_ramp", "ramp")


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return (
        upper
        - lower
        + (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def valid_stations(cfg: dict, *, root: Path = ROOT) -> list[str]:
    audit = pd.read_csv(root / cfg["data"]["station_audit_table"])
    development = set(map(str, cfg["project"].get("development_sites", [])))
    rows = audit[(audit["status"].astype(str) == "valid") & (~audit["site_id"].astype(str).isin(development))]
    return sorted(rows["site_id"].astype(str).tolist())


def capacity_map(cfg: dict, *, root: Path = ROOT) -> dict[str, float]:
    manifest = pd.read_csv(root / cfg["data"]["manifest"])
    return {str(row.site_id): float(row.nominal_capacity_mw) for row in manifest.itertuples(index=False)}


def _job_artifact_paths(
    cfg: dict,
    station: str,
    fold: str,
    horizon: int,
    *,
    root: Path,
) -> list[Path]:
    chunks = root / cfg["outputs"]["root"] / "stations" / station / "cache" / "chunks"
    stem = f"{fold}_h{int(horizon)}"
    return [
        chunks / f"predictions_{stem}.pkl.gz",
        chunks / f"run_{stem}.csv",
        chunks / f"calibration_{stem}.csv",
    ]


def validated_chunk_evidence(
    cfg: dict,
    *,
    expected_marker_environment: dict | None = None,
    root: Path = ROOT,
) -> tuple[list[tuple[str, str, int, Path]], pd.DataFrame]:
    """Require schema-v2 completion markers and hash-validated chunk triplets.

    Markers must match the current source, configuration, data, and runtime.
    """

    station_root = root / cfg["outputs"]["root"] / "stations"
    marker_root = root / cfg["outputs"]["completion_markers"]
    expected: list[tuple[str, str, int, Path]] = []
    expected_artifacts: set[str] = set()
    evidence_rows: list[dict] = []
    current_model_source_hashes = set(model_execution_source_sha256_candidates(root))
    marker_environment = (
        expected_marker_environment
        if expected_marker_environment is not None
        else runtime_environment()
    )
    for station in valid_stations(cfg, root=root):
        for fold in cfg["data"]["site1_calendar_folds"]:
            for horizon in cfg["forecast"]["horizons_steps"]:
                horizon = int(horizon)
                fold_name = str(fold["name"])
                artifacts = _job_artifact_paths(
                    cfg, station, fold_name, horizon, root=root
                )
                prediction_path = artifacts[0]
                expected.append((station, fold_name, horizon, prediction_path))
                expected_artifacts.update(path.relative_to(root).as_posix() for path in artifacts)

                marker_path = marker_root / f"{station}_{fold_name}_h{horizon}.json"
                if not marker_path.is_file():
                    raise RuntimeError(f"Missing corrected completion marker: {marker_path.relative_to(root)}")
                try:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Unreadable completion marker: {marker_path.relative_to(root)}") from exc
                expected_job = {
                    "station": station,
                    "fold": fold_name,
                    "horizon_steps": horizon,
                }
                if marker.get("schema_version") != COMPLETION_MARKER_SCHEMA_VERSION:
                    raise RuntimeError(f"Completion marker schema mismatch: {marker_path.relative_to(root)}")
                if marker.get("status") != "COMPLETE" or marker.get("mode") != "full":
                    raise RuntimeError(f"Completion marker is not a full successful run: {marker_path.relative_to(root)}")
                if marker.get("job") != expected_job:
                    raise RuntimeError(f"Completion marker job mismatch: {marker_path.relative_to(root)}")

                provenance = marker.get("provenance", {})
                for path_key, hash_key in (
                    ("root_config", "root_config_sha256"),
                    ("station_config", "station_config_sha256"),
                    ("data_path", "data_sha256"),
                ):
                    relative = provenance.get(path_key)
                    if not relative:
                        raise RuntimeError(f"Completion marker lacks {path_key}: {marker_path.relative_to(root)}")
                    current = root / str(relative)
                    if not file_matches_sha256_receipt(
                        current,
                        expected_sha256=provenance.get(hash_key),
                    ):
                        raise RuntimeError(f"Completion marker {hash_key} mismatch: {marker_path.relative_to(root)}")
                    if path_key == "root_config":
                        marker_cfg = yaml.safe_load(current.read_text(encoding="utf-8"))
                        if marker_cfg != cfg:
                            raise RuntimeError(
                                f"Completion marker root config does not match the analysis config: "
                                f"{marker_path.relative_to(root)}"
                            )
                if provenance.get("model_execution_source_sha256") not in current_model_source_hashes:
                    raise RuntimeError(
                        f"Completion marker model-execution source hash mismatch: "
                        f"{marker_path.relative_to(root)}"
                    )
                expected_seeds = job_seed_map(
                    int(cfg["project"]["seed"]), station, fold_name, horizon
                )
                if marker.get("seeds") != expected_seeds:
                    raise RuntimeError(f"Completion marker seed map mismatch: {marker_path.relative_to(root)}")
                if marker.get("environment") != marker_environment:
                    raise RuntimeError(f"Completion marker environment mismatch: {marker_path.relative_to(root)}")

                recorded_artifacts = {
                    item.get("path"): item for item in marker.get("artifacts", [])
                }
                expected_paths = {
                    path.relative_to(root).as_posix() for path in artifacts
                }
                if set(recorded_artifacts) != expected_paths:
                    raise RuntimeError(
                        f"Completion marker artifact set mismatch: {marker_path.relative_to(root)}"
                    )
                for path in artifacts:
                    relative = path.relative_to(root).as_posix()
                    record = recorded_artifacts.get(relative)
                    if (
                        not isinstance(record, dict)
                        or set(record) != {"path", "exists", "bytes", "sha256"}
                        or record.get("path") != relative
                        or record.get("exists") is not True
                        or not file_matches_sha256_receipt(
                            path,
                            expected_sha256=record.get("sha256"),
                            expected_bytes=record.get("bytes"),
                        )
                    ):
                        raise RuntimeError(
                            f"Completion marker artifact mismatch: {path.relative_to(root)}"
                        )

                split = marker.get("split", {})
                if any(split.get(key) != value for key, value in expected_job.items()):
                    raise RuntimeError(f"Completion marker split identity mismatch: {marker_path.relative_to(root)}")
                split_audit = split.get("split_audit", {})
                if set(split_audit) != {"train", "calibration", "test"}:
                    raise RuntimeError(f"Completion marker split audit is incomplete: {marker_path.relative_to(root)}")
                corrected_train = int(split_audit["train"]["rows"])
                corrected_calibration = int(split_audit["calibration"]["rows"])
                corrected_test = int(split_audit["test"]["rows"])
                train_after_subsample = int(split["train_n_after_subsample"])
                if train_after_subsample > corrected_train:
                    raise RuntimeError(f"Subsampled training count exceeds split count: {marker_path.relative_to(root)}")
                if int(split["calibration_n"]) != corrected_calibration or int(split["test_n"]) != corrected_test:
                    raise RuntimeError(f"Completion marker split counts disagree: {marker_path.relative_to(root)}")
                evidence_rows.append(
                    {
                        "station": station,
                        "fold": fold_name,
                        "horizon_steps": horizon,
                        "train_rows_pre_subsample": corrected_train,
                        "train_rows_after_subsample": train_after_subsample,
                        "calibration_rows": corrected_calibration,
                        "test_rows": corrected_test,
                        "marker_path": marker_path.relative_to(root).as_posix(),
                        "prediction_path": prediction_path.relative_to(root).as_posix(),
                    }
                )

    actual_artifacts: set[str] = set()
    if station_root.exists():
        for pattern in ("predictions_*.pkl.gz", "run_*.csv", "calibration_*.csv"):
            actual_artifacts.update(path.relative_to(root).as_posix() for path in station_root.rglob(pattern))
    missing = sorted(expected_artifacts - actual_artifacts)
    unexpected = sorted(actual_artifacts - expected_artifacts)
    if missing or unexpected:
        raise RuntimeError(
            "Corrected chunk set is not exact: "
            f"missing={missing[:1] or 'none'}, unexpected={unexpected[:1] or 'none'}"
        )
    evidence = pd.DataFrame(evidence_rows).sort_values(
        ["station", "fold", "horizon_steps"]
    ).reset_index(drop=True)
    return expected, evidence


def expected_chunk_paths(cfg: dict) -> list[tuple[str, str, int, Path]]:
    paths, _ = validated_chunk_evidence(cfg)
    return paths


def chunk_group_metrics(
    predictions: pd.DataFrame,
    *,
    station: str,
    fold: str,
    horizon: int,
    threshold: float,
    alpha: float,
    capacity: float,
    required_fully_usable_methods: Iterable[str] = (),
    ramp_fraction_override: np.ndarray | None = None,
    ramp_group_override: np.ndarray | None = None,
) -> pd.DataFrame:
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity must be finite and positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    required = {
        "method",
        "condition",
        "ramp_group",
        "ramp_fraction",
        "y",
        "median",
        "lower",
        "upper",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction chunk is missing columns: {sorted(missing)}")
    frame = predictions[sorted(required)].copy()
    if ramp_fraction_override is not None:
        exact_ramp_fraction = np.asarray(ramp_fraction_override, dtype=float)
        if exact_ramp_fraction.shape != (len(frame),) or not np.isfinite(exact_ramp_fraction).all():
            raise ValueError("ramp_fraction_override must contain one finite value per prediction row")
        frame["ramp_fraction"] = exact_ramp_fraction
    frame = prepare_prediction_metric_frame(
        frame,
        alpha=alpha,
        additional_finite_columns=("ramp_fraction",),
    )
    require_full_availability(
        frame,
        required_fully_usable_methods,
        context=f"{station}/{fold}/h{int(horizon)}",
    )
    if ramp_group_override is None:
        frame["ramp_group"] = np.where(
            frame["ramp_fraction"].to_numpy(dtype=float) >= threshold,
            "ramp",
            "non_ramp",
        )
    else:
        labels = np.asarray(ramp_group_override, dtype=str)
        if labels.shape != (len(frame),) or not set(labels).issubset({"ramp", "non_ramp"}):
            raise ValueError("ramp_group_override must contain one ramp/non_ramp label per row")
        frame["ramp_group"] = labels
    grouped = aggregate_prediction_metrics(
        frame,
        group_columns=["method", "condition", "ramp_group"],
        alpha=alpha,
        capacity=capacity,
    )
    grouped.insert(0, "station", station)
    grouped.insert(1, "fold", fold)
    grouped.insert(2, "horizon_steps", int(horizon))
    grouped.insert(3, "threshold", float(threshold))
    return grouped


def exact_ramp_fraction_by_origin(
    cfg: dict,
    station: str,
    horizon: int,
    *,
    root: Path = ROOT,
) -> pd.Series:
    """Reconstruct pre-serialization ramp fractions from the hash-validated raw data."""

    manifest = pd.read_csv(root / cfg["data"]["manifest"])
    station_rows = manifest[manifest["site_id"].astype(str) == str(station)]
    if len(station_rows) != 1:
        raise RuntimeError(f"Expected one manifest row for {station}, found {len(station_rows)}")
    station_row = station_rows.iloc[0]
    frame = load_solar_data(
        root / cfg["data"]["raw_dir"] / str(station_row["filename"]),
        nominal_capacity_mw=float(station_row["nominal_capacity_mw"]),
    )
    x, _, meta = make_supervised(
        frame,
        horizon_steps=int(horizon),
        lag_steps=list(cfg["forecast"]["lag_steps"]),
        rolling_windows=list(cfg["forecast"]["rolling_windows"]),
        daylight_hours=tuple(cfg["data"]["daylight_hours"]),
        nominal_capacity_mw=float(station_row["nominal_capacity_mw"]),
    )
    values = pd.Series(
        meta.loc[x.index, "ramp_fraction"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(x.index),
        name="exact_ramp_fraction",
    )
    if values.index.has_duplicates or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Exact ramp lookup is invalid for {station}/h{int(horizon)}")
    return values


def map_exact_ramp_fraction(
    predictions: pd.DataFrame,
    lookup: pd.Series,
    *,
    context: str,
) -> np.ndarray:
    if "timestamp" not in predictions:
        raise RuntimeError(f"Prediction chunk lacks timestamp needed for exact ramp reconstruction: {context}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(predictions["timestamp"], errors="coerce"))
    if timestamps.isna().any():
        raise RuntimeError(f"Prediction chunk has invalid timestamps: {context}")
    exact = lookup.reindex(timestamps).to_numpy(dtype=float)
    if not np.isfinite(exact).all():
        raise RuntimeError(f"Prediction timestamps are absent from exact ramp lookup: {context}")
    stored = predictions["ramp_fraction"].to_numpy(dtype=float)
    if not np.allclose(exact, stored, rtol=1e-6, atol=1e-7):
        mismatch = int((~np.isclose(exact, stored, rtol=1e-6, atol=1e-7)).sum())
        raise RuntimeError(
            f"Stored ramp fractions disagree with hash-validated raw data in {context}: "
            f"mismatched_rows={mismatch}"
        )
    return exact


def _selection_sha256(timestamps: pd.DatetimeIndex) -> str:
    canonical = "\n".join(
        pd.Timestamp(value).isoformat() for value in timestamps.sort_values()
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def natural_missingness_chunk_performance(
    predictions: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    target: pd.Series,
    meta: pd.DataFrame,
    audit_groups: pd.DataFrame,
    *,
    expected_test_origins: pd.DatetimeIndex,
    station: str,
    fold: str,
    horizon: int,
    horizon_minutes: int,
    capacity: float,
    alpha: float,
    minimum_group_n: int,
    methods: Iterable[str],
    baseline_method: str,
    nominal_ramp_threshold: float,
    raw_file: str,
    raw_file_sha256: str,
) -> pd.DataFrame:
    """Measure naturally missing clean-test cells without refitting any model.

    The mask is reconstructed from the hash-pinned raw feature matrix.  It is
    then joined one-to-one to preserved clean-condition predictions by origin
    and target timestamps.  Metrics remain blank unless the exact
    station/fold/horizon/family/ramp cell reaches the frozen minimum size.
    """

    methods = tuple(map(str, methods))
    if not methods or baseline_method not in methods or len(set(methods)) != len(methods):
        raise ValueError("Natural-missingness methods must be unique and include the baseline")
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity must be finite and positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if int(minimum_group_n) <= 0:
        raise ValueError("minimum_group_n must be positive")
    prediction_columns = {
        "fold",
        "timestamp",
        "target_timestamp",
        "horizon_steps",
        "horizon_minutes",
        "condition",
        "method",
        "ramp_group",
        "y",
        "median",
        "lower",
        "upper",
    }
    missing_prediction_columns = sorted(prediction_columns.difference(predictions.columns))
    if missing_prediction_columns:
        raise ValueError(
            f"Prediction chunk lacks natural-missingness columns: {missing_prediction_columns}"
        )
    audit_columns = {
        "group",
        "source_columns",
        "structurally_absent_optional_channels",
        "missing_n_among_model_evaluable",
        "confirmatory_performance_eligible",
        "eligibility_reason",
    }
    missing_audit_columns = sorted(audit_columns.difference(audit_groups.columns))
    if missing_audit_columns:
        raise ValueError(
            f"Natural-missingness audit lacks columns: {missing_audit_columns}"
        )
    if set(audit_groups["group"].astype(str)) != set(NATURAL_MISSINGNESS_GROUPS):
        raise RuntimeError(
            f"Natural-missingness group set changed for {station}/h{int(horizon)}"
        )
    if feature_matrix.index.has_duplicates or target.index.has_duplicates or meta.index.has_duplicates:
        raise RuntimeError("Supervised natural-missingness inputs contain duplicate origins")
    if not feature_matrix.index.equals(target.index) or not feature_matrix.index.equals(meta.index):
        raise RuntimeError("Supervised feature, target, and metadata origins are not aligned")
    if not {"target_timestamp", "ramp_fraction"}.issubset(meta.columns):
        raise RuntimeError("Supervised metadata lacks target_timestamp or ramp_fraction")

    primary = predictions[
        predictions["condition"].astype(str).eq("clean")
        & predictions["method"].astype(str).isin(methods)
    ].copy()
    if primary.empty:
        raise RuntimeError(f"No clean primary predictions for {station}/{fold}/h{int(horizon)}")
    if not primary["fold"].astype(str).eq(str(fold)).all():
        raise RuntimeError(f"Prediction fold identity changed for {station}/{fold}/h{int(horizon)}")
    if not primary["horizon_steps"].astype(int).eq(int(horizon)).all() or not primary[
        "horizon_minutes"
    ].astype(int).eq(int(horizon_minutes)).all():
        raise RuntimeError(f"Prediction horizon identity changed for {station}/{fold}/h{int(horizon)}")
    primary["timestamp"] = pd.to_datetime(primary["timestamp"], errors="coerce")
    primary["target_timestamp"] = pd.to_datetime(
        primary["target_timestamp"], errors="coerce"
    )
    if primary[["timestamp", "target_timestamp"]].isna().any().any():
        raise RuntimeError(f"Prediction timestamps are invalid for {station}/{fold}/h{int(horizon)}")

    by_method: dict[str, pd.DataFrame] = {}
    for method in methods:
        method_frame = primary[primary["method"].astype(str).eq(method)].sort_values(
            "timestamp", kind="stable"
        ).reset_index(drop=True)
        if method_frame.empty or method_frame["timestamp"].duplicated().any():
            raise RuntimeError(
                f"Natural-missingness predictions are absent or duplicated for "
                f"{station}/{fold}/h{int(horizon)}/{method}"
            )
        numeric = method_frame[["y", "median", "lower", "upper"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise RuntimeError(
                f"Natural-missingness primary predictions are unavailable for "
                f"{station}/{fold}/h{int(horizon)}/{method}"
            )
        if (numeric["upper"] < numeric["lower"]).any():
            raise RuntimeError(
                f"Natural-missingness interval endpoints are reversed for "
                f"{station}/{fold}/h{int(horizon)}/{method}"
            )
        method_frame[["y", "median", "lower", "upper"]] = numeric
        by_method[method] = method_frame

    baseline = by_method[baseline_method]
    origins = pd.DatetimeIndex(baseline["timestamp"])
    expected_test_origins = pd.DatetimeIndex(
        pd.to_datetime(expected_test_origins, errors="coerce")
    ).sort_values()
    if expected_test_origins.hasnans or expected_test_origins.has_duplicates:
        raise RuntimeError(
            f"Expected corrected test origins are invalid for "
            f"{station}/{fold}/h{int(horizon)}"
        )
    if not origins.equals(expected_test_origins):
        missing = expected_test_origins.difference(origins)
        unexpected = origins.difference(expected_test_origins)
        raise RuntimeError(
            f"Predictions do not match the exact corrected test origin set for "
            f"{station}/{fold}/h{int(horizon)}: missing={len(missing)} "
            f"(first={missing[:1].tolist()}), unexpected={len(unexpected)} "
            f"(first={unexpected[:1].tolist()})"
        )
    missing_origins = origins.difference(feature_matrix.index)
    if len(missing_origins):
        raise RuntimeError(
            f"Corrected prediction origins are absent from raw-feature reconstruction for "
            f"{station}/{fold}/h{int(horizon)}: {missing_origins[:1].tolist()}"
        )
    expected_targets = pd.DatetimeIndex(meta.loc[origins, "target_timestamp"])
    observed_targets = pd.DatetimeIndex(baseline["target_timestamp"])
    if not expected_targets.equals(observed_targets):
        raise RuntimeError(
            f"Corrected target timestamps do not join exactly to raw data for "
            f"{station}/{fold}/h{int(horizon)}"
        )
    expected_y = target.loc[origins].to_numpy(dtype=float)
    expected_ramp_group = np.where(
        meta.loc[origins, "ramp_fraction"].to_numpy(dtype=float)
        >= float(nominal_ramp_threshold),
        "ramp",
        "non_ramp",
    )
    for method, method_frame in by_method.items():
        if not pd.DatetimeIndex(method_frame["timestamp"]).equals(origins):
            raise RuntimeError(
                f"Primary method origin sets differ for {station}/{fold}/h{int(horizon)}/{method}"
            )
        if not pd.DatetimeIndex(method_frame["target_timestamp"]).equals(expected_targets):
            raise RuntimeError(
                f"Primary method target timestamps differ for "
                f"{station}/{fold}/h{int(horizon)}/{method}"
            )
        if not np.allclose(
            method_frame["y"].to_numpy(dtype=float),
            expected_y,
            rtol=1e-6,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"Primary method targets disagree with hash-pinned raw data for "
                f"{station}/{fold}/h{int(horizon)}/{method}"
            )
        if not np.array_equal(
            method_frame["ramp_group"].astype(str).to_numpy(), expected_ramp_group
        ):
            raise RuntimeError(
                f"Primary method ramp labels disagree with hash-pinned raw data for "
                f"{station}/{fold}/h{int(horizon)}/{method}"
            )

    rows: list[dict] = []
    aligned_features = feature_matrix.loc[origins]
    for audit_row in audit_groups.sort_values("group", kind="stable").itertuples(index=False):
        group = str(audit_row.group)
        source_columns = tuple(
            column for column in str(audit_row.source_columns).split("|") if column
        )
        absent_optional = tuple(
            channel
            for channel in str(
                audit_row.structurally_absent_optional_channels
            ).split("|")
            if channel and channel.lower() != "nan"
        )
        if len(set(absent_optional)) != len(absent_optional):
            raise RuntimeError(
                f"Natural-missingness audit repeats a structurally absent channel "
                f"for {station}/h{int(horizon)}/{group}"
            )
        expected_source_columns = tuple(
            column
            for column in condition_mask_columns(
                aligned_features.columns, f"{group}_outage"
            )
            if not any(
                column == channel or column.startswith(f"{channel}_")
                for channel in absent_optional
            )
        )
        if len(set(source_columns)) != len(source_columns) or set(source_columns) != set(
            expected_source_columns
        ):
            raise RuntimeError(
                f"Natural-missingness audit mask differs from the implemented outage mask "
                f"for {station}/h{int(horizon)}/{group}: "
                f"audit={list(source_columns)}, implemented={list(expected_source_columns)}"
            )
        absent_columns = sorted(set(source_columns).difference(aligned_features.columns))
        if absent_columns:
            raise RuntimeError(
                f"Natural-missingness source columns are absent for "
                f"{station}/h{int(horizon)}/{group}: {absent_columns}"
            )
        natural_mask = (
            aligned_features[list(source_columns)].isna().any(axis=1).to_numpy()
            if source_columns
            else np.zeros(len(aligned_features), dtype=bool)
        )
        for ramp_group in NATURAL_MISSINGNESS_RAMP_GROUPS:
            selected = natural_mask & (expected_ramp_group == ramp_group)
            selected_timestamps = origins[selected]
            attempted_n = int(selected.sum())
            eligible = attempted_n >= int(minimum_group_n)
            if attempted_n == 0:
                reason = "no_natural_missingness_rows_in_corrected_test_cell"
            elif not eligible:
                reason = f"natural_missing_group_below_minimum_{int(minimum_group_n)}"
            else:
                reason = "eligible_corrected_test_cell"
            selection_hash = _selection_sha256(selected_timestamps)
            for method in methods:
                method_frame = by_method[method]
                cell = method_frame.loc[selected]
                if len(cell) != attempted_n:
                    raise RuntimeError("Natural-missingness selection lost prediction rows")
                coverage = width = score = undercoverage = np.nan
                if eligible:
                    y = cell["y"].to_numpy(dtype=float)
                    lower = cell["lower"].to_numpy(dtype=float)
                    upper = cell["upper"].to_numpy(dtype=float)
                    coverage = float(np.mean((lower <= y) & (y <= upper)))
                    width = float(np.mean(upper - lower) / capacity)
                    score = float(np.mean(interval_score(y, lower, upper, alpha)) / capacity)
                    undercoverage = max(0.0, 1.0 - alpha - coverage)
                rows.append(
                    {
                        "schema_version": NATURAL_MISSINGNESS_SCHEMA_VERSION,
                        "status": NATURAL_MISSINGNESS_STATUS,
                        "station": str(station),
                        "fold": str(fold),
                        "horizon_steps": int(horizon),
                        "horizon_minutes": int(horizon_minutes),
                        "natural_missingness_group": group,
                        "source_columns": "|".join(source_columns),
                        "structurally_absent_optional_channels": "|".join(
                            absent_optional
                        ),
                        "condition": "clean",
                        "ramp_group": ramp_group,
                        "method": method,
                        "attempted_n": attempted_n,
                        "available_n": attempted_n,
                        "unavailable_n": 0,
                        "availability_rate": 1.0 if attempted_n else np.nan,
                        "nominal_coverage": 1.0 - alpha,
                        "minimum_group_n": int(minimum_group_n),
                        "performance_eligible": bool(eligible),
                        "exploratory_only": bool(not eligible),
                        "analysis_role": (
                            "post_run_descriptive_performance"
                            if eligible
                            else "exploratory_only_no_performance_metrics"
                        ),
                        "eligibility_reason": reason,
                        "coverage": coverage,
                        "undercoverage_error": undercoverage,
                        "mean_normalized_width": width,
                        "normalized_interval_score": score,
                        "natural_timestamp_set_sha256": selection_hash,
                        "whole_record_missing_n": int(
                            audit_row.missing_n_among_model_evaluable
                        ),
                        "whole_record_count_eligible": bool(
                            audit_row.confirmatory_performance_eligible
                        ),
                        "whole_record_eligibility_reason": str(
                            audit_row.eligibility_reason
                        ),
                        "target_masked": False,
                        "target_interpolated": False,
                        "aggregation_unit": (
                            "station_fold_horizon_natural_family_ramp_cell"
                        ),
                        "performance_scope": "corrected_test_predictions_only",
                        "interpretation": (
                            "strictly_post_run_descriptive_sensitivity_not_confirmatory"
                        ),
                        "raw_file": str(raw_file),
                        "raw_file_sha256": str(raw_file_sha256),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [
            "station",
            "fold",
            "horizon_steps",
            "natural_missingness_group",
            "ramp_group",
            "method",
        ],
        kind="stable",
    ).reset_index(drop=True)


def natural_missingness_summary(performance: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate feature-family masks, then pool eligible folds descriptively."""

    required = {
        "station",
        "fold",
        "horizon_steps",
        "horizon_minutes",
        "ramp_group",
        "method",
        "attempted_n",
        "nominal_coverage",
        "performance_eligible",
        "coverage",
        "undercoverage_error",
        "mean_normalized_width",
        "normalized_interval_score",
        "natural_timestamp_set_sha256",
    }
    missing = sorted(required.difference(performance.columns))
    if missing:
        raise ValueError(f"Natural-missingness performance lacks summary columns: {missing}")
    eligible = performance[performance["performance_eligible"].astype(bool)].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "station",
                "horizon_steps",
                "horizon_minutes",
                "ramp_group",
                "method",
                "eligible_unique_cells",
                "prediction_rows",
                "coverage",
                "undercoverage_error",
                "mean_normalized_width",
                "normalized_interval_score",
            ]
        )
    metric_columns = [
        "coverage",
        "undercoverage_error",
        "mean_normalized_width",
        "normalized_interval_score",
    ]
    if not np.isfinite(eligible[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Eligible natural-missingness rows contain blank/non-finite metrics")
    unique = eligible.drop_duplicates(
        [
            "station",
            "fold",
            "horizon_steps",
            "ramp_group",
            "method",
            "natural_timestamp_set_sha256",
        ]
    )
    rows: list[dict] = []
    keys = ["station", "horizon_steps", "horizon_minutes", "ramp_group", "method"]
    for key, group in unique.groupby(keys, sort=True):
        weights = group["attempted_n"].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Eligible natural-missingness weights must be finite and positive")
        row = dict(zip(keys, key))
        row["eligible_unique_cells"] = int(len(group))
        row["prediction_rows"] = int(weights.sum())
        for column in (
            "coverage",
            "mean_normalized_width",
            "normalized_interval_score",
        ):
            row[column] = float(
                np.average(group[column].to_numpy(dtype=float), weights=weights)
            )
        nominal_values = group["nominal_coverage"].astype(float).unique()
        if len(nominal_values) != 1 or not np.isfinite(nominal_values[0]):
            raise ValueError("Natural-missingness nominal coverage changed within a summary cell")
        row["undercoverage_error"] = max(
            0.0, float(nominal_values[0]) - row["coverage"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def build_natural_missingness_performance(
    cfg: dict,
    chunk_paths: list[tuple[str, str, int, Path]],
    output: Path,
    *,
    root: Path = ROOT,
) -> tuple[pd.DataFrame, list[Path]]:
    audit_path = output / "natural_missingness_audit.csv"
    if not audit_path.is_file():
        raise RuntimeError(
            f"Natural-missingness audit must exist before performance analysis: "
            f"{audit_path.relative_to(root)}"
        )
    audit = pd.read_csv(audit_path)
    feature_audit = audit[
        audit["aggregation_level"].astype(str).eq("forecast_feature_family")
    ].copy()
    stations = valid_stations(cfg, root=root)
    horizons = [int(value) for value in cfg["forecast"]["horizons_steps"]]
    expected_audit_keys = {
        (station, horizon, group)
        for station in stations
        for horizon in horizons
        for group in NATURAL_MISSINGNESS_GROUPS
    }
    audit_keys = set(
        feature_audit[
            feature_audit["site_id"].astype(str).isin(stations)
        ][["site_id", "horizon_steps", "group"]]
        .assign(horizon_steps=lambda frame: frame["horizon_steps"].astype(int))
        .itertuples(index=False, name=None)
    )
    if audit_keys != expected_audit_keys:
        raise RuntimeError(
            "Natural-missingness audit does not cover the exact corrected station/horizon/group topology"
        )
    manifest = pd.read_csv(root / cfg["data"]["manifest"]).copy()
    manifest["site_id"] = manifest["site_id"].astype(str)
    capacities = capacity_map(cfg, root=root)
    methods = [
        str(cfg["methods"]["primary_baseline"]),
        *map(str, cfg["methods"]["primary_deployable_candidates"]),
    ]
    fold_specs = list(cfg["data"]["site1_calendar_folds"])
    fold_by_name = {str(item["name"]): item for item in fold_specs}
    if len(fold_by_name) != len(fold_specs):
        raise RuntimeError("Calendar fold names must be unique")
    rows: list[pd.DataFrame] = []
    supervised: dict[tuple[str, int], tuple[pd.DataFrame, pd.Series, pd.DataFrame, str, str]] = {}
    for station, fold, horizon, prediction_path in chunk_paths:
        key = (str(station), int(horizon))
        if key not in supervised:
            manifest_row = manifest[manifest["site_id"].eq(str(station))]
            if len(manifest_row) != 1:
                raise RuntimeError(f"Expected one data manifest row for {station}")
            record = manifest_row.iloc[0]
            raw_path = root / cfg["data"]["raw_dir"] / str(record["filename"])
            raw_hash = file_sha256(raw_path)
            audit_subset = feature_audit[
                feature_audit["site_id"].astype(str).eq(str(station))
                & feature_audit["horizon_steps"].astype(int).eq(int(horizon))
            ]
            recorded_hashes = set(audit_subset["raw_file_sha256"].astype(str))
            recorded_paths = set(audit_subset["raw_file"].astype(str))
            expected_relative = raw_path.relative_to(root).as_posix()
            if (
                len(recorded_hashes) != 1
                or recorded_paths != {expected_relative}
                or not file_matches_sha256_receipt(
                    raw_path,
                    expected_sha256=next(iter(recorded_hashes), ""),
                )
            ):
                raise RuntimeError(
                    f"Natural-missingness audit raw provenance is stale for {station}/h{int(horizon)}"
                )
            raw = load_solar_data(
                raw_path,
                nominal_capacity_mw=float(record["nominal_capacity_mw"]),
            )
            x, y, meta = make_supervised(
                raw,
                horizon_steps=int(horizon),
                lag_steps=list(cfg["forecast"]["lag_steps"]),
                rolling_windows=list(cfg["forecast"]["rolling_windows"]),
                daylight_hours=tuple(cfg["data"]["daylight_hours"]),
                nominal_capacity_mw=float(record["nominal_capacity_mw"]),
            )
            supervised[key] = (x, y, meta, expected_relative, raw_hash)
        x, y, meta, raw_file, raw_hash = supervised[key]
        audit_subset = feature_audit[
            feature_audit["site_id"].astype(str).eq(str(station))
            & feature_audit["horizon_steps"].astype(int).eq(int(horizon))
        ]
        if str(fold) not in fold_by_name:
            raise RuntimeError(f"Prediction chunk names unknown calendar fold: {fold}")
        masks = leakage_safe_fold_masks(
            x.index,
            meta["target_timestamp"],
            fold_by_name[str(fold)],
            horizon_steps=int(horizon),
            sampling_minutes=int(cfg["data"]["sampling_minutes"]),
        )
        expected_test_origins = pd.DatetimeIndex(x.index[masks.test.to_numpy()])
        rows.append(
            natural_missingness_chunk_performance(
                pd.read_pickle(prediction_path, compression="gzip"),
                x,
                y,
                meta,
                audit_subset,
                expected_test_origins=expected_test_origins,
                station=str(station),
                fold=str(fold),
                horizon=int(horizon),
                horizon_minutes=int(horizon) * int(cfg["data"]["sampling_minutes"]),
                capacity=capacities[str(station)],
                alpha=float(cfg["project"]["alpha"]),
                minimum_group_n=int(cfg["calibration"]["min_group_n"]),
                methods=methods,
                baseline_method=str(cfg["methods"]["primary_baseline"]),
                nominal_ramp_threshold=float(
                    cfg["forecast"]["ramp_threshold_capacity_fraction"][str(horizon)]
                ),
                raw_file=raw_file,
                raw_file_sha256=raw_hash,
            )
        )
    performance = pd.concat(rows, ignore_index=True).sort_values(
        [
            "station",
            "fold",
            "horizon_steps",
            "natural_missingness_group",
            "ramp_group",
            "method",
        ],
        kind="stable",
    ).reset_index(drop=True)
    expected_rows = (
        len(stations)
        * len(cfg["data"]["site1_calendar_folds"])
        * len(horizons)
        * len(NATURAL_MISSINGNESS_GROUPS)
        * len(NATURAL_MISSINGNESS_RAMP_GROUPS)
        * len(methods)
    )
    if len(performance) != expected_rows:
        raise RuntimeError(
            f"Natural-missingness performance row count changed: "
            f"expected={expected_rows}, observed={len(performance)}"
        )
    metric_columns = [
        "coverage",
        "undercoverage_error",
        "mean_normalized_width",
        "normalized_interval_score",
    ]
    eligible = performance["performance_eligible"].astype(bool)
    if performance.loc[~eligible, metric_columns].notna().any().any():
        raise RuntimeError("Ineligible natural-missingness cells contain performance metrics")
    if not np.isfinite(performance.loc[eligible, metric_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Eligible natural-missingness cells lack performance metrics")
    csv_path = output / "natural_missingness_performance.csv"
    atomic_text(
        csv_path,
        performance.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
    )
    return performance, [csv_path]


def collect_threshold_metrics(
    cfg: dict,
    *,
    chunk_paths: list[tuple[str, str, int, Path]] | None = None,
    root: Path = ROOT,
) -> pd.DataFrame:
    alpha = float(cfg["project"]["alpha"])
    capacities = capacity_map(cfg, root=root)
    rows: list[pd.DataFrame] = []
    if chunk_paths is None:
        chunk_paths, _ = validated_chunk_evidence(cfg, root=root)
    expected_methods = set(map(str, cfg["methods"]["full_set"]))
    expected_conditions = set(map(str, cfg["conditions"]))
    decision_methods = [
        str(cfg["methods"]["primary_baseline"]),
        *map(str, cfg["methods"]["primary_deployable_candidates"]),
    ]
    exact_ramp_lookups: dict[tuple[str, int], pd.Series] = {}
    for station, fold, horizon, path in chunk_paths:
        predictions = pd.read_pickle(path, compression="gzip")
        observed_methods = set(predictions["method"].astype(str)) if "method" in predictions else set()
        observed_conditions = (
            set(predictions["condition"].astype(str)) if "condition" in predictions else set()
        )
        if observed_methods != expected_methods:
            raise RuntimeError(
                f"Prediction chunk method set mismatch in {path.relative_to(root)}: "
                f"expected={sorted(expected_methods)}, observed={sorted(observed_methods)}"
            )
        if observed_conditions != expected_conditions:
            raise RuntimeError(
                f"Prediction chunk condition set mismatch in {path.relative_to(root)}: "
                f"expected={sorted(expected_conditions)}, observed={sorted(observed_conditions)}"
            )
        require_complete_method_condition_rows(
            predictions,
            expected_methods=expected_methods,
            expected_conditions=expected_conditions,
            baseline_method=str(cfg["methods"]["primary_baseline"]),
            context=path.relative_to(root).as_posix(),
        )
        thresholds = [float(v) for v in cfg["forecast"]["ramp_sensitivity"][str(horizon)]]
        nominal = float(cfg["forecast"]["ramp_threshold_capacity_fraction"][str(horizon)])
        if nominal not in thresholds:
            thresholds.append(nominal)
        lookup_key = (station, int(horizon))
        if lookup_key not in exact_ramp_lookups:
            exact_ramp_lookups[lookup_key] = exact_ramp_fraction_by_origin(
                cfg, station, horizon, root=root
            )
        exact_ramp_fraction = map_exact_ramp_fraction(
            predictions,
            exact_ramp_lookups[lookup_key],
            context=path.relative_to(root).as_posix(),
        )
        for threshold in sorted(set(thresholds)):
            if np.isclose(threshold, nominal, rtol=0.0, atol=1e-12):
                ramp_group = predictions["ramp_group"].astype(str).to_numpy()
                reconstructed_group = np.where(
                    exact_ramp_fraction >= nominal, "ramp", "non_ramp"
                )
                if not np.array_equal(ramp_group, reconstructed_group):
                    mismatch = int((ramp_group != reconstructed_group).sum())
                    raise RuntimeError(
                        f"Saved nominal ramp labels disagree with exact raw-data reconstruction in "
                        f"{path.relative_to(root).as_posix()}: mismatched_rows={mismatch}"
                    )
            else:
                ramp_group = np.where(
                    exact_ramp_fraction >= threshold, "ramp", "non_ramp"
                )
            rows.append(
                chunk_group_metrics(
                    predictions,
                    station=station,
                    fold=fold,
                    horizon=horizon,
                    threshold=threshold,
                    alpha=alpha,
                    capacity=capacities[station],
                    required_fully_usable_methods=decision_methods,
                    ramp_fraction_override=exact_ramp_fraction,
                    ramp_group_override=ramp_group,
                )
            )
    return pd.concat(rows, ignore_index=True)


def validate_nominal_group_metrics(
    group_metrics: pd.DataFrame,
    cfg: dict,
    *,
    root: Path = ROOT,
) -> None:
    """Fail closed if Phase-6 nominal groups diverge from corrected aggregation."""

    authoritative_path = root / cfg["outputs"]["tables"] / "conditional_metrics.csv"
    if not authoritative_path.is_file():
        raise RuntimeError(
            f"Missing authoritative corrected conditional metrics: {authoritative_path.relative_to(root)}"
        )
    authoritative = pd.read_csv(authoritative_path)
    nominal = {
        int(key): float(value)
        for key, value in cfg["forecast"]["ramp_threshold_capacity_fraction"].items()
    }
    derived = group_metrics[
        np.isclose(
            group_metrics["threshold"].to_numpy(dtype=float),
            group_metrics["horizon_steps"].map(nominal).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
    ].drop(columns=["threshold"])
    keys = ["station", "fold", "horizon_steps", "method", "condition", "ramp_group"]
    common = [column for column in derived.columns if column in authoritative.columns]
    compare_columns = keys + [column for column in common if column not in keys]
    left = derived[compare_columns].sort_values(keys).reset_index(drop=True)
    right = authoritative[compare_columns].sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "Nominal-threshold Phase-6 metrics diverge from authoritative corrected "
            f"conditional metrics ({authoritative_path.relative_to(root)}): {exc}"
        ) from exc


def station_first_tradeoffs(group_metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    min_n = int(cfg["calibration"]["min_group_n"])
    nominal = {int(k): float(v) for k, v in cfg["forecast"]["ramp_threshold_capacity_fraction"].items()}
    eligible = group_metrics[group_metrics["n"] >= min_n].copy()
    eligible = eligible[
        np.isclose(
            eligible["threshold"].to_numpy(dtype=float),
            eligible["horizon_steps"].map(nominal).to_numpy(dtype=float),
        )
    ]
    station_rows: list[dict] = []
    strata: list[tuple[str, Iterable[str]]] = [
        ("all", ("ramp", "non_ramp")),
        ("ramp", ("ramp",)),
        ("non_ramp", ("non_ramp",)),
    ]
    for (station, horizon, method), method_group in eligible.groupby(
        ["station", "horizon_steps", "method"], observed=True, sort=True
    ):
        for stratum, labels in strata:
            group = method_group[method_group["ramp_group"].astype(str).isin(labels)]
            if group.empty:
                continue
            station_rows.append(
                {
                    "station": str(station),
                    "horizon_steps": int(horizon),
                    "method": str(method),
                    "stratum": stratum,
                    "eligible_groups": int(len(group)),
                    "eligible_prediction_rows": int(group["n"].sum()),
                    "attempted_prediction_rows": int(group["attempted_n"].sum()),
                    "unavailable_prediction_rows": int(group["unavailable_n"].sum()),
                    "prediction_availability": float(
                        group["usable_n"].sum() / group["attempted_n"].sum()
                    ),
                    "service_coverage": float(group["service_coverage"].mean()),
                    "coverage": float(group["coverage"].mean()),
                    "worst_group_undercoverage": float(group["undercoverage_error"].max()),
                    "normalized_width": float(group["normalized_width"].mean()),
                    "normalized_interval_score": float(group["normalized_interval_score"].mean()),
                }
            )
    station = pd.DataFrame(station_rows)
    result = (
        station.groupby(["horizon_steps", "method", "stratum"], observed=True, sort=True)
        .agg(
            stations=("station", "nunique"),
            eligible_groups=("eligible_groups", "sum"),
            eligible_prediction_rows=("eligible_prediction_rows", "sum"),
            attempted_prediction_rows=("attempted_prediction_rows", "sum"),
            unavailable_prediction_rows=("unavailable_prediction_rows", "sum"),
            mean_station_prediction_availability=("prediction_availability", "mean"),
            worst_station_prediction_availability=("prediction_availability", "min"),
            mean_station_service_coverage=("service_coverage", "mean"),
            mean_station_coverage=("coverage", "mean"),
            mean_station_worst_undercoverage=("worst_group_undercoverage", "mean"),
            worst_station_worst_undercoverage=("worst_group_undercoverage", "max"),
            mean_normalized_width=("normalized_width", "mean"),
            mean_normalized_interval_score=("normalized_interval_score", "mean"),
        )
        .reset_index()
    )
    baseline_name = str(cfg["methods"]["primary_baseline"])
    baseline = result[result["method"] == baseline_name][
        [
            "horizon_steps",
            "stratum",
            "mean_station_coverage",
            "mean_station_worst_undercoverage",
            "mean_normalized_width",
            "mean_normalized_interval_score",
        ]
    ]
    result = result.merge(baseline, on=["horizon_steps", "stratum"], suffixes=("", "_baseline"), how="left")
    result["coverage_change_vs_baseline"] = (
        result["mean_station_coverage"] - result["mean_station_coverage_baseline"]
    )
    result["worst_undercoverage_improvement_vs_baseline"] = (
        result["mean_station_worst_undercoverage_baseline"] - result["mean_station_worst_undercoverage"]
    )
    result["normalized_width_change_vs_baseline"] = (
        result["mean_normalized_width"] - result["mean_normalized_width_baseline"]
    )
    result["normalized_interval_score_change_vs_baseline"] = (
        result["mean_normalized_interval_score"] - result["mean_normalized_interval_score_baseline"]
    )
    baseline_columns = [
        "mean_station_coverage_baseline",
        "mean_station_worst_undercoverage_baseline",
        "mean_normalized_width_baseline",
        "mean_normalized_interval_score_baseline",
    ]
    result = result.drop(columns=baseline_columns)
    result["aggregation_unit"] = "station"
    result["within_station_group_weighting"] = "equal eligible fold-condition-ramp groups"
    result["minimum_group_n"] = min_n
    result["nominal_ramp_threshold"] = result["horizon_steps"].map(nominal)
    return result


def ramp_threshold_station_table(
    group_metrics: pd.DataFrame,
    cfg: dict,
    expected_stations: Iterable[str] | None = None,
) -> pd.DataFrame:
    min_n = int(cfg["calibration"]["min_group_n"])
    alpha = float(cfg["project"]["alpha"])
    baseline_name = str(cfg["methods"]["primary_baseline"])
    candidates = list(map(str, cfg["methods"]["primary_deployable_candidates"]))
    source = group_metrics.copy()
    expected_station_set = set(
        map(str, expected_stations)
        if expected_stations is not None
        else source["station"].astype(str).unique()
    )
    eligible = source[source["n"] >= min_n]

    physical = source[(source["method"].astype(str) == baseline_name) & (source["condition"].astype(str) == "clean")]
    physical = (
        physical.groupby(["station", "horizon_steps", "threshold", "ramp_group"], observed=True)["n"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    if "ramp" not in physical:
        physical["ramp"] = 0
    if "non_ramp" not in physical:
        physical["non_ramp"] = 0
    physical["test_n"] = physical["ramp"] + physical["non_ramp"]
    physical["ramp_n"] = physical["ramp"]
    physical["ramp_prevalence"] = physical["ramp_n"] / physical["test_n"]

    station_method = (
        eligible.groupby(["station", "horizon_steps", "threshold", "method"], observed=True, sort=True)
        .agg(
            eligible_groups=("undercoverage_error", "size"),
            worst_undercoverage=("undercoverage_error", "max"),
        )
        .reset_index()
    )
    baseline = station_method[station_method["method"].astype(str) == baseline_name].rename(
        columns={
            "eligible_groups": "baseline_eligible_groups",
            "worst_undercoverage": "baseline_worst_undercoverage",
        }
    )
    rows = station_method[station_method["method"].astype(str).isin(candidates)].rename(
        columns={
            "method": "candidate_method",
            "eligible_groups": "candidate_eligible_groups",
            "worst_undercoverage": "candidate_worst_undercoverage",
        }
    )
    rows = rows.merge(
        baseline.drop(columns=["method"]),
        on=["station", "horizon_steps", "threshold"],
        how="left",
        validate="many_to_one",
    ).merge(
        physical[
            ["station", "horizon_steps", "threshold", "test_n", "ramp_n", "ramp_prevalence"]
        ],
        on=["station", "horizon_steps", "threshold"],
        how="left",
        validate="many_to_one",
    )
    required_values = [
        "baseline_eligible_groups",
        "baseline_worst_undercoverage",
        "candidate_eligible_groups",
        "candidate_worst_undercoverage",
        "test_n",
        "ramp_n",
        "ramp_prevalence",
    ]
    if rows[required_values].isna().any().any():
        raise RuntimeError("Threshold sensitivity is missing a baseline, candidate, or physical count")
    rows["station_improvement"] = (
        rows["baseline_worst_undercoverage"] - rows["candidate_worst_undercoverage"]
    )
    delta = float(cfg["decision_rules"]["absolute_reliability_gate"]["max_worst_group_undercoverage_each_horizon"])
    rows["absolute_gate_delta"] = delta
    rows["required_minimum_coverage"] = (1.0 - alpha) - delta
    rows["candidate_absolute_gate_passed"] = rows["candidate_worst_undercoverage"] <= delta
    gate_keys = ["horizon_steps", "threshold", "candidate_method"]
    grouped_gate = rows.groupby(gate_keys, observed=True, sort=True)[
        "candidate_absolute_gate_passed"
    ]
    rows["candidate_stations"] = grouped_gate.transform("size").astype(int)
    rows["candidate_stations_passing_absolute_gate"] = grouped_gate.transform("sum").astype(int)
    observed_by_gate = {
        key: set(group["station"].astype(str))
        for key, group in rows.groupby(gate_keys, observed=True, sort=True)
    }
    completeness_by_gate = {
        key: not (expected_station_set - observed) and not (observed - expected_station_set)
        for key, observed in observed_by_gate.items()
    }
    missing_by_gate = {
        key: ";".join(sorted(expected_station_set - observed))
        for key, observed in observed_by_gate.items()
    }
    row_gate_keys = list(
        rows[[*gate_keys]].itertuples(index=False, name=None)
    )
    rows["expected_stations"] = len(expected_station_set)
    rows["candidate_station_set_complete"] = [
        completeness_by_gate[key] for key in row_gate_keys
    ]
    rows["candidate_missing_stations"] = [missing_by_gate[key] for key in row_gate_keys]
    rows["candidate_all_stations_absolute_gate_passed"] = (
        rows["candidate_station_set_complete"]
        & grouped_gate.transform("all").astype(bool)
    )
    nominal = {
        int(key): float(value)
        for key, value in cfg["forecast"]["ramp_threshold_capacity_fraction"].items()
    }
    rows["is_nominal_threshold"] = np.isclose(
        rows["threshold"].to_numpy(dtype=float),
        rows["horizon_steps"].map(nominal).to_numpy(dtype=float),
    )
    rows["minimum_group_n"] = min_n
    rows["aggregation_unit"] = "station"
    return rows.sort_values(["horizon_steps", "threshold", "candidate_method", "station"]).reset_index(drop=True)


def gate_sensitivity_table(
    group_metrics: pd.DataFrame,
    cfg: dict,
    deltas: tuple[float, ...] = (0.05, 0.10, 0.15),
    expected_stations: Iterable[str] | None = None,
) -> pd.DataFrame:
    min_n = int(cfg["calibration"]["min_group_n"])
    alpha = float(cfg["project"]["alpha"])
    nominal = {int(k): float(v) for k, v in cfg["forecast"]["ramp_threshold_capacity_fraction"].items()}
    methods = [str(cfg["methods"]["primary_baseline"])] + list(
        map(str, cfg["methods"]["primary_deployable_candidates"])
    )
    source = group_metrics[group_metrics["n"] >= min_n].copy()
    expected_station_set = set(
        map(str, expected_stations)
        if expected_stations is not None
        else group_metrics["station"].astype(str).unique()
    )
    source = source[
        np.isclose(
            source["threshold"].to_numpy(dtype=float),
            source["horizon_steps"].map(nominal).to_numpy(dtype=float),
        )
        & source["method"].astype(str).isin(methods)
    ]
    station = (
        source.groupby(["station", "horizon_steps", "method"], observed=True, sort=True)
        .agg(worst_undercoverage=("undercoverage_error", "max"))
        .reset_index()
    )
    rows: list[dict] = []
    for horizon in sorted(nominal):
        for method in methods:
            group = station[
                (station["horizon_steps"].astype(int) == horizon)
                & (station["method"].astype(str) == method)
            ]
            observed_station_set = set(group["station"].astype(str))
            missing_stations = sorted(expected_station_set - observed_station_set)
            unexpected_stations = sorted(observed_station_set - expected_station_set)
            station_set_complete = not missing_stations and not unexpected_stations
            for delta in deltas:
                passed = group["worst_undercoverage"].to_numpy(dtype=float) <= delta
                passing_stations = sorted(group.loc[passed, "station"].astype(str).tolist())
                failing_stations = sorted(group.loc[~passed, "station"].astype(str).tolist())
                rows.append(
                    {
                        "horizon_steps": int(horizon),
                        "method": str(method),
                        "delta": float(delta),
                        "nominal_coverage": 1.0 - alpha,
                        "required_minimum_coverage": (1.0 - alpha) - float(delta),
                        "stations": int(len(group)),
                        "expected_stations": int(len(expected_station_set)),
                        "station_set_complete": bool(station_set_complete),
                        "missing_stations": ";".join(missing_stations),
                        "unexpected_stations": ";".join(unexpected_stations),
                        "stations_passing": int(passed.sum()),
                        "station_pass_proportion": (
                            float(passed.mean()) if passed.size else 0.0
                        ),
                        "passing_stations": ";".join(passing_stations),
                        "failing_stations": ";".join(failing_stations),
                        "max_station_worst_undercoverage": (
                            float(group["worst_undercoverage"].max())
                            if not group.empty
                            else np.nan
                        ),
                        "all_stations_pass": bool(station_set_complete and passed.all()),
                        "aggregation_unit": "station",
                        "decision_rule": "all station worst-group undercoverage <= delta",
                    }
                )
    return pd.DataFrame(rows)


def future_label_diagnostic_table(
    source: pd.DataFrame,
    *,
    expected_stations: Iterable[str],
    expected_horizons: Iterable[int],
) -> pd.DataFrame:
    """Create direct, descriptive deployable-vs-future-label comparisons."""

    required = {
        "station",
        "horizon_steps",
        "local_worst_undercoverage",
        "oracle_worst_undercoverage",
        "local_interval_score",
        "oracle_interval_score",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise RuntimeError(f"Future-label diagnostic source is missing columns: {missing}")
    station_ids = [str(station) for station in expected_stations]
    horizon_ids = [int(horizon) for horizon in expected_horizons]
    expected_pairs = {
        (str(station), int(horizon))
        for station in station_ids
        for horizon in horizon_ids
    }
    observed_pairs = set(
        source[["station", "horizon_steps"]]
        .assign(station=lambda frame: frame["station"].astype(str))
        .assign(horizon_steps=lambda frame: frame["horizon_steps"].astype(int))
        .itertuples(index=False, name=None)
    )
    duplicate_pairs = source.duplicated(["station", "horizon_steps"], keep=False)
    if duplicate_pairs.any() or observed_pairs != expected_pairs:
        raise RuntimeError(
            "Future-label diagnostic source topology mismatch: "
            f"missing={sorted(expected_pairs - observed_pairs)}, "
            f"unexpected={sorted(observed_pairs - expected_pairs)}, "
            f"duplicate_rows={int(duplicate_pairs.sum())}"
        )

    result = source[
        [
            "station",
            "horizon_steps",
            "local_worst_undercoverage",
            "oracle_worst_undercoverage",
            "local_interval_score",
            "oracle_interval_score",
        ]
    ].rename(
        columns={
            "local_worst_undercoverage": "deployable_worst_undercoverage",
            "oracle_worst_undercoverage": "future_label_worst_undercoverage",
            "local_interval_score": "deployable_normalized_interval_score",
            "oracle_interval_score": "future_label_normalized_interval_score",
        }
    )
    numeric = [
        "deployable_worst_undercoverage",
        "future_label_worst_undercoverage",
        "deployable_normalized_interval_score",
        "future_label_normalized_interval_score",
    ]
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Future-label diagnostic source contains non-finite values")
    result.insert(2, "deployable_method", "robust_local_cqr")
    result.insert(3, "reference_display_name", "Future-label Mondrian diagnostic")
    result["uses_realized_future_ramp_status"] = True
    result["deployable"] = False
    result["future_label_minus_deployable_worst_undercoverage"] = (
        result["future_label_worst_undercoverage"]
        - result["deployable_worst_undercoverage"]
    )
    result["future_label_worse_undercoverage_than_deployable"] = (
        result["future_label_minus_deployable_worst_undercoverage"] > 0.0
    )
    result["future_label_minus_deployable_normalized_interval_score"] = (
        result["future_label_normalized_interval_score"]
        - result["deployable_normalized_interval_score"]
    )
    result["future_label_worse_score_than_deployable"] = (
        result["future_label_minus_deployable_normalized_interval_score"] > 0.0
    )
    result["comparison_type"] = "direct_descriptive_not_stacked"
    result["causal_or_additive_interpretation_allowed"] = False
    return result.sort_values(["station", "horizon_steps"]).reset_index(drop=True)


def write_future_label_diagnostic(
    cfg: dict,
    output: Path,
    *,
    root: Path = ROOT,
) -> tuple[pd.DataFrame, list[Path]]:
    source_path = root / cfg["outputs"]["tables"] / "oracle_gap_summary.csv"
    if not source_path.is_file():
        raise RuntimeError(f"Missing future-label diagnostic source: {source_path.relative_to(root)}")
    frame = future_label_diagnostic_table(
        pd.read_csv(source_path),
        expected_stations=valid_stations(cfg, root=root),
        expected_horizons=map(int, cfg["forecast"]["horizons_steps"]),
    )
    csv_path = output / "future_label_diagnostic.csv"
    json_path = output / "future_label_diagnostic.json"
    atomic_text(csv_path, frame.to_csv(index=False, lineterminator="\n", float_format="%.12g"))
    generator_path = root / "scripts" / "analyze_results.py"
    adverse = frame[
        frame["future_label_worse_undercoverage_than_deployable"]
        | frame["future_label_worse_score_than_deployable"]
    ]
    payload = {
        "schema_version": "1.0.0",
        "artifact": "future_label_mondrian_diagnostic",
        "display_name": "Future-label Mondrian diagnostic",
        "status": "nondeployable_descriptive_reference_only",
        "definition": "conditions on realized future ramp status",
        "deployable": False,
        "generated_at": None,
        "deterministic": True,
        "provenance": {
            "generator": generator_path.relative_to(root).as_posix(),
            "generator_sha256": file_sha256(generator_path),
            "source_table": source_path.relative_to(root).as_posix(),
            "source_table_sha256": file_sha256(source_path),
            "derived_csv": csv_path.relative_to(root).as_posix(),
            "derived_csv_sha256": file_sha256(csv_path),
        },
        "interpretation_constraints": [
            "direct deployable-versus-reference comparisons only",
            "descriptive, not causal or additive",
            "not a stacked decomposition",
            "finite-sample conditioning can perform worse than a deployable method",
        ],
        "rows": int(len(frame)),
        "adverse_case_count": int(len(adverse)),
        "adverse_cases": [
            {
                "station": str(row.station),
                "horizon_steps": int(row.horizon_steps),
                "future_label_worse_undercoverage_than_deployable": bool(
                    row.future_label_worse_undercoverage_than_deployable
                ),
                "future_label_worse_score_than_deployable": bool(
                    row.future_label_worse_score_than_deployable
                ),
            }
            for row in adverse.itertuples(index=False)
        ],
    }
    atomic_text(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return frame, [csv_path, json_path]

def build_tradeoff_outputs(cfg: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    chunk_paths, completion_evidence = validated_chunk_evidence(cfg)
    group_metrics = collect_threshold_metrics(cfg, chunk_paths=chunk_paths)
    validate_nominal_group_metrics(group_metrics, cfg)
    expected_stations = valid_stations(cfg)
    tradeoffs = station_first_tradeoffs(group_metrics, cfg)
    method_tradeoffs = tradeoffs[tradeoffs["stratum"] == "all"].reset_index(drop=True)
    ramp_tradeoffs = tradeoffs[tradeoffs["stratum"].isin(["ramp", "non_ramp"])].reset_index(drop=True)
    threshold_station = ramp_threshold_station_table(group_metrics, cfg, expected_stations=expected_stations)
    gate = gate_sensitivity_table(group_metrics, cfg, expected_stations=expected_stations)
    output.mkdir(parents=True, exist_ok=True)
    group_metrics.to_csv(output / "ramp_threshold_group_metrics.csv", index=False)
    method_tradeoffs.to_csv(output / "method_tradeoffs.csv", index=False)
    ramp_tradeoffs.to_csv(output / "ramp_tradeoffs.csv", index=False)
    threshold_station.to_csv(output / "ramp_threshold_sensitivity.csv", index=False)
    gate.to_csv(output / "gate_sensitivity.csv", index=False)
    build_natural_missingness_performance(cfg, chunk_paths, output)
    write_future_label_diagnostic(cfg, output)
    return group_metrics, completion_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate derived reliability analyses.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / cfg["outputs"]["tables"]
    metrics, evidence = build_tradeoff_outputs(cfg, output)
    print(json.dumps({"status": "COMPLETE", "metric_rows": len(metrics), "validated_jobs": len(evidence)}, indent=2))


if __name__ == "__main__":
    main()
