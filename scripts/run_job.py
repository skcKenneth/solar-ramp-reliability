from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.data import load_solar_data  # noqa: E402
from solar_reliability.features import make_supervised  # noqa: E402
from solar_reliability.reproducibility import (  # noqa: E402
    artifact_record,
    atomic_write_json,
    file_sha256,
    job_seed_map,
    runtime_environment,
    model_execution_source_sha256,
)
from solar_reliability.splits import leakage_safe_fold_masks, split_audit_summary  # noqa: E402


MARKER_SCHEMA_VERSION = 2


def station_row(cfg: dict, station: str) -> pd.Series:
    manifest = pd.read_csv(ROOT / cfg["data"]["manifest"])
    matches = manifest[manifest["site_id"].astype(str) == station]
    if matches.empty:
        raise ValueError(f"Unknown station: {station}")
    return matches.iloc[0]


def assert_station_allowed(cfg: dict, station: str) -> None:
    if station in set(cfg["project"].get("development_sites", [])):
        return
    audit_path = ROOT / cfg["data"]["station_audit_table"]
    if not audit_path.exists():
        raise FileNotFoundError(f"Run station data audit first: {audit_path}")
    audit = pd.read_csv(audit_path)
    row = audit[audit["site_id"].astype(str) == station]
    if row.empty:
        raise ValueError(f"Station missing from audit table: {station}")
    status = str(row.iloc[0]["status"])
    if status != "valid":
        reason = str(row.iloc[0].get("exclusion_reason", ""))
        raise ValueError(f"Station {station} is not valid for confirmatory execution: {reason}")


def build_station_config(cfg: dict, station: str, output_root: Path) -> tuple[dict, Path]:
    row = station_row(cfg, station)
    station_cfg = {
        "project": {
            "seed": int(cfg["project"]["seed"]),
            "station_id": station,
            "nominal_capacity_mw": float(row["nominal_capacity_mw"]),
            "alpha": float(cfg["project"]["alpha"]),
            "status": f"confirmatory_{station}",
        },
        "data": {
            "path": (Path(cfg["data"]["raw_dir"]) / str(row["filename"])).as_posix(),
            "daylight_hours": list(cfg["data"]["daylight_hours"]),
            "sampling_minutes": int(cfg["data"]["sampling_minutes"]),
            "folds": list(cfg["data"]["site1_calendar_folds"]),
        },
        "forecast": cfg["forecast"],
        "model": cfg["model"],
        "calibration": cfg["calibration"],
        "bootstrap": {"replicates": int(cfg["bootstrap"]["day_inner_replicates"])},
        "conditions": cfg["conditions"],
        "methods": cfg["methods"]["full_set"],
        "outputs": {"root": output_root.relative_to(ROOT).as_posix()},
    }
    config_dir = ROOT / cfg["outputs"]["cache"] / "generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{station}.yaml"
    config_path.write_text(yaml.safe_dump(station_cfg, sort_keys=False), encoding="utf-8")
    return station_cfg, config_path


def expected_artifact_paths(station_cfg: dict, fold_name: str, horizon: int) -> list[Path]:
    output_root = ROOT / station_cfg["outputs"]["root"]
    chunks = output_root / "cache" / "chunks"
    stem = f"{fold_name}_h{int(horizon)}"
    return [
        chunks / f"predictions_{stem}.pkl.gz",
        chunks / f"run_{stem}.csv",
        chunks / f"calibration_{stem}.csv",
    ]


def marker_is_reusable(marker: Path, expected: dict, artifact_paths: list[Path]) -> tuple[bool, str]:
    if not marker.is_file():
        return False, "marker_missing"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "marker_unreadable"
    expected_status = "SMOKE_OK" if expected["mode"] == "smoke" else "COMPLETE"
    if payload.get("schema_version") != MARKER_SCHEMA_VERSION:
        return False, "marker_schema_mismatch"
    if payload.get("status") != expected_status:
        return False, f"marker_status_{payload.get('status')}"
    for key in ("mode", "job", "provenance", "seeds", "environment"):
        if payload.get(key) != expected.get(key):
            return False, f"marker_{key}_mismatch"
    if expected["mode"] == "smoke":
        return True, "validated_smoke_marker"

    recorded = {item.get("path"): item for item in payload.get("artifacts", [])}
    expected_records = {
        artifact_record(path, root=ROOT)["path"] for path in artifact_paths
    }
    if set(recorded) != expected_records:
        return False, "artifact_set_mismatch"
    for path in artifact_paths:
        current = artifact_record(path, root=ROOT)
        if not current["exists"]:
            return False, f"artifact_missing:{current['path']}"
        if recorded.get(current["path"]) != current:
            return False, f"artifact_mismatch:{current['path']}"
    return True, "validated_marker_and_artifacts"


def selected_fold(station_cfg: dict, fold_name: str) -> dict:
    for fold in station_cfg["data"]["folds"]:
        if str(fold["name"]) == fold_name:
            return fold
    raise ValueError(f"Unknown fold: {fold_name}")


def smoke_check(station: str, station_cfg: dict, fold_name: str, horizon: int, data_hash: str) -> dict:
    fold = selected_fold(station_cfg, fold_name)
    frame = load_solar_data(
        ROOT / station_cfg["data"]["path"],
        nominal_capacity_mw=float(station_cfg["project"]["nominal_capacity_mw"]),
    )
    x, y, meta = make_supervised(
        frame,
        horizon_steps=horizon,
        lag_steps=list(station_cfg["forecast"]["lag_steps"]),
        rolling_windows=list(station_cfg["forecast"]["rolling_windows"]),
        daylight_hours=tuple(station_cfg["data"]["daylight_hours"]),
        nominal_capacity_mw=float(station_cfg["project"]["nominal_capacity_mw"]),
    )
    masks = leakage_safe_fold_masks(
        x.index,
        meta["target_timestamp"],
        fold,
        horizon_steps=horizon,
        sampling_minutes=int(station_cfg["data"]["sampling_minutes"]),
    )
    threshold = float(station_cfg["forecast"]["ramp_threshold_capacity_fraction"][str(horizon)])
    test_ramps = int((meta.loc[masks.test, "ramp_fraction"] >= threshold).sum())
    return {
        "nominal_capacity_mw": float(station_cfg["project"]["nominal_capacity_mw"]),
        "data_sha256": data_hash,
        "rows": {
            "supervised": int(len(x)),
            **masks.counts(),
            "test_ramps": test_ramps,
        },
        "split_audit": split_audit_summary(x.index, meta["target_timestamp"], masks),
        "ramp_threshold_capacity_fraction": threshold,
    }


def run_artifact_summary(run_path: Path) -> dict:
    frame = pd.read_csv(run_path)
    if len(frame) != 1:
        raise RuntimeError(f"Expected one run-summary row in {run_path}, found {len(frame)}")
    row = frame.iloc[0]
    split_audit = json.loads(str(row["split_audit_json"]))
    return {
        "station": str(row["station"]),
        "fold": str(row["fold"]),
        "horizon_steps": int(row["horizon_steps"]),
        "train_n_after_subsample": int(row["train_n"]),
        "calibration_n": int(row["calibration_n"]),
        "test_n": int(row["test_n"]),
        "split_audit": split_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or smoke-test frozen multi-site confirmation.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--station", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = time.time()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    station = str(args.station)
    assert_station_allowed(cfg, station)
    row = station_row(cfg, station)
    data_path = ROOT / cfg["data"]["raw_dir"] / str(row["filename"])
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    output_root = ROOT / cfg["outputs"]["root"] / "stations" / station
    station_cfg, station_cfg_path = build_station_config(cfg, station, output_root)
    stem = f"{station}_{args.fold}_h{args.horizon}"
    marker_dir = ROOT / cfg["outputs"]["completion_markers"]
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{stem}{'_smoke' if args.smoke else ''}.json"

    data_hash = file_sha256(data_path)
    root_config_hash = file_sha256(cfg_path)
    config_hash = file_sha256(station_cfg_path)
    source_hash = model_execution_source_sha256(ROOT)
    seeds = job_seed_map(
        int(cfg["project"]["seed"]),
        station,
        str(args.fold),
        int(args.horizon),
    )
    mode = "smoke" if args.smoke else "full"
    common = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "mode": mode,
        "job": {
            "station": station,
            "fold": str(args.fold),
            "horizon_steps": int(args.horizon),
        },
        "provenance": {
            "root_config": str(Path(args.config).as_posix()),
            "root_config_sha256": root_config_hash,
            "station_config": str(station_cfg_path.relative_to(ROOT).as_posix()),
            "station_config_sha256": config_hash,
            "data_path": str(data_path.relative_to(ROOT).as_posix()),
            "data_sha256": data_hash,
            "model_execution_source_sha256": source_hash,
        },
        "seeds": seeds,
        "environment": runtime_environment(),
    }
    artifact_paths = expected_artifact_paths(station_cfg, str(args.fold), int(args.horizon))
    reusable, reason = marker_is_reusable(marker, common, artifact_paths)
    if reusable and not args.force:
        print(f"already complete ({reason}): {marker}")
        return
    if marker.exists() and not args.force:
        print(f"re-running because {reason}: {marker}", flush=True)

    if args.smoke:
        smoke = smoke_check(station, station_cfg, args.fold, args.horizon, data_hash)
        finished = time.time()
        payload = common | {
            "status": "SMOKE_OK",
            "started_at_unix": started,
            "finished_at_unix": finished,
            "smoke": smoke,
        }
        payload["runtime_seconds"] = round(payload["finished_at_unix"] - started, 3)
        atomic_write_json(marker, payload)
        print(json.dumps(payload, indent=2))
        return

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_advanced_study.py"),
        "--config",
        str(station_cfg_path.relative_to(ROOT)),
        "--fold",
        args.fold,
        "--horizon",
        str(args.horizon),
        "--chunk-only",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    payload = common | {
        "status": "COMPLETE" if result.returncode == 0 else "FAILED",
        "returncode": result.returncode,
        "command": command,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
    }
    payload["runtime_seconds"] = round(payload["finished_at_unix"] - started, 3)
    payload["artifacts"] = [artifact_record(path, root=ROOT) for path in artifact_paths]
    if result.returncode == 0:
        if not all(item["exists"] for item in payload["artifacts"]):
            payload["status"] = "FAILED"
            payload["failure_reason"] = "subprocess returned zero but expected artifacts are missing"
            payload["returncode"] = 86
        else:
            try:
                payload["split"] = run_artifact_summary(artifact_paths[1])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
                payload["status"] = "FAILED"
                payload["failure_reason"] = f"invalid run-summary evidence: {exc}"
                payload["returncode"] = 87
    atomic_write_json(marker, payload)
    if payload["status"] != "COMPLETE":
        raise SystemExit(int(payload["returncode"]))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
