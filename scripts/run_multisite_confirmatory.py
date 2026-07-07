from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


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
            "nominal_capacity_mw": float(row["nominal_capacity_mw"]),
            "alpha": float(cfg["project"]["alpha"]),
            "status": f"confirmatory_{station}",
        },
        "data": {
            "path": (Path(cfg["data"]["raw_dir"]) / str(row["filename"])).as_posix(),
            "daylight_hours": list(cfg["data"]["daylight_hours"]),
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
    train_mask = x.index <= pd.Timestamp(fold["train_end"])
    cal_mask = x.index.to_series().between(
        pd.Timestamp(fold["calibration_start"]),
        pd.Timestamp(fold["calibration_end"]),
    ).to_numpy()
    test_mask = x.index.to_series().between(
        pd.Timestamp(fold["test_start"]),
        pd.Timestamp(fold["test_end"]),
    ).to_numpy()
    threshold = float(station_cfg["forecast"]["ramp_threshold_capacity_fraction"][str(horizon)])
    test_ramps = int((meta.loc[test_mask, "ramp_fraction"] >= threshold).sum())
    return {
        "status": "SMOKE_OK",
        "station": station,
        "fold": fold_name,
        "horizon_steps": horizon,
        "nominal_capacity_mw": float(station_cfg["project"]["nominal_capacity_mw"]),
        "data_sha256": data_hash,
        "rows": {
            "supervised": int(len(x)),
            "train": int(train_mask.sum()),
            "calibration": int(cal_mask.sum()),
            "test": int(test_mask.sum()),
            "test_ramps": test_ramps,
        },
        "ramp_threshold_capacity_fraction": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or smoke-test frozen multi-site confirmation.")
    parser.add_argument("--config", default="configs/multisite_confirmatory.yaml")
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
    if marker.exists() and not args.force:
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("status") in {"COMPLETE", "SMOKE_OK"}:
            print(f"already complete: {marker}")
            return
        print(f"re-running after previous marker status={previous.get('status')}: {marker}", flush=True)

    data_hash = sha256(data_path)
    config_hash = sha256(station_cfg_path)
    common = {
        "station": station,
        "fold": args.fold,
        "horizon_steps": args.horizon,
        "config": str(Path(args.config)),
        "station_config": str(station_cfg_path.relative_to(ROOT)),
        "config_sha256": config_hash,
        "data_path": str(data_path.relative_to(ROOT)),
        "data_sha256": data_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "started_at_unix": started,
    }

    if args.smoke:
        payload = common | smoke_check(station, station_cfg, args.fold, args.horizon, data_hash)
        payload["runtime_seconds"] = round(time.time() - started, 3)
        atomic_json(marker, payload)
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
        "runtime_seconds": round(time.time() - started, 3),
    }
    atomic_json(marker, payload)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
