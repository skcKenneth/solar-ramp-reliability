from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.data import COLUMN_MAP, SENSOR_COLUMNS, load_solar_data  # noqa: E402
from solar_reliability.features import make_supervised  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_timestamp_column(columns: list[str]) -> str:
    for source, mapped in COLUMN_MAP.items():
        if mapped == "timestamp" and source in columns:
            return source
    raise ValueError("No timestamp column found")


def raw_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def fold_feasibility(
    frame: pd.DataFrame,
    capacity: float,
    cfg: dict,
) -> tuple[int, int, str]:
    feasible = 0
    checked = 0
    reasons: list[str] = []
    min_test = int(cfg["data"]["fold_policy"]["minimum_test_daylight_targets_per_fold_horizon"])
    min_ramps = int(cfg["data"]["fold_policy"]["minimum_ramp_targets_per_fold_horizon"])
    for horizon in cfg["forecast"]["horizons_steps"]:
        x, y, meta = make_supervised(
            frame,
            horizon_steps=int(horizon),
            lag_steps=list(cfg["forecast"]["lag_steps"]),
            rolling_windows=list(cfg["forecast"]["rolling_windows"]),
            daylight_hours=tuple(cfg["data"]["daylight_hours"]),
            nominal_capacity_mw=capacity,
        )
        threshold = float(cfg["forecast"]["ramp_threshold_capacity_fraction"][str(horizon)])
        for fold in cfg["data"]["site1_calendar_folds"]:
            checked += 1
            train_mask = x.index <= pd.Timestamp(fold["train_end"])
            cal_mask = x.index.to_series().between(
                pd.Timestamp(fold["calibration_start"]),
                pd.Timestamp(fold["calibration_end"]),
            ).to_numpy()
            test_mask = x.index.to_series().between(
                pd.Timestamp(fold["test_start"]),
                pd.Timestamp(fold["test_end"]),
            ).to_numpy()
            train_n = int(train_mask.sum())
            cal_n = int(cal_mask.sum())
            test_n = int(test_mask.sum())
            ramp_n = int((meta.loc[test_mask, "ramp_fraction"] >= threshold).sum())
            ok = train_n > 0 and cal_n > 0 and test_n >= min_test and ramp_n >= min_ramps
            if ok:
                feasible += 1
            else:
                reasons.append(
                    f"{fold['name']}:h{horizon}:train={train_n},cal={cal_n},test={test_n},ramp={ramp_n}"
                )
    return feasible, checked, "; ".join(reasons[:8])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit downloaded multi-site solar data.")
    parser.add_argument("--config", default="configs/multisite_confirmatory.yaml")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(ROOT / cfg["data"]["manifest"])
    out_table = ROOT / cfg["data"]["station_audit_table"]
    out_table.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / cfg["data"]["raw_dir"]

    records: list[dict] = []
    for row in manifest.itertuples(index=False):
        site_id = str(row.site_id)
        capacity = float(row.nominal_capacity_mw)
        path = raw_dir / str(row.filename)
        record: dict[str, object] = {
            "site_id": site_id,
            "source_url": row.source_url,
            "filename": row.filename,
            "nominal_capacity_mw": capacity,
            "exists": path.exists(),
            "status": "missing",
            "exclusion_reason": "",
        }
        if not path.exists():
            record["exclusion_reason"] = "downloaded file missing"
            records.append(record)
            continue

        raw = raw_frame(path)
        timestamp_col = infer_timestamp_column(list(raw.columns))
        raw_timestamps = pd.to_datetime(raw[timestamp_col], errors="coerce")
        duplicate_timestamps = int(raw_timestamps.duplicated().sum())
        duplicate_rate = float(duplicate_timestamps / len(raw)) if len(raw) else 1.0
        try:
            frame = load_solar_data(path, nominal_capacity_mw=capacity)
        except ValueError as exc:
            record.update(
                {
                    "status": "excluded",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "raw_rows": int(len(raw)),
                    "raw_columns_json": json.dumps(list(raw.columns), ensure_ascii=True),
                    "duplicate_timestamps": duplicate_timestamps,
                    "duplicate_timestamp_rate": duplicate_rate,
                    "exclusion_reason": f"schema validation failed: {exc}",
                }
            )
            records.append(record)
            continue
        expected_count = len(pd.date_range(frame.index.min(), frame.index.max(), freq="15min"))
        missing_timestamp_count = int(expected_count - raw_timestamps.dropna().nunique())
        daylight = frame.index.hour.to_series(index=frame.index).between(*cfg["data"]["daylight_hours"])
        target_missing_daylight = float(frame.loc[daylight, "power"].isna().mean())
        sensor_missing = {
            f"{sensor}_missing_rate": float(frame[sensor].isna().mean()) for sensor in SENSOR_COLUMNS
        }
        rules = cfg["data"]["integrity_exclusion_rules"]
        optional_sensors = set(rules.get("optional_sensor_columns", []))
        required_sensor_missing = {
            sensor: rate
            for sensor, rate in ((sensor, sensor_missing[f"{sensor}_missing_rate"]) for sensor in SENSOR_COLUMNS)
            if sensor not in optional_sensors
        }
        max_sensor_missing = max(required_sensor_missing.values())
        feasible, checked, infeasible_detail = fold_feasibility(frame, capacity, cfg)

        reasons: list[str] = []
        if duplicate_rate > float(rules["max_duplicate_timestamp_rate"]):
            reasons.append("duplicate timestamp rate above threshold")
        if target_missing_daylight > float(rules["max_target_missing_rate_daylight"]):
            reasons.append("daylight target missingness above threshold")
        if max_sensor_missing > float(rules["max_sensor_missing_rate_any_required_channel"]):
            reasons.append("sensor missingness above threshold")
        if feasible < checked and bool(rules["exclude_if_no_supported_fold"]):
            if feasible == 0:
                reasons.append("no supported Site 1 calendar fold/horizon cells")
        status = "valid" if not reasons else "excluded"
        record.update(
            {
                "status": status,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "raw_rows": int(len(raw)),
                "raw_columns_json": json.dumps(list(raw.columns), ensure_ascii=True),
                "start_timestamp": str(frame.index.min()),
                "end_timestamp": str(frame.index.max()),
                "sampling_interval_minutes": 15,
                "duplicate_timestamps": duplicate_timestamps,
                "duplicate_timestamp_rate": duplicate_rate,
                "missing_timestamps_after_15min_reindex": missing_timestamp_count,
                "target_missing_rate_daylight": target_missing_daylight,
                **sensor_missing,
                "max_sensor_missing_rate": max_sensor_missing,
                "optional_sensor_columns_json": json.dumps(sorted(optional_sensors), ensure_ascii=True),
                "site1_calendar_fold_horizon_cells_feasible": feasible,
                "site1_calendar_fold_horizon_cells_checked": checked,
                "site1_calendar_infeasible_detail": infeasible_detail,
                "exclusion_reason": "; ".join(reasons),
            }
        )
        records.append(record)

    audit = pd.DataFrame(records)
    audit.to_csv(out_table, index=False)
    print(f"wrote {out_table}")
    print(audit[["site_id", "status", "nominal_capacity_mw", "exclusion_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
