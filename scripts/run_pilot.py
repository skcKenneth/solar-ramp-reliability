from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.conformal import (  # noqa: E402
    adjust_interval,
    cqr_scores,
    fit_calibration_tables,
    qhat_for_method,
)
from solar_reliability.data import load_solar_data, missingness_summary  # noqa: E402
from solar_reliability.features import apply_condition, make_supervised  # noqa: E402
from solar_reliability.metrics import method_primary_scores, summarize_predictions  # noqa: E402
from solar_reliability.models import fit_quantile_models, fit_ramp_risk_model  # noqa: E402
from solar_reliability.statistics import day_cluster_bootstrap_difference  # noqa: E402

METHODS = ["native", "clean_cqr", "augmented_cqr", "mondrian_cqr", "hierarchical_cqr"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()

    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_tables = ROOT / "outputs" / "tables"
    out_cache = ROOT / "outputs" / "cache"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_cache.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / cfg["data"]["path"]
    frame = load_solar_data(data_path)
    missingness_summary(frame).to_csv(out_tables / "data_missingness.csv", index=False)

    data_audit = {
        "rows": int(len(frame)),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "frequency": "15min",
        "dataset_sha256": sha256(data_path),
        "config_sha256": sha256(config_path),
        "power_missing": int(frame["power"].isna().sum()),
        "sensor_missing_rows": int(frame.drop(columns="power").isna().any(axis=1).sum()),
    }
    (out_tables / "data_audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")

    alpha = float(cfg["project"]["alpha"])
    capacity = float(cfg["project"]["nominal_capacity_mw"])
    conditions = list(cfg["conditions"])
    seed = int(cfg["project"]["seed"])
    all_predictions: list[pd.DataFrame] = []
    model_records: list[dict] = []

    for horizon in cfg["forecast"]["horizons_steps"]:
        x, y, meta = make_supervised(
            frame,
            int(horizon),
            lag_steps=list(cfg["forecast"]["lag_steps"]),
            rolling_windows=list(cfg["forecast"]["rolling_windows"]),
            daylight_hours=tuple(cfg["data"]["daylight_hours"]),
            nominal_capacity_mw=capacity,
        )
        train_mask = x.index <= pd.Timestamp(cfg["data"]["train_end"])
        cal_mask = x.index.to_series().between(
            pd.Timestamp(cfg["data"]["calibration_start"]), pd.Timestamp(cfg["data"]["calibration_end"])
        ).to_numpy()
        test_mask = x.index.to_series().between(
            pd.Timestamp(cfg["data"]["test_start"]), pd.Timestamp(cfg["data"]["test_end"])
        ).to_numpy()

        x_train, y_train, meta_train = x.loc[train_mask], y.loc[train_mask], meta.loc[train_mask]
        max_train_rows = int(cfg["model"].get("max_train_rows", len(x_train)))
        if len(x_train) > max_train_rows:
            take = np.linspace(0, len(x_train) - 1, max_train_rows, dtype=int)
            x_train, y_train, meta_train = x_train.iloc[take], y_train.iloc[take], meta_train.iloc[take]
        x_cal, y_cal, meta_cal = x.loc[cal_mask], y.loc[cal_mask], meta.loc[cal_mask]
        x_test, y_test, meta_test = x.loc[test_mask], y.loc[test_mask], meta.loc[test_mask]
        if min(len(x_train), len(x_cal), len(x_test)) == 0:
            raise RuntimeError(f"Empty split for horizon {horizon}")

        models = fit_quantile_models(x_train, y_train, alpha, cfg["model"], seed + int(horizon) * 10)
        ramp_threshold = float(cfg["forecast"]["ramp_threshold_capacity_fraction"])
        ramp_model = fit_ramp_risk_model(
            x_train,
            (meta_train["ramp_fraction"].to_numpy() >= ramp_threshold).astype(int),
            cfg["model"],
            seed + int(horizon) * 10 + 5,
        )
        model_records.append(
            {
                "horizon_steps": int(horizon),
                "horizon_minutes": int(horizon) * 15,
                "train_n": int(len(x_train)),
                "calibration_n": int(len(x_cal)),
                "test_n": int(len(x_test)),
                "features": int(x_train.shape[1]),
            }
        )

        calibration_rows: list[pd.DataFrame] = []
        for condition in conditions:
            xc = apply_condition(x_cal, condition)
            lo, med, hi = models.predict(xc)
            scores = cqr_scores(y_cal.to_numpy(), lo, hi)
            calibration_rows.append(
                pd.DataFrame(
                    {
                        "score": scores,
                        "condition": condition,
                        "volatility": ramp_model.predict_proba(xc),
                    },
                    index=xc.index,
                )
            )
        score_frame = pd.concat(calibration_rows, axis=0, ignore_index=True)
        calibration_ramp_prevalence = float((meta_cal["ramp_fraction"] >= ramp_threshold).mean())
        risk_quantile = float(np.clip(1.0 - calibration_ramp_prevalence, 0.50, 0.95))
        tables = fit_calibration_tables(score_frame, alpha, risk_quantile)

        for condition in conditions:
            xt = apply_condition(x_test, condition)
            lo_native, med, hi_native = models.predict(xt)
            risk = ramp_model.predict_proba(xt)
            ramp_group = np.where(meta_test["ramp_fraction"].to_numpy() >= ramp_threshold, "ramp", "non_ramp")
            for method in METHODS:
                qhat = qhat_for_method(
                    method,
                    condition,
                    risk,
                    tables,
                    float(cfg["calibration"]["hierarchical_shrinkage_k"]),
                )
                lo, hi = adjust_interval(lo_native, hi_native, qhat, capacity)
                covered = (y_test.to_numpy() >= lo) & (y_test.to_numpy() <= hi)
                all_predictions.append(
                    pd.DataFrame(
                        {
                            "timestamp": xt.index,
                            "date": meta_test["date"].to_numpy(),
                            "horizon_steps": int(horizon),
                            "horizon_minutes": int(horizon) * 15,
                            "condition": condition,
                            "method": method,
                            "ramp_group": ramp_group,
                            "ramp_fraction": meta_test["ramp_fraction"].to_numpy(),
                            "y": y_test.to_numpy(),
                            "median": med,
                            "lower": lo,
                            "upper": hi,
                            "covered": covered.astype(int),
                            "interval_width": hi - lo,
                            "transition_risk": risk,
                        }
                    )
                )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out_cache / "pilot_predictions.csv.gz", index=False, compression="gzip")
    summary = summarize_predictions(predictions, alpha, int(cfg["calibration"]["min_group_n"]))
    primary = method_primary_scores(summary)
    bootstrap = day_cluster_bootstrap_difference(
        predictions,
        "hierarchical_cqr",
        "clean_cqr",
        alpha,
        int(cfg["bootstrap"]["replicates"]),
        seed,
    )

    pd.DataFrame(model_records).to_csv(out_tables / "model_and_split_summary.csv", index=False)
    summary.to_csv(out_tables / "conditional_metrics.csv", index=False)
    primary.to_csv(out_tables / "method_primary_scores.csv", index=False)
    bootstrap.to_csv(out_tables / "bootstrap_hierarchical_vs_clean.csv", index=False)

    best = primary.sort_values(["horizon_steps", "worst_group_coverage_error", "mean_interval_score"]).groupby("horizon_steps").head(1)
    verdict = {
        "status": "PILOT_COMPLETE",
        "verdict": "GO_AFTER_MAJOR_REVISION",
        "best_by_horizon": best.to_dict(orient="records"),
        "method_claim_allowed": False,
        "reason": "Single-site pilot; external multi-site confirmation is required before a full-paper method claim.",
    }
    (out_tables / "pilot_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
