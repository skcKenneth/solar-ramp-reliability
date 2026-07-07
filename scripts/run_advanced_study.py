from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - Windows compatibility
    resource = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.advanced import (  # noqa: E402
    augment_training_set,
    condition_quantiles,
    descriptor_frame,
    fit_local_score_calibrator,
    fit_risk_mondrian,
    oracle_ramp_quantiles,
    rolling_qhat,
)
from solar_reliability.conformal import adjust_interval, cqr_scores  # noqa: E402
from solar_reliability.data import load_solar_data  # noqa: E402
from solar_reliability.features import apply_condition, latest_available_power, make_supervised  # noqa: E402
from solar_reliability.metrics import interval_metrics, point_metrics  # noqa: E402
from solar_reliability.models import (  # noqa: E402
    fit_hist_quantile_models,
    fit_quantile_models,
    fit_ramp_risk_model,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_mb() -> float | None:
    if resource is None:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def clip_interval(lower: np.ndarray, upper: np.ndarray, capacity: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.clip(np.asarray(lower, dtype=float), 0.0, capacity)
    hi = np.clip(np.asarray(upper, dtype=float), 0.0, capacity)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def load_configured_frame(cfg: dict) -> tuple[pd.DataFrame, float, Path]:
    capacity = float(cfg["project"]["nominal_capacity_mw"])
    data_path = ROOT / cfg["data"]["path"]
    frame = load_solar_data(data_path, nominal_capacity_mw=capacity)
    return frame, capacity, data_path


def subsample_training(
    x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, max_rows: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if len(x) <= max_rows:
        return x, y, meta
    take = np.linspace(0, len(x) - 1, max_rows, dtype=int)
    return x.iloc[take], y.iloc[take], meta.iloc[take]


def summarize_advanced(predictions: pd.DataFrame, alpha: float, min_group_n: int, capacity: float) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["fold", "horizon_steps", "method", "condition", "ramp_group"]
    for key, group in predictions.groupby(group_cols, sort=True, observed=True):
        if len(group) < min_group_n:
            continue
        im = interval_metrics(group["y"], group["lower"], group["upper"], alpha)
        pm = point_metrics(group["y"], group["median"])
        coverage = im["coverage"]
        row = dict(zip(group_cols, key)) | im | pm
        row["undercoverage_error"] = max(0.0, (1 - alpha) - coverage)
        row["overcoverage_error"] = max(0.0, coverage - (1 - alpha))
        row["normalized_width"] = im["mean_width"] / capacity
        row["normalized_interval_score"] = im["interval_score"] / capacity
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_methods(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (horizon, method), group in summary.groupby(["horizon_steps", "method"], sort=True, observed=True):
        clean = group[group["condition"] == "clean"]
        rows.append(
            {
                "horizon_steps": int(horizon),
                "horizon_minutes": int(horizon) * 15,
                "method": method,
                "groups": int(len(group)),
                "folds": int(group["fold"].nunique()),
                "worst_group_undercoverage": float(group["undercoverage_error"].max()),
                "q90_group_undercoverage": float(group["undercoverage_error"].quantile(0.9)),
                "mean_group_absolute_coverage_error": float(group["coverage_error"].mean()),
                "mean_normalized_width": float(group["normalized_width"].mean()),
                "mean_normalized_interval_score": float(group["normalized_interval_score"].mean()),
                "clean_normalized_interval_score": float(clean["normalized_interval_score"].mean()),
                "mean_mae": float(group["mae"].mean()),
            }
        )
    return pd.DataFrame(rows)


def fold_method_scores(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["fold", "horizon_steps", "method"], as_index=False, observed=True)
        .agg(
            worst_undercoverage=("undercoverage_error", "max"),
            mean_abs_coverage_error=("coverage_error", "mean"),
            mean_normalized_width=("normalized_width", "mean"),
            mean_normalized_interval_score=("normalized_interval_score", "mean"),
        )
    )


def bootstrap_worst_group_difference(
    predictions: pd.DataFrame,
    method_a: str,
    method_b: str,
    alpha: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = 1 - alpha
    records: list[dict] = []
    daily = (
        predictions[predictions["method"].isin([method_a, method_b])]
        .groupby(["horizon_steps", "method", "fold", "date", "condition", "ramp_group"], as_index=False, observed=True)
        .agg(covered_sum=("covered", "sum"), n=("covered", "size"))
    )
    for horizon, hdf in daily.groupby("horizon_steps", sort=True):
        group_keys = sorted(
            {(str(r.fold), str(r.condition), str(r.ramp_group)) for r in hdf.itertuples()}
        )
        key_to_col = {key: i for i, key in enumerate(group_keys)}
        fold_payload: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for fold, fdf in hdf.groupby("fold", sort=True, observed=True):
            days = np.array(sorted(fdf["date"].unique()))
            day_to_row = {day: i for i, day in enumerate(days)}
            arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for method in [method_a, method_b]:
                cov = np.zeros((len(days), len(group_keys)), dtype=float)
                cnt = np.zeros_like(cov)
                mdf = fdf[fdf["method"] == method]
                for row in mdf.itertuples():
                    i = day_to_row[row.date]
                    j = key_to_col[(str(row.fold), str(row.condition), str(row.ramp_group))]
                    cov[i, j] = float(row.covered_sum)
                    cnt[i, j] = float(row.n)
                arrays[method] = (cov, cnt)
            fold_payload.append((*arrays[method_a], *arrays[method_b]))

        diffs = np.empty(replicates, dtype=float)
        for rep in range(replicates):
            a_cov_parts, a_n_parts, b_cov_parts, b_n_parts = [], [], [], []
            for a_cov, a_n, b_cov, b_n in fold_payload:
                idx = rng.integers(0, a_cov.shape[0], size=a_cov.shape[0])
                a_cov_parts.append(a_cov[idx].sum(axis=0))
                a_n_parts.append(a_n[idx].sum(axis=0))
                b_cov_parts.append(b_cov[idx].sum(axis=0))
                b_n_parts.append(b_n[idx].sum(axis=0))
            a_cov_sum = np.sum(a_cov_parts, axis=0)
            a_n_sum = np.sum(a_n_parts, axis=0)
            b_cov_sum = np.sum(b_cov_parts, axis=0)
            b_n_sum = np.sum(b_n_parts, axis=0)
            a_valid = a_n_sum > 0
            b_valid = b_n_sum > 0
            a_under = np.maximum(0.0, target - a_cov_sum[a_valid] / a_n_sum[a_valid])
            b_under = np.maximum(0.0, target - b_cov_sum[b_valid] / b_n_sum[b_valid])
            diffs[rep] = float(a_under.max() - b_under.max())
        records.append(
            {
                "horizon_steps": int(horizon),
                "method_a": method_a,
                "method_b": method_b,
                "difference_a_minus_b": float(diffs.mean()),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "replicates": int(replicates),
            }
        )
    return pd.DataFrame(records)


def ramp_sensitivity_table(predictions: pd.DataFrame, cfg: dict, alpha: float) -> pd.DataFrame:
    methods = ["robust_condition_cqr", "robust_risk_mondrian", "robust_local_cqr", "robust_rolling_cqr"]
    rows: list[dict] = []
    for horizon, hdf in predictions[predictions["method"].isin(methods)].groupby("horizon_steps", observed=True):
        thresholds = cfg["forecast"]["ramp_sensitivity"][str(int(horizon))]
        for threshold in thresholds:
            temp = hdf.copy()
            temp["sensitivity_group"] = np.where(temp["ramp_fraction"] >= float(threshold), "ramp", "non_ramp")
            for (method, condition, group_name), group in temp.groupby(
                ["method", "condition", "sensitivity_group"], sort=True, observed=True
            ):
                if len(group) < 80:
                    continue
                coverage = float(group["covered"].mean())
                rows.append(
                    {
                        "horizon_steps": int(horizon),
                        "threshold_fraction": float(threshold),
                        "method": method,
                        "condition": condition,
                        "ramp_group": group_name,
                        "n": int(len(group)),
                        "coverage": coverage,
                        "undercoverage_error": max(0.0, (1 - alpha) - coverage),
                        "mean_width": float(group["interval_width"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def make_prediction_rows(
    fold_name: str,
    horizon: int,
    condition: str,
    method: str,
    meta: pd.DataFrame,
    y: pd.Series,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    risk: np.ndarray,
    ramp_threshold: float,
) -> pd.DataFrame:
    y_arr = y.to_numpy(dtype=float)
    covered = (y_arr >= lower) & (y_arr <= upper)
    frame = pd.DataFrame(
        {
            "fold": fold_name,
            "timestamp": meta.index,
            "target_timestamp": meta["target_timestamp"].to_numpy(),
            "date": meta["date"].to_numpy(),
            "horizon_steps": np.int8(horizon),
            "horizon_minutes": np.int16(horizon * 15),
            "condition": condition,
            "method": method,
            "ramp_group": np.where(meta["ramp_fraction"].to_numpy() >= ramp_threshold, "ramp", "non_ramp"),
            "ramp_direction": meta["ramp_direction"].to_numpy(),
            "ramp_fraction": meta["ramp_fraction"].to_numpy(dtype=np.float32),
            "y": y_arr.astype(np.float32),
            "median": np.asarray(median, dtype=np.float32),
            "lower": np.asarray(lower, dtype=np.float32),
            "upper": np.asarray(upper, dtype=np.float32),
            "covered": covered.astype(np.int8),
            "interval_width": np.asarray(upper - lower, dtype=np.float32),
            "transition_risk": np.asarray(risk, dtype=np.float32),
        }
    )
    for column in ["fold", "condition", "method", "ramp_group", "ramp_direction"]:
        frame[column] = frame[column].astype("category")
    return frame


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/advanced.yaml")
    parser.add_argument("--fold", default=None, help="Run one named temporal fold")
    parser.add_argument("--horizon", type=int, default=None, help="Run one horizon in 15-minute steps")
    parser.add_argument("--chunk-only", action="store_true", help="Save one combo and skip aggregation")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    out_root = ROOT / cfg.get("outputs", {}).get("root", "outputs/advanced")
    out_tables = out_root / "tables"
    out_cache = out_root / "cache"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_cache.mkdir(parents=True, exist_ok=True)

    started = time.time()
    frame, capacity, data_path = load_configured_frame(cfg)
    alpha = float(cfg["project"]["alpha"])
    seed = int(cfg["project"]["seed"])
    conditions = list(cfg["conditions"])

    all_predictions: list[pd.DataFrame] = []
    run_records: list[dict] = []
    calibration_records: list[dict] = []

    supervised_cache: dict[int, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}
    for horizon in cfg["forecast"]["horizons_steps"]:
        supervised_cache[int(horizon)] = make_supervised(
            frame,
            int(horizon),
            lag_steps=list(cfg["forecast"]["lag_steps"]),
            rolling_windows=list(cfg["forecast"]["rolling_windows"]),
            daylight_hours=tuple(cfg["data"]["daylight_hours"]),
            nominal_capacity_mw=capacity,
            include_current_power=True,
        )

    selected_folds = [f for f in cfg["data"]["folds"] if args.fold is None or f["name"] == args.fold]
    selected_horizons = [h for h in cfg["forecast"]["horizons_steps"] if args.horizon is None or int(h) == args.horizon]
    if not selected_folds or not selected_horizons:
        raise ValueError("Requested fold or horizon was not found in config")
    if args.chunk_only and (len(selected_folds) != 1 or len(selected_horizons) != 1):
        raise ValueError("--chunk-only requires exactly one --fold and one --horizon")

    for fold_index, fold in enumerate(selected_folds):
        fold_name = str(fold["name"])
        for horizon in selected_horizons:
            horizon = int(horizon)
            x, y, meta = supervised_cache[horizon]
            train_mask = x.index <= pd.Timestamp(fold["train_end"])
            cal_mask = x.index.to_series().between(
                pd.Timestamp(fold["calibration_start"]), pd.Timestamp(fold["calibration_end"])
            ).to_numpy()
            test_mask = x.index.to_series().between(
                pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"])
            ).to_numpy()
            x_train, y_train, meta_train = x.loc[train_mask], y.loc[train_mask], meta.loc[train_mask]
            x_train, y_train, meta_train = subsample_training(
                x_train, y_train, meta_train, int(cfg["model"]["max_train_rows"])
            )
            x_cal, y_cal, meta_cal = x.loc[cal_mask], y.loc[cal_mask], meta.loc[cal_mask]
            x_test, y_test, meta_test = x.loc[test_mask], y.loc[test_mask], meta.loc[test_mask]
            if min(len(x_train), len(x_cal), len(x_test)) == 0:
                raise RuntimeError(f"Empty split: {fold_name}, h={horizon}")

            ramp_threshold = float(cfg["forecast"]["ramp_threshold_capacity_fraction"][str(horizon)])
            ramp_train = meta_train["ramp_fraction"].to_numpy() >= ramp_threshold
            model_seed = seed + 1000 * fold_index + 10 * horizon

            print(f"starting {fold_name}, horizon={horizon}: train={len(x_train)}, cal={len(x_cal)}, test={len(x_test)}", flush=True)
            clean_model = fit_quantile_models(x_train, y_train, alpha, cfg["model"], model_seed)
            print("  clean model fitted", flush=True)
            lgbm_model = fit_hist_quantile_models(x_train, y_train, alpha, cfg["model"], model_seed + 1)
            print("  LightGBM baseline fitted", flush=True)
            x_aug, y_aug, w_aug, aug_labels = augment_training_set(
                x_train,
                y_train,
                ramp_train,
                conditions,
                float(cfg["model"]["augmented_condition_fraction"]),
                float(cfg["model"]["ramp_sample_weight"]),
                float(cfg["model"]["degraded_sample_weight"]),
                model_seed + 2,
            )
            robust_model = fit_quantile_models(
                x_aug, y_aug, alpha, cfg["model"], model_seed + 3, sample_weight=w_aug
            )
            print(f"  robust model fitted on {len(x_aug)} rows", flush=True)
            ramp_model = fit_ramp_risk_model(x_train, ramp_train.astype(int), cfg["model"], model_seed + 4)
            print("  ramp-risk model fitted", flush=True)

            run_records.append(
                {
                    "fold": fold_name,
                    "horizon_steps": horizon,
                    "train_n": len(x_train),
                    "augmented_train_n": len(x_aug),
                    "calibration_n": len(x_cal),
                    "test_n": len(x_test),
                    "ramp_threshold_fraction": ramp_threshold,
                    "train_ramp_prevalence": float(ramp_train.mean()),
                    "calibration_ramp_prevalence": float(
                        (meta_cal["ramp_fraction"] >= ramp_threshold).mean()
                    ),
                    "test_ramp_prevalence": float((meta_test["ramp_fraction"] >= ramp_threshold).mean()),
                    "features": x_train.shape[1],
                }
            )

            cal_strategy_parts: dict[str, list[pd.DataFrame]] = {
                "persistence": [],
                "lgbm": [],
                "clean": [],
                "robust": [],
            }
            local_parts: list[pd.DataFrame] = []

            for condition in conditions:
                xc = apply_condition(x_cal, condition)
                risk = ramp_model.predict_proba(xc)
                ramp_group = np.where(meta_cal["ramp_fraction"].to_numpy() >= ramp_threshold, "ramp", "non_ramp")

                clean_lo, clean_med, clean_hi = clean_model.predict(xc)
                robust_lo, robust_med, robust_hi = robust_model.predict(xc)
                lgbm_lo, lgbm_med, lgbm_hi = lgbm_model.predict(xc)
                clean_lo, clean_hi = clip_interval(clean_lo, clean_hi, capacity)
                robust_lo, robust_hi = clip_interval(robust_lo, robust_hi, capacity)
                lgbm_lo, lgbm_hi = clip_interval(lgbm_lo, lgbm_hi, capacity)
                persistence = np.clip(latest_available_power(xc), 0.0, capacity)

                strategy_values = {
                    "persistence": (persistence, persistence, persistence),
                    "lgbm": (lgbm_lo, lgbm_med, lgbm_hi),
                    "clean": (clean_lo, clean_med, clean_hi),
                    "robust": (robust_lo, robust_med, robust_hi),
                }
                for strategy, (lo, med, hi) in strategy_values.items():
                    scores = cqr_scores(y_cal.to_numpy(), lo, hi)
                    cal_strategy_parts[strategy].append(
                        pd.DataFrame(
                            {
                                "timestamp": xc.index,
                                "condition": condition,
                                "ramp_group": ramp_group,
                                "risk": risk,
                                "score": scores,
                                "lower": lo,
                                "median": med,
                                "upper": hi,
                            }
                        )
                    )

                robust_scores = cqr_scores(y_cal.to_numpy(), robust_lo, robust_hi)
                desc = descriptor_frame(
                    xc,
                    condition,
                    risk,
                    robust_lo,
                    robust_med,
                    robust_hi,
                    capacity,
                    conditions,
                )
                desc.insert(0, "timestamp", xc.index)
                desc["condition"] = condition
                desc["ramp_group"] = ramp_group
                desc["score"] = robust_scores
                local_parts.append(desc.reset_index(drop=True))

            cal_frames = {key: pd.concat(parts, ignore_index=True) for key, parts in cal_strategy_parts.items()}
            q_condition = {key: condition_quantiles(value, alpha) for key, value in cal_frames.items()}
            risk_tables = fit_risk_mondrian(
                cal_frames["robust"], alpha, int(cfg["calibration"]["risk_bins"]), shrinkage_k=100.0
            )
            oracle_q = oracle_ramp_quantiles(cal_frames["robust"], alpha)
            local_cal_frame = pd.concat(local_parts, ignore_index=True)
            descriptor_cols = [
                c
                for c in local_cal_frame.columns
                if c
                not in {
                    "timestamp",
                    "condition",
                    "ramp_group",
                    "score",
                }
            ]
            print("  calibration predictions complete", flush=True)
            local_calibrator = fit_local_score_calibrator(
                local_cal_frame,
                descriptor_cols,
                alpha,
                float(cfg["calibration"]["local_fit_fraction"]),
                int(cfg["calibration"]["local_min_samples_leaf"]),
                int(cfg["calibration"]["local_max_iter"]),
                float(cfg["calibration"]["local_learning_rate"]),
                model_seed + 5,
            )
            print("  local calibrator fitted", flush=True)

            for strategy, frame_strategy in cal_frames.items():
                for condition, group in frame_strategy.groupby("condition"):
                    calibration_records.append(
                        {
                            "fold": fold_name,
                            "horizon_steps": horizon,
                            "strategy": strategy,
                            "condition": condition,
                            "n": len(group),
                            "qhat": q_condition[strategy][condition],
                            "mean_score": float(group["score"].mean()),
                        }
                    )

            for condition in conditions:
                xt = apply_condition(x_test, condition)
                risk = ramp_model.predict_proba(xt)
                clean_lo, clean_med, clean_hi = clean_model.predict(xt)
                robust_lo, robust_med, robust_hi = robust_model.predict(xt)
                lgbm_lo, lgbm_med, lgbm_hi = lgbm_model.predict(xt)
                clean_lo, clean_hi = clip_interval(clean_lo, clean_hi, capacity)
                robust_lo, robust_hi = clip_interval(robust_lo, robust_hi, capacity)
                lgbm_lo, lgbm_hi = clip_interval(lgbm_lo, lgbm_hi, capacity)
                persistence = np.clip(latest_available_power(xt), 0.0, capacity)
                method_specs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
                method_specs["persistence_condition_cqr"] = (
                    persistence,
                    persistence,
                    persistence,
                    np.full(len(xt), q_condition["persistence"][condition]),
                )
                method_specs["lgbm_condition_cqr"] = (
                    lgbm_lo,
                    lgbm_med,
                    lgbm_hi,
                    np.full(len(xt), q_condition["lgbm"][condition]),
                )
                method_specs["clean_condition_cqr"] = (
                    clean_lo,
                    clean_med,
                    clean_hi,
                    np.full(len(xt), q_condition["clean"][condition]),
                )
                method_specs["robust_condition_cqr"] = (
                    robust_lo,
                    robust_med,
                    robust_hi,
                    np.full(len(xt), q_condition["robust"][condition]),
                )
                method_specs["robust_risk_mondrian"] = (
                    robust_lo,
                    robust_med,
                    robust_hi,
                    risk_tables.qhat(condition, risk),
                )
                desc_test = descriptor_frame(
                    xt,
                    condition,
                    risk,
                    robust_lo,
                    robust_med,
                    robust_hi,
                    capacity,
                    conditions,
                )
                method_specs["robust_local_cqr"] = (
                    robust_lo,
                    robust_med,
                    robust_hi,
                    local_calibrator.predict(desc_test),
                )
                actual_group = np.where(
                    meta_test["ramp_fraction"].to_numpy() >= ramp_threshold, "ramp", "non_ramp"
                )
                oracle_qhat = np.array([oracle_q[(condition, group)] for group in actual_group], dtype=float)
                method_specs["oracle_ramp_mondrian"] = (
                    robust_lo,
                    robust_med,
                    robust_hi,
                    oracle_qhat,
                )
                test_scores = cqr_scores(y_test.to_numpy(), robust_lo, robust_hi)
                cal_scores = cal_frames["robust"].loc[
                    cal_frames["robust"]["condition"] == condition, "score"
                ].to_numpy()
                rolling = rolling_qhat(
                    cal_scores,
                    test_scores,
                    pd.DatetimeIndex(xt.index),
                    pd.DatetimeIndex(meta_test["target_timestamp"]),
                    alpha,
                    int(cfg["calibration"]["rolling_window"]),
                    int(cfg["calibration"]["rolling_min_history"]),
                )
                method_specs["robust_rolling_cqr"] = (
                    robust_lo,
                    robust_med,
                    robust_hi,
                    rolling,
                )

                for method in cfg["methods"]:
                    base_lo, med, base_hi, qhat = method_specs[method]
                    lo, hi = adjust_interval(base_lo, base_hi, qhat, capacity)
                    all_predictions.append(
                        make_prediction_rows(
                            fold_name,
                            horizon,
                            condition,
                            method,
                            meta_test,
                            y_test,
                            np.clip(med, 0.0, capacity),
                            lo,
                            hi,
                            risk,
                            ramp_threshold,
                        )
                    )

            rss_mb = peak_rss_mb()
            rss_text = f"{rss_mb:.0f}MB" if rss_mb is not None else "unavailable"
            print(
                f"completed {fold_name}, horizon={horizon}, elapsed={time.time() - started:.1f}s, peak_rss={rss_text}",
                flush=True,
            )
            del clean_model, lgbm_model, robust_model, ramp_model, local_calibrator
            gc.collect()

    predictions = pd.concat(all_predictions, ignore_index=True)
    if args.chunk_only:
        chunks = out_cache / "chunks"
        chunks.mkdir(parents=True, exist_ok=True)
        fold_name = str(run_records[0]["fold"])
        horizon = int(run_records[0]["horizon_steps"])
        stem = f"{fold_name}_h{horizon}"
        predictions.to_pickle(chunks / f"predictions_{stem}.pkl.gz", compression="gzip")
        pd.DataFrame(run_records).to_csv(chunks / f"run_{stem}.csv", index=False)
        pd.DataFrame(calibration_records).to_csv(chunks / f"calibration_{stem}.csv", index=False)
        print(f"saved chunk {stem}: {len(predictions):,} rows", flush=True)
        return
    predictions.to_pickle(out_cache / "advanced_predictions.pkl.gz", compression="gzip")
    predictions.to_csv(out_cache / "advanced_predictions.csv.gz", index=False, compression="gzip")
    summary = summarize_advanced(
        predictions, alpha, int(cfg["calibration"]["min_group_n"]), capacity
    )
    aggregate = aggregate_methods(summary)
    fold_scores = fold_method_scores(summary)
    sensitivity = ramp_sensitivity_table(predictions, cfg, alpha)

    candidate_methods = ["robust_risk_mondrian", "robust_local_cqr", "robust_rolling_cqr"]
    print("starting day-cluster bootstrap", flush=True)
    bootstrap_parts = [
        bootstrap_worst_group_difference(
            predictions,
            candidate,
            "robust_condition_cqr",
            alpha,
            int(cfg["bootstrap"]["replicates"]),
            seed + i,
        )
        for i, candidate in enumerate(candidate_methods)
    ]
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    print("bootstrap complete", flush=True)

    pd.DataFrame(run_records).to_csv(out_tables / "fold_model_summary.csv", index=False)
    pd.DataFrame(calibration_records).to_csv(out_tables / "calibration_quantiles.csv", index=False)
    summary.to_csv(out_tables / "conditional_metrics.csv", index=False)
    aggregate.to_csv(out_tables / "method_aggregate.csv", index=False)
    fold_scores.to_csv(out_tables / "fold_method_scores.csv", index=False)
    sensitivity.to_csv(out_tables / "ramp_threshold_sensitivity.csv", index=False)
    bootstrap.to_csv(out_tables / "bootstrap_worst_group_difference.csv", index=False)

    baseline = aggregate[aggregate["method"] == "robust_condition_cqr"].set_index("horizon_steps")
    deployable = aggregate[aggregate["method"].isin(candidate_methods)].copy()
    best_rows = (
        deployable.sort_values(
            ["horizon_steps", "worst_group_undercoverage", "mean_normalized_interval_score"]
        )
        .groupby("horizon_steps", as_index=False)
        .head(1)
    )
    gate_records = []
    for _, best in best_rows.iterrows():
        h = int(best["horizon_steps"])
        base = baseline.loc[h]
        per_fold = fold_scores[
            (fold_scores["horizon_steps"] == h)
            & (fold_scores["method"].isin([best["method"], "robust_condition_cqr"]))
        ].pivot(index="fold", columns="method", values="worst_undercoverage")
        fold_wins = int((per_fold[best["method"]] + 0.02 < per_fold["robust_condition_cqr"]).sum())
        boot = bootstrap[
            (bootstrap["horizon_steps"] == h) & (bootstrap["method_a"] == best["method"])
        ].iloc[0]
        gate_records.append(
            {
                "horizon_steps": h,
                "best_method": best["method"],
                "baseline_worst_undercoverage": float(base["worst_group_undercoverage"]),
                "best_worst_undercoverage": float(best["worst_group_undercoverage"]),
                "improvement": float(
                    base["worst_group_undercoverage"] - best["worst_group_undercoverage"]
                ),
                "fold_wins_ge_0_02": fold_wins,
                "bootstrap_ci_low": float(boot["ci_low"]),
                "bootstrap_ci_high": float(boot["ci_high"]),
                "clean_score_ratio": float(
                    best["clean_normalized_interval_score"]
                    / base["clean_normalized_interval_score"]
                ),
            }
        )
    gates = pd.DataFrame(gate_records)
    gates.to_csv(out_tables / "method_gates.csv", index=False)

    phenomenon = bool(
        summary[
            (summary["method"] == "robust_condition_cqr")
            & (summary["ramp_group"] == "ramp")
        ]["undercoverage_error"].max()
        >= 0.10
    )
    comparative_improvement = bool(
        (gates["improvement"] >= 0.05).all()
        and (gates["fold_wins_ge_0_02"] >= 3).all()
        and (gates["bootstrap_ci_high"] < 0).all()
        and (gates["clean_score_ratio"] <= 1.10).all()
    )
    absolute_reliability = bool((gates["best_worst_undercoverage"] <= 0.10).all())
    if phenomenon and comparative_improvement:
        verdict = "GO_FULL_PAPER_EVALUATION_METHOD_PROMISING_SINGLE_SITE"
    elif phenomenon:
        verdict = "GO_FULL_PAPER_EVALUATION_REQUIRES_EXTERNAL_SITE"
    else:
        verdict = "GO_SHORT_PAPER_OR_REDESIGN"
    decision = {
        "status": "ADVANCED_TEMPORAL_REPLICATION_COMPLETE",
        "verdict": verdict,
        "method_superiority_claim_allowed": False,
        "single_site_limitation": True,
        "phenomenon_replication": phenomenon,
        "comparative_improvement_gate_passed": comparative_improvement,
        "absolute_reliability_gate_passed": absolute_reliability,
        "absolute_reliability_threshold_worst_undercoverage": 0.10,
        "best_by_horizon": best_rows.to_dict(orient="records"),
        "gate_records": gate_records,
        "runtime_seconds": round(time.time() - started, 2),
        "data_sha256": sha256(data_path),
        "config_sha256": sha256(cfg_path),
    }
    (out_tables / "advanced_verdict.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
