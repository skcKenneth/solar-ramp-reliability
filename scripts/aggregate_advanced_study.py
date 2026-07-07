from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_advanced_study import (  # noqa: E402
    aggregate_methods,
    bootstrap_worst_group_difference,
    fold_method_scores,
    ramp_sensitivity_table,
    sha256,
    summarize_advanced,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/advanced.yaml")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    started = time.time()

    out_root = ROOT / "outputs" / "advanced"
    out_tables = out_root / "tables"
    out_cache = out_root / "cache"
    chunks = out_cache / "chunks"
    out_tables.mkdir(parents=True, exist_ok=True)

    prediction_paths = sorted(chunks.glob("predictions_*.pkl.gz"))
    expected = len(cfg["data"]["folds"]) * len(cfg["forecast"]["horizons_steps"])
    consolidated_cache = out_cache / "advanced_predictions.pkl.gz"
    csv_cache = out_cache / "advanced_predictions.csv.gz"
    if len(prediction_paths) == expected:
        print(f"loading {len(prediction_paths)} chunks", flush=True)
        predictions = pd.concat(
            [pd.read_pickle(path, compression="gzip") for path in prediction_paths],
            ignore_index=True,
        )
    elif consolidated_cache.exists():
        print(
            f"found {len(prediction_paths)} of {expected} chunks; loading consolidated cache",
            flush=True,
        )
        predictions = pd.read_pickle(consolidated_cache, compression="gzip")
    elif csv_cache.exists():
        print(
            f"found {len(prediction_paths)} of {expected} chunks; loading csv cache",
            flush=True,
        )
        predictions = pd.read_csv(csv_cache, compression="gzip")
    else:
        raise RuntimeError(
            f"Expected {expected} prediction chunks, {consolidated_cache}, or {csv_cache}, "
            f"found {len(prediction_paths)} chunks"
        )
    for column in ["fold", "condition", "method", "ramp_group", "ramp_direction"]:
        predictions[column] = predictions[column].astype("category")
    predictions.to_pickle(out_cache / "advanced_predictions.pkl.gz", compression="gzip")

    run_paths = sorted(chunks.glob("run_*.csv"))
    calibration_paths = sorted(chunks.glob("calibration_*.csv"))
    if run_paths:
        run_records = pd.concat([pd.read_csv(path) for path in run_paths], ignore_index=True)
    else:
        existing_run_records = out_tables / "fold_model_summary.csv"
        if not existing_run_records.exists():
            raise RuntimeError("Missing run chunk files and existing fold_model_summary.csv")
        print("reusing existing fold_model_summary.csv", flush=True)
        run_records = pd.read_csv(existing_run_records)
    if calibration_paths:
        calibration_records = pd.concat(
            [pd.read_csv(path) for path in calibration_paths],
            ignore_index=True,
        )
    else:
        existing_calibration = out_tables / "calibration_quantiles.csv"
        if not existing_calibration.exists():
            raise RuntimeError("Missing calibration chunk files and existing calibration_quantiles.csv")
        print("reusing existing calibration_quantiles.csv", flush=True)
        calibration_records = pd.read_csv(existing_calibration)

    alpha = float(cfg["project"]["alpha"])
    capacity = float(cfg["project"]["nominal_capacity_mw"])
    seed = int(cfg["project"]["seed"])
    print(f"aggregating {len(predictions):,} prediction rows", flush=True)
    summary = summarize_advanced(predictions, alpha, int(cfg["calibration"]["min_group_n"]), capacity)
    aggregate = aggregate_methods(summary)
    fold_scores = fold_method_scores(summary)
    sensitivity = ramp_sensitivity_table(predictions, cfg, alpha)

    candidate_methods = ["robust_risk_mondrian", "robust_local_cqr", "robust_rolling_cqr"]
    print("running clustered bootstrap", flush=True)
    bootstrap = pd.concat(
        [
            bootstrap_worst_group_difference(
                predictions,
                candidate,
                "robust_condition_cqr",
                alpha,
                int(cfg["bootstrap"]["replicates"]),
                seed + i,
            )
            for i, candidate in enumerate(candidate_methods)
        ],
        ignore_index=True,
    )

    run_records.to_csv(out_tables / "fold_model_summary.csv", index=False)
    calibration_records.to_csv(out_tables / "calibration_quantiles.csv", index=False)
    summary.to_csv(out_tables / "conditional_metrics.csv", index=False)
    aggregate.to_csv(out_tables / "method_aggregate.csv", index=False)
    fold_scores.to_csv(out_tables / "fold_method_scores.csv", index=False)
    sensitivity.to_csv(out_tables / "ramp_threshold_sensitivity.csv", index=False)
    bootstrap.to_csv(out_tables / "bootstrap_worst_group_difference.csv", index=False)

    baseline = aggregate[aggregate["method"] == "robust_condition_cqr"].set_index("horizon_steps")
    deployable = aggregate[aggregate["method"].isin(candidate_methods)].copy()
    best_rows = (
        deployable.sort_values(["horizon_steps", "worst_group_undercoverage", "mean_normalized_interval_score"])
        .groupby("horizon_steps", as_index=False)
        .head(1)
    )
    gate_records: list[dict] = []
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
                "improvement": float(base["worst_group_undercoverage"] - best["worst_group_undercoverage"]),
                "fold_wins_ge_0_02": fold_wins,
                "bootstrap_ci_low": float(boot["ci_low"]),
                "bootstrap_ci_high": float(boot["ci_high"]),
                "clean_score_ratio": float(best["clean_normalized_interval_score"] / base["clean_normalized_interval_score"]),
            }
        )
    gates = pd.DataFrame(gate_records)
    gates.to_csv(out_tables / "method_gates.csv", index=False)

    phenomenon = bool(
        summary[(summary["method"] == "robust_condition_cqr") & (summary["ramp_group"] == "ramp")][
            "undercoverage_error"
        ].max()
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
        "prediction_rows": int(len(predictions)),
        "runtime_seconds": round(time.time() - started, 2),
        "data_sha256": sha256(ROOT / cfg["data"]["path"]),
        "config_sha256": sha256(cfg_path),
    }
    (out_tables / "advanced_verdict.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
