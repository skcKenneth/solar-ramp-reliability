from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prediction_metrics import (  # noqa: E402
    aggregate_prediction_metrics,
    prepare_prediction_metric_frame,
    require_complete_method_condition_rows,
    require_full_availability,
)


def candidate_methods(cfg: dict) -> list[str]:
    return list(cfg["methods"]["primary_deployable_candidates"])


def valid_confirmatory_stations(cfg: dict) -> list[str]:
    audit = pd.read_csv(ROOT / cfg["data"]["station_audit_table"])
    development = set(cfg["project"].get("development_sites", []))
    valid = audit[(audit["status"] == "valid") & (~audit["site_id"].isin(development))]
    return sorted(valid["site_id"].astype(str).tolist())


def capacity_map(cfg: dict) -> dict[str, float]:
    manifest = pd.read_csv(ROOT / cfg["data"]["manifest"])
    return {
        str(row.site_id): float(row.nominal_capacity_mw)
        for row in manifest.itertuples(index=False)
    }


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    score = hi - lo
    score += (2 / alpha) * (lo - y) * (y < lo)
    score += (2 / alpha) * (y - hi) * (y > hi)
    return score


def chunk_paths(cfg: dict, stations: list[str]) -> list[Path]:
    output_root = ROOT / cfg["outputs"]["root"] / "stations"
    expected = len(stations) * len(cfg["data"]["site1_calendar_folds"]) * len(cfg["forecast"]["horizons_steps"])
    paths = sorted(
        path
        for station in stations
        for path in (output_root / station / "cache" / "chunks").glob("predictions_*.pkl.gz")
    )
    if len(paths) != expected:
        raise RuntimeError(f"Expected {expected} chunks for {stations}, found {len(paths)}")
    return paths


def summarize_chunk(
    path: Path,
    alpha: float,
    min_group_n: int,
    capacity: float,
    station: str,
    *,
    required_fully_usable_methods: Iterable[str] = (),
    expected_methods: Iterable[str] | None = None,
    expected_conditions: Iterable[str] | None = None,
    baseline_method: str = "robust_condition_cqr",
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    raw_predictions = pd.read_pickle(path, compression="gzip")
    if expected_methods is not None and expected_conditions is not None:
        require_complete_method_condition_rows(
            raw_predictions,
            expected_methods=expected_methods,
            expected_conditions=expected_conditions,
            baseline_method=baseline_method,
            context=f"{path.name} ({station})",
        )
    pred = prepare_prediction_metric_frame(
        raw_predictions,
        alpha=alpha,
        additional_finite_columns=("ramp_fraction",),
    )
    pred["station"] = station
    require_full_availability(
        pred,
        required_fully_usable_methods,
        context=f"{path.name} ({station})",
    )
    directional_rows: list[dict] = []
    risk_rows: list[dict] = []
    physical_counts = (
        pred[pred["method"] == "robust_condition_cqr"]
        .groupby(
            ["fold", "horizon_steps", "condition", "ramp_group"],
            observed=True,
        )
        .size()
        .rename("physical_group_n")
        .reset_index()
    )
    physical_counts["condition_n"] = physical_counts.groupby(
        ["fold", "horizon_steps", "condition"], observed=True
    )["physical_group_n"].transform("sum")
    physical_counts["ramp_prevalence"] = (
        physical_counts["physical_group_n"] / physical_counts["condition_n"]
    )
    group_cols = ["station", "fold", "horizon_steps", "method", "condition", "ramp_group"]
    all_conditional = aggregate_prediction_metrics(
        pred,
        group_columns=group_cols,
        alpha=alpha,
        capacity=capacity,
    ).merge(
        physical_counts[
            [
                "fold",
                "horizon_steps",
                "condition",
                "ramp_group",
                "physical_group_n",
                "condition_n",
                "ramp_prevalence",
            ]
        ],
        on=["fold", "horizon_steps", "condition", "ramp_group"],
        how="left",
        validate="many_to_one",
    )
    if all_conditional[
        ["physical_group_n", "condition_n", "ramp_prevalence"]
    ].isna().any().any():
        raise RuntimeError(f"{path.name}: physical ramp counts are incomplete")
    all_conditional["eligible_for_conditional_metrics"] = (
        all_conditional["usable_n"] >= int(min_group_n)
    )
    availability_columns = [
        *group_cols,
        "attempted_n",
        "usable_n",
        "unavailable_n",
        "availability_rate",
        "covered_n",
        "issued_coverage",
        "service_coverage",
        "eligible_for_conditional_metrics",
        "metric_denominator",
        "service_denominator",
    ]
    availability_rows = all_conditional[availability_columns].to_dict(orient="records")
    conditional_rows = all_conditional[
        all_conditional["eligible_for_conditional_metrics"]
    ].drop(columns=["physical_group_n", "condition_n"]).to_dict(orient="records")

    risk_source = pred[pred["method"] == "robust_condition_cqr"].copy()
    for key, group in risk_source.groupby(["station", "fold", "horizon_steps", "condition"], observed=True, sort=True):
        truth = (group["ramp_group"].astype(str) == "ramp").astype(int).to_numpy()
        risk = group["transition_risk"].to_numpy(dtype=float)
        positives = int(truth.sum())
        top_n = max(1, int(np.ceil(0.10 * len(group))))
        top_idx = np.argsort(risk)[-top_n:]
        try:
            from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

            roc_auc = float(roc_auc_score(truth, risk)) if len(np.unique(truth)) == 2 else np.nan
            average_precision = float(average_precision_score(truth, risk)) if positives else np.nan
            brier = float(brier_score_loss(truth, risk))
        except Exception:
            roc_auc = np.nan
            average_precision = np.nan
            brier = np.nan
        risk_rows.append(
            {
                "station": str(key[0]),
                "fold": str(key[1]),
                "horizon_steps": int(key[2]),
                "condition": str(key[3]),
                "n": int(len(group)),
                "ramp_n": positives,
                "ramp_prevalence": float(truth.mean()),
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "brier": brier,
                "top_decile_ramp_recall": float(truth[top_idx].sum() / positives) if positives else np.nan,
            }
        )

    ramp = pred[pred["ramp_group"].astype(str) == "ramp"].copy()
    directional_group_cols = [
        "station",
        "fold",
        "horizon_steps",
        "method",
        "condition",
        "ramp_direction",
    ]
    if not ramp.empty:
        directional = aggregate_prediction_metrics(
            ramp,
            group_columns=directional_group_cols,
            alpha=alpha,
            capacity=capacity,
        )
        directional = directional[directional["usable_n"] >= 40].copy()
        directional["mean_interval_score"] = directional["interval_score"]
        directional_rows = directional.to_dict(orient="records")
    return conditional_rows, risk_rows, directional_rows, availability_rows


def aggregate_methods(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (station, horizon, method), group in summary.groupby(["station", "horizon_steps", "method"], sort=True, observed=True):
        clean = group[group["condition"] == "clean"]
        rows.append(
            {
                "station": station,
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
                "attempted_predictions": int(group["attempted_n"].sum()),
                "usable_predictions": int(group["usable_n"].sum()),
                "unavailable_predictions": int(group["unavailable_n"].sum()),
                "prediction_availability": float(
                    group["usable_n"].sum() / group["attempted_n"].sum()
                ),
                "minimum_group_availability": float(group["availability_rate"].min()),
                "mean_group_service_coverage": float(group["service_coverage"].mean()),
                "worst_group_service_undercoverage": float(
                    group["service_undercoverage_error"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def fold_scores(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["station", "fold", "horizon_steps", "method"], as_index=False, observed=True)
        .agg(
            worst_undercoverage=("undercoverage_error", "max"),
            mean_abs_coverage_error=("coverage_error", "mean"),
            mean_normalized_width=("normalized_width", "mean"),
            mean_normalized_interval_score=("normalized_interval_score", "mean"),
            attempted_predictions=("attempted_n", "sum"),
            usable_predictions=("usable_n", "sum"),
            unavailable_predictions=("unavailable_n", "sum"),
            minimum_group_availability=("availability_rate", "min"),
            mean_group_service_coverage=("service_coverage", "mean"),
        )
    )


def station_effects(method_aggregate: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    baseline = method_aggregate[method_aggregate["method"] == cfg["methods"]["primary_baseline"]]
    for method in candidate_methods(cfg) + list(cfg["methods"]["non_deployable_diagnostics"]):
        comp = method_aggregate[method_aggregate["method"] == method]
        merged = baseline.merge(
            comp,
            on=["station", "horizon_steps"],
            suffixes=("_baseline", "_method"),
        )
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "station": row.station,
                    "horizon_steps": int(row.horizon_steps),
                    "method": method,
                    "baseline_worst_undercoverage": float(row.worst_group_undercoverage_baseline),
                    "method_worst_undercoverage": float(row.worst_group_undercoverage_method),
                    "improvement": float(row.worst_group_undercoverage_baseline - row.worst_group_undercoverage_method),
                    "delta_method_minus_baseline": float(row.worst_group_undercoverage_method - row.worst_group_undercoverage_baseline),
                    "clean_score_ratio": float(
                        row.clean_normalized_interval_score_method / row.clean_normalized_interval_score_baseline
                    ),
                    "method_mean_normalized_interval_score": float(row.mean_normalized_interval_score_method),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_station_delta(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(replicates, values.size), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def stable_method_offset(method: str) -> int:
    return sum((i + 1) * ord(char) for i, char in enumerate(method))


def candidate_gates(
    effects: pd.DataFrame,
    cfg: dict,
    expected_stations: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    reps = int(cfg["bootstrap"]["station_outer_replicates"])
    expected_station_set = set(
        map(str, expected_stations)
        if expected_stations is not None
        else effects["station"].astype(str).unique()
    )
    for horizon in map(int, cfg["forecast"]["horizons_steps"]):
        for method in candidate_methods(cfg):
            group = effects[
                (effects["horizon_steps"].astype(int) == horizon)
                & (effects["method"].astype(str) == method)
            ]
            rows.append(
                _candidate_gate_row(
                    group,
                    cfg,
                    horizon=horizon,
                    method=method,
                    expected_station_set=expected_station_set,
                    replicates=reps,
                )
            )
    return pd.DataFrame(rows)


def _candidate_gate_row(
    group: pd.DataFrame,
    cfg: dict,
    *,
    horizon: int,
    method: str,
    expected_station_set: set[str],
    replicates: int,
) -> dict:
    observed_station_set = set(group["station"].astype(str))
    missing_stations = sorted(expected_station_set - observed_station_set)
    unexpected_stations = sorted(observed_station_set - expected_station_set)
    station_set_complete = not missing_stations and not unexpected_stations
    mean_delta, ci_low, ci_high = bootstrap_station_delta(
        group["delta_method_minus_baseline"].to_numpy(),
        replicates,
        int(cfg["project"]["seed"]) + int(horizon) * 101 + stable_method_offset(method),
    )
    improvements = group["improvement"].to_numpy(dtype=float)
    mean_improvement = float(improvements.mean()) if improvements.size else np.nan
    return {
                "horizon_steps": int(horizon),
                "method": method,
                "stations": int(group["station"].nunique()),
                "expected_stations": int(len(expected_station_set)),
                "station_set_complete": bool(station_set_complete),
                "missing_stations": ";".join(missing_stations),
                "unexpected_stations": ";".join(unexpected_stations),
                "mean_improvement": mean_improvement,
                "median_improvement": float(np.median(improvements)) if improvements.size else np.nan,
                "min_improvement": float(improvements.min()) if improvements.size else np.nan,
                "max_improvement": float(improvements.max()) if improvements.size else np.nan,
                "stations_improved_n": int((improvements > 0).sum()),
                "stations_improved_prop": float((improvements > 0).mean()) if improvements.size else 0.0,
                "stations_improved_ge_0_05_n": int((improvements >= 0.05).sum()),
                "mean_delta_method_minus_baseline": mean_delta,
                "station_bootstrap_ci_low": ci_low,
                "station_bootstrap_ci_high": ci_high,
                "max_method_worst_undercoverage": float(group["method_worst_undercoverage"].max()) if not group.empty else np.nan,
                "mean_clean_score_ratio": float(group["clean_score_ratio"].mean()) if not group.empty else np.nan,
                "comparative_gate_passed": bool(
                    station_set_complete
                    and mean_improvement >= float(cfg["decision_rules"]["comparative_gate"]["improvement_over_baseline_at_each_horizon"])
                    and (improvements > 0).mean()
                    >= float(cfg["decision_rules"]["comparative_gate"]["minimum_station_proportion_improved"])
                    and ci_high < 0
                    and group["clean_score_ratio"].mean()
                    <= float(cfg["decision_rules"]["comparative_gate"]["clean_context_score_ratio_max"])
                ),
                "absolute_reliability_gate_passed": bool(
                    station_set_complete
                    and group["method_worst_undercoverage"].max()
                    <= float(cfg["decision_rules"]["absolute_reliability_gate"]["max_worst_group_undercoverage_each_horizon"])
                ),
        }


def leave_one_station_out(effects: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for (horizon, method), group in effects[
        effects["method"].isin(candidate_methods(cfg))
    ].groupby(["horizon_steps", "method"], sort=True):
        for held_out in sorted(group["station"].unique()):
            sub = group[group["station"] != held_out]
            rows.append(
                {
                    "horizon_steps": int(horizon),
                    "method": method,
                    "held_out_station": held_out,
                    "remaining_stations": int(sub["station"].nunique()),
                    "mean_improvement": float(sub["improvement"].mean()),
                    "stations_improved_prop": float((sub["improvement"] > 0).mean()),
                    "max_method_worst_undercoverage": float(sub["method_worst_undercoverage"].max()),
                }
            )
    return pd.DataFrame(rows)


def oracle_gap(method_aggregate: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    baseline = method_aggregate[method_aggregate["method"] == cfg["methods"]["primary_baseline"]]
    local = method_aggregate[method_aggregate["method"] == "robust_local_cqr"]
    oracle = method_aggregate[method_aggregate["method"] == "oracle_ramp_mondrian"]
    merged = baseline.merge(local, on=["station", "horizon_steps"], suffixes=("_baseline", "_local")).merge(
        oracle,
        on=["station", "horizon_steps"],
    )
    for row in merged.itertuples(index=False):
        rows.append(
            {
                "station": row.station,
                "horizon_steps": int(row.horizon_steps),
                "baseline_worst_undercoverage": float(row.worst_group_undercoverage_baseline),
                "local_worst_undercoverage": float(row.worst_group_undercoverage_local),
                "oracle_worst_undercoverage": float(row.worst_group_undercoverage),
                "recoverable_gap_base_to_oracle": float(
                    row.worst_group_undercoverage_baseline - row.worst_group_undercoverage
                ),
                "remaining_local_to_oracle_gap": float(
                    row.worst_group_undercoverage_local - row.worst_group_undercoverage
                ),
                "local_interval_score": float(row.mean_normalized_interval_score_local),
                "oracle_interval_score": float(row.mean_normalized_interval_score),
            }
        )
    return pd.DataFrame(rows)


def decision_payload(
    cfg: dict,
    stations: list[str],
    conditional: pd.DataFrame,
    gates: pd.DataFrame,
    effects: pd.DataFrame,
    gap: pd.DataFrame,
    runtime: float,
) -> dict:
    baseline_ramp = conditional[
        (conditional["method"] == cfg["methods"]["primary_baseline"])
        & (conditional["ramp_group"].astype(str) == "ramp")
    ]
    phenomenon_by_station = (
        baseline_ramp.groupby(["station", "horizon_steps"], observed=True)["undercoverage_error"].max().reset_index()
    )
    phenomenon_support = (
        phenomenon_by_station.groupby("horizon_steps")["undercoverage_error"]
        .apply(lambda x: int((x >= 0.10).sum()))
        .to_dict()
    )
    local_gap = gap.groupby("horizon_steps")["remaining_local_to_oracle_gap"].median().to_dict()
    comparative_by_method = (
        gates.groupby("method")["comparative_gate_passed"].all().to_dict()
        if not gates.empty
        else {}
    )
    absolute_by_method = (
        gates.groupby("method")["absolute_reliability_gate_passed"].all().to_dict()
        if not gates.empty
        else {}
    )
    method_gate_methods = [m for m in comparative_by_method if comparative_by_method[m] and absolute_by_method.get(m, False)]
    eval_supported = bool(
        max(phenomenon_support.values() or [0]) >= max(2, int(np.ceil(0.60 * len(stations))))
        or any(v > 0.05 for v in local_gap.values())
    )
    if method_gate_methods:
        verdict = cfg["decision_rules"]["labels"]["full_method"]
    elif eval_supported:
        verdict = cfg["decision_rules"]["labels"]["full_evaluation"]
    elif len(stations) >= 3:
        verdict = cfg["decision_rules"]["labels"]["partial_criteria"]
    else:
        verdict = cfg["decision_rules"]["labels"]["no_go"]
    return {
        "status": "MULTISITE_CONFIRMATORY_COMPLETE",
        "verdict": verdict,
        "valid_confirmatory_stations": stations,
        "station_count": len(stations),
        "prediction_chunks": int(len(chunk_paths(cfg, stations))),
        "conditional_metric_rows": int(len(conditional)),
        "phenomenon_station_counts_by_horizon": {str(k): int(v) for k, v in phenomenon_support.items()},
        "median_remaining_local_to_oracle_gap_by_horizon": {str(k): float(v) for k, v in local_gap.items()},
        "candidate_comparative_gate_all_horizons": comparative_by_method,
        "candidate_absolute_gate_all_horizons": absolute_by_method,
        "method_gate_methods": method_gate_methods,
        "method_superiority_claim_allowed": bool(method_gate_methods),
        "site1_counted_as_confirmation": False,
        "runtime_seconds": round(runtime, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    started = time.time()
    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tables = ROOT / cfg["outputs"]["tables"]
    tables.mkdir(parents=True, exist_ok=True)
    stations = valid_confirmatory_stations(cfg)
    capacities = capacity_map(cfg)
    paths = chunk_paths(cfg, stations)

    conditional_rows: list[dict] = []
    risk_rows: list[dict] = []
    directional_rows: list[dict] = []
    availability_rows: list[dict] = []
    decision_methods = [
        str(cfg["methods"]["primary_baseline"]),
        *map(str, cfg["methods"]["primary_deployable_candidates"]),
    ]
    for path in paths:
        station = path.parts[path.parts.index("stations") + 1]
        c_rows, r_rows, d_rows, a_rows = summarize_chunk(
            path,
            float(cfg["project"]["alpha"]),
            int(cfg["calibration"]["min_group_n"]),
            capacities[station],
            station,
            required_fully_usable_methods=decision_methods,
            expected_methods=cfg["methods"]["full_set"],
            expected_conditions=cfg["conditions"],
            baseline_method=str(cfg["methods"]["primary_baseline"]),
        )
        conditional_rows.extend(c_rows)
        risk_rows.extend(r_rows)
        directional_rows.extend(d_rows)
        availability_rows.extend(a_rows)
        print(f"summarized {path}", flush=True)

    conditional = pd.DataFrame(conditional_rows)
    risk = pd.DataFrame(risk_rows)
    directional = pd.DataFrame(directional_rows)
    availability = pd.DataFrame(availability_rows)
    method_agg = aggregate_methods(conditional)
    fold = fold_scores(conditional)
    effects = station_effects(method_agg, cfg)
    gates = candidate_gates(effects, cfg, expected_stations=stations)
    loo = leave_one_station_out(effects, cfg)
    gap = oracle_gap(method_agg, cfg)

    base = conditional[conditional.method == cfg["methods"]["primary_baseline"]].copy()
    local = conditional[conditional.method == "robust_local_cqr"].copy()
    keys = ["station", "fold", "horizon_steps", "condition", "ramp_group"]
    paired = base.merge(local, on=keys, suffixes=("_base", "_local"))
    paired["undercoverage_improvement"] = paired["undercoverage_error_base"] - paired["undercoverage_error_local"]
    paired["coverage_error_improvement"] = paired["coverage_error_base"] - paired["coverage_error_local"]
    paired["interval_score_change"] = paired["interval_score_local"] - paired["interval_score_base"]
    paired["width_change"] = paired["mean_width_local"] - paired["mean_width_base"]

    conditional.to_csv(tables / "conditional_metrics.csv", index=False)
    method_agg.to_csv(tables / "station_method_aggregate.csv", index=False)
    fold.to_csv(tables / "fold_method_scores.csv", index=False)
    effects.to_csv(tables / "station_level_effects.csv", index=False)
    gates.to_csv(tables / "candidate_station_gates.csv", index=False)
    loo.to_csv(tables / "leave_one_station_out.csv", index=False)
    gap.to_csv(tables / "oracle_gap_summary.csv", index=False)
    paired.to_csv(tables / "paired_group_improvements.csv", index=False)
    risk.to_csv(tables / "ramp_risk_diagnostics.csv", index=False)
    directional.to_csv(tables / "directional_ramp_metrics.csv", index=False)
    availability.to_csv(tables / "prediction_availability.csv", index=False)

    decision = decision_payload(cfg, stations, conditional, gates, effects, gap, time.time() - started)
    (tables / "multisite_verdict.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
