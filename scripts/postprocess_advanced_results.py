from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "advanced" / "cache" / "advanced_predictions.pkl.gz"
TABLES = ROOT / "outputs" / "advanced" / "tables"


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    score = hi - lo
    score += (2 / alpha) * (lo - y) * (y < lo)
    score += (2 / alpha) * (y - hi) * (y > hi)
    return score


def main() -> None:
    pred = pd.read_pickle(CACHE, compression="gzip")
    alpha = 0.10
    target = 1 - alpha

    risk_source = pred[pred["method"] == "robust_condition_cqr"].copy()
    risk_rows: list[dict] = []
    for key, group in risk_source.groupby(
        ["fold", "horizon_steps", "condition"], observed=True, sort=True
    ):
        truth = (group["ramp_group"].astype(str) == "ramp").astype(int).to_numpy()
        risk = group["transition_risk"].to_numpy(dtype=float)
        top_n = max(1, int(np.ceil(0.10 * len(group))))
        top_idx = np.argsort(risk)[-top_n:]
        positives = int(truth.sum())
        risk_rows.append(
            {
                "fold": str(key[0]),
                "horizon_steps": int(key[1]),
                "condition": str(key[2]),
                "n": int(len(group)),
                "ramp_n": positives,
                "ramp_prevalence": float(truth.mean()),
                "roc_auc": float(roc_auc_score(truth, risk)) if len(np.unique(truth)) == 2 else np.nan,
                "average_precision": float(average_precision_score(truth, risk)) if positives else np.nan,
                "brier": float(brier_score_loss(truth, risk)),
                "top_decile_ramp_recall": float(truth[top_idx].sum() / positives) if positives else np.nan,
            }
        )
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(TABLES / "ramp_risk_diagnostics.csv", index=False)

    ramp = pred[pred["ramp_group"].astype(str) == "ramp"].copy()
    direction_rows: list[dict] = []
    for key, group in ramp.groupby(
        ["fold", "horizon_steps", "method", "condition", "ramp_direction"],
        observed=True,
        sort=True,
    ):
        if len(group) < 40:
            continue
        coverage = float(group["covered"].mean())
        direction_rows.append(
            {
                "fold": str(key[0]),
                "horizon_steps": int(key[1]),
                "method": str(key[2]),
                "condition": str(key[3]),
                "ramp_direction": str(key[4]),
                "n": int(len(group)),
                "coverage": coverage,
                "undercoverage_error": max(0.0, target - coverage),
                "mean_width": float(group["interval_width"].mean()),
                "mean_interval_score": float(
                    interval_score(
                        group["y"].to_numpy(), group["lower"].to_numpy(), group["upper"].to_numpy(), alpha
                    ).mean()
                ),
            }
        )
    pd.DataFrame(direction_rows).to_csv(TABLES / "directional_ramp_metrics.csv", index=False)

    metrics = pd.read_csv(TABLES / "conditional_metrics.csv")
    base = metrics[metrics.method == "robust_condition_cqr"].copy()
    best = metrics[metrics.method == "robust_local_cqr"].copy()
    keys = ["fold", "horizon_steps", "condition", "ramp_group"]
    paired = base.merge(best, on=keys, suffixes=("_base", "_local"))
    paired["undercoverage_improvement"] = (
        paired["undercoverage_error_base"] - paired["undercoverage_error_local"]
    )
    paired["coverage_error_improvement"] = paired["coverage_error_base"] - paired["coverage_error_local"]
    paired["interval_score_change"] = paired["interval_score_local"] - paired["interval_score_base"]
    paired["width_change"] = paired["mean_width_local"] - paired["mean_width_base"]
    paired.to_csv(TABLES / "paired_group_improvements.csv", index=False)

    aggregate = pd.read_csv(TABLES / "method_aggregate.csv")
    oracle = aggregate[aggregate.method == "oracle_ramp_mondrian"].set_index("horizon_steps")
    local = aggregate[aggregate.method == "robust_local_cqr"].set_index("horizon_steps")
    base_agg = aggregate[aggregate.method == "robust_condition_cqr"].set_index("horizon_steps")
    rows = []
    for horizon in sorted(oracle.index):
        rows.append(
            {
                "horizon_steps": int(horizon),
                "baseline_worst_undercoverage": float(base_agg.loc[horizon, "worst_group_undercoverage"]),
                "local_worst_undercoverage": float(local.loc[horizon, "worst_group_undercoverage"]),
                "oracle_worst_undercoverage": float(oracle.loc[horizon, "worst_group_undercoverage"]),
                "recoverable_gap_base_to_oracle": float(
                    base_agg.loc[horizon, "worst_group_undercoverage"]
                    - oracle.loc[horizon, "worst_group_undercoverage"]
                ),
                "remaining_local_to_oracle_gap": float(
                    local.loc[horizon, "worst_group_undercoverage"]
                    - oracle.loc[horizon, "worst_group_undercoverage"]
                ),
                "local_interval_score": float(local.loc[horizon, "mean_normalized_interval_score"]),
                "oracle_interval_score": float(oracle.loc[horizon, "mean_normalized_interval_score"]),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "oracle_gap_summary.csv", index=False)
    print("Advanced diagnostic tables written.")


if __name__ == "__main__":
    main()
