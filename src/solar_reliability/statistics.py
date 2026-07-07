from __future__ import annotations

import numpy as np
import pandas as pd


def day_cluster_bootstrap_difference(
    predictions: pd.DataFrame,
    method_a: str,
    method_b: str,
    alpha: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    results: list[dict] = []
    base = predictions[predictions["method"].isin([method_a, method_b])].copy()
    for (horizon, condition, ramp_group), group in base.groupby(
        ["horizon_steps", "condition", "ramp_group"], sort=True
    ):
        daily = (
            group.groupby(["date", "method"])["covered"]
            .agg(["sum", "count"])
            .reset_index()
        )
        a = daily[daily["method"] == method_a].set_index("date")
        b = daily[daily["method"] == method_b].set_index("date")
        dates = a.index.intersection(b.index)
        if len(dates) < 10:
            continue
        a = a.loc[dates]
        b = b.loc[dates]
        index = rng.integers(0, len(dates), size=(replicates, len(dates)))
        a_coverage = a["sum"].to_numpy()[index].sum(axis=1) / a["count"].to_numpy()[index].sum(axis=1)
        b_coverage = b["sum"].to_numpy()[index].sum(axis=1) / b["count"].to_numpy()[index].sum(axis=1)
        diffs = np.abs(a_coverage - (1 - alpha)) - np.abs(b_coverage - (1 - alpha))
        results.append(
            {
                "horizon_steps": int(horizon),
                "condition": condition,
                "ramp_group": ramp_group,
                "method_a": method_a,
                "method_b": method_b,
                "difference_a_minus_b": float(np.mean(diffs)),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "days": int(len(dates)),
            }
        )
    return pd.DataFrame(results)
