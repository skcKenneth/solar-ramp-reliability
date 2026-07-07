from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_reliability.conformal import adjust_interval, conformal_quantile, cqr_scores
from solar_reliability.data import load_solar_data
from solar_reliability.metrics import interval_metrics

FIXTURE_CSV = Path(__file__).resolve().parent / "data" / "minimal_site.csv"


def test_load_solar_data_from_csv() -> None:
    loaded = load_solar_data(FIXTURE_CSV, nominal_capacity_mw=50.0)

    assert len(loaded) == 4
    assert list(loaded.columns) == ["tsi", "dni", "ghi", "temperature", "pressure", "humidity", "power"]
    assert loaded.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all()


def test_interval_metrics_perfect_coverage() -> None:
    y = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([2.0, 3.0, 4.0])

    metrics = interval_metrics(y, lower, upper, alpha=0.10)

    assert metrics["coverage"] == 1.0
    assert metrics["coverage_error"] == pytest.approx(0.1)
    assert metrics["mean_width"] == 2.0


def test_cqr_scores_and_adjust_interval() -> None:
    y = np.array([1.0, 3.0])
    lower = np.array([0.0, 1.0])
    upper = np.array([2.0, 4.0])

    scores = cqr_scores(y, lower, upper)
    qhat = conformal_quantile(scores, alpha=0.10)
    adj_lo, adj_hi = adjust_interval(lower, upper, qhat, capacity=10.0)

    assert scores.shape == y.shape
    assert np.all(adj_lo <= adj_hi)
    assert np.all(adj_lo >= 0.0)
    assert np.all(adj_hi <= 10.0)
