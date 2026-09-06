from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from solar_reliability.advanced import (augment_training_set, chronological_fit_adjust_masks,
    descriptor_frame, fit_risk_mondrian, rolling_qhat, DESCRIPTOR_CONTEXT_COLUMNS)
from solar_reliability.conformal import cqr_scores, conformal_quantile
from solar_reliability.data import SENSOR_COLUMNS
from solar_reliability.features import apply_condition, condition_mask_columns, make_supervised
from solar_reliability.models import QuantileModels, fit_ramp_risk_model
from solar_reliability.reproducibility import job_seed_map
from solar_reliability.splits import leakage_safe_fold_masks
from analyze_results import (ramp_threshold_station_table, natural_missingness_chunk_performance,
    natural_missingness_summary, station_first_tradeoffs, gate_sensitivity_table)
from prediction_metrics import prepare_prediction_metric_frame, aggregate_prediction_metrics, require_full_availability
from reporting_support import population_counts, expanded_method_details
from verify_results import verify_frozen_results

CFG = yaml.safe_load((ROOT / 'configs/experiment.yaml').read_text())


def synthetic_raw():
    index = pd.date_range('2020-01-01', periods=96 * 6, freq='15min')
    raw = pd.DataFrame({s: np.ones(len(index)) for s in SENSOR_COLUMNS}, index=index)
    raw['power'] = np.arange(len(index)) % 9 + 1.0
    return raw


FOLD = dict(name='toy', train_end='2020-01-02 23:45', calibration_start='2020-01-03',
            calibration_end='2020-01-04 23:45', test_start='2020-01-05', test_end='2020-01-06 23:45')


@pytest.mark.parametrize('h', [1, 4])
def test_target_time_purging_and_population_exclusions(h):
    raw = synthetic_raw()
    raw.loc['2020-01-05 12:00', 'power'] = np.nan
    raw.loc['2020-01-05 14:00', 'tsi'] = np.nan
    x, y, meta = make_supervised(raw, h, nominal_capacity_mw=20)
    masks = leakage_safe_fold_masks(x.index, meta['target_timestamp'], FOLD, horizon_steps=h)
    for name, end in [('train', 'train_end'), ('calibration', 'calibration_end'), ('test', 'test_end')]:
        assert (meta.loc[getattr(masks, name), 'target_timestamp'] <= pd.Timestamp(FOLD[end])).all()
    assert not (masks.train & masks.calibration | masks.calibration & masks.test).any()
    counts, origins = population_counts(raw, CFG, FOLD, h)
    assert counts['excluded_current_power_missing_n'] == 1
    assert counts['future_target_missing_n'] == 1
    assert counts['target_available_daylight_n'] == counts['model_evaluable_n'] + 1
    assert origins.equals(pd.DatetimeIndex(x.index[masks.test]))
    assert pd.Timestamp('2020-01-05 12:00') not in origins
    assert pd.Timestamp('2020-01-05 14:00') in origins  # Missing sensor is not a target exclusion.
    assert y.notna().all()


def test_job_seeds_do_not_depend_on_loop_order():
    jobs = [('CSGS2', 'q1_2020', 1), ('CSGS8', 'q4_2020', 4)]
    first = {j: job_seed_map(CFG['project']['seed'], *j) for j in jobs}
    second = {j: job_seed_map(CFG['project']['seed'], *j) for j in reversed(jobs)}
    assert first == second
    assert len(set(first[jobs[0]].values())) == 6


def test_exact_masks_and_target_preservation():
    x, y, meta = make_supervised(synthetic_raw(), 1, nominal_capacity_mw=20)
    original_y, original_meta = y.copy(), meta.copy()
    recent = condition_mask_columns(x.columns, 'recent_power_outage')
    assert 'power_lag_12' in recent  # Intentional suffix rule, not just the four named lags.
    assert 'power_lag_8' not in recent and 'power_mean_24' not in recent
    for condition in CFG['conditions']:
        out = apply_condition(x, condition)
        mask = condition_mask_columns(x.columns, condition)
        assert out[mask].isna().all().all()
        untouched = [c for c in x if c not in mask and 'fraction' not in c]
        pd.testing.assert_frame_equal(out[untouched], x[untouched])
    pd.testing.assert_series_equal(y, original_y)
    pd.testing.assert_frame_equal(meta, original_meta)


def test_augmentation_fraction_is_total_not_per_condition():
    x, y, _ = make_supervised(synthetic_raw(), 1, nominal_capacity_mw=20)
    labels = np.arange(len(x)) % 2 == 0
    xa, ya, weights, conditions = augment_training_set(x, y, labels, CFG['conditions'], .65, 2.5, .75, 7)
    degraded = conditions.ne('clean')
    assert int(degraded.sum()) == round(len(x) * .65)
    assert conditions.index[degraded].is_unique
    np.testing.assert_array_equal(ya.iloc[:len(x)], y)
    np.testing.assert_allclose(weights[:len(x)], np.where(labels, 2.5, 1))
    assert len(xa) == len(x) + round(len(x) * .65)


def test_descriptor_schema_and_chronological_timestamp_groups():
    x, _, _ = make_supervised(synthetic_raw(), 1, nominal_capacity_mw=20)
    n = len(x)
    d = descriptor_frame(x, 'clean', np.zeros(n), np.zeros(n), np.ones(n), np.full(n, 2), 20, CFG['conditions'])
    assert len(d.columns) == 15
    assert set(d) == {'risk', 'native_width', 'median_level', *DESCRIPTOR_CONTEXT_COLUMNS,
                      *('condition__' + c for c in CFG['conditions'])}
    assert not any('target' in c or 'ramp_group' in c for c in d)
    times = pd.Series(np.repeat(pd.date_range('2020-01-01', periods=10), 5)).sample(frac=1, random_state=1)
    fit, adjust = chronological_fit_adjust_masks(times, .60)
    assert fit.sum() == 30 and adjust.sum() == 20
    assert times.iloc[np.flatnonzero(fit)].max() < times.iloc[np.flatnonzero(adjust)].min()


def test_empirical_tree_prediction_quantiles_not_leaf_response_quantiles():
    class Tree:
        def __init__(self, value): self.value = value
        def predict(self, x): return np.full(len(x), self.value)
    imputer = SimpleNamespace(transform=lambda x: x)
    model = QuantileModels(imputer, SimpleNamespace(estimators_=[Tree(v) for v in [0, 10, 20]]), .10)
    lo, med, hi = model.predict(pd.DataFrame({'x': [1, 2]}))
    np.testing.assert_allclose(lo, 1)
    np.testing.assert_allclose(med, 10)
    np.testing.assert_allclose(hi, 19)


def test_risk_model_parameters_and_empty_bin_fallback():
    x = pd.DataFrame({'x': np.arange(200, dtype=float), 'missing': [np.nan, 1.] * 100})
    model = fit_ramp_risk_model(x, np.arange(200) % 2, CFG['model'], 7)
    spec = expanded_method_details(CFG)['risk_model']
    params = model.classifier.get_params()
    for key in ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'bootstrap', 'max_samples', 'class_weight']:
        assert params[key] == spec[key]
    assert model.imputer.strategy == 'median' and model.imputer.add_indicator
    assert np.isfinite(model.predict_proba(x)).all()
    scores = pd.DataFrame({'condition': ['clean'] * 100, 'risk': np.zeros(100), 'score': np.arange(100.)})
    table = fit_risk_mondrian(scores, .1, 4)
    np.testing.assert_allclose(table.qhat('clean', np.ones(3)), table.condition_q['clean'])


def test_manual_interval_score_and_issued_service_denominators():
    predictions = pd.DataFrame({'method': ['m'] * 3, 'y': [1., 4., 2.],
                                'median': [1., 1., np.nan], 'lower': [0., 0., np.nan], 'upper': [2., 2., np.nan]})
    prepared = prepare_prediction_metric_frame(predictions, alpha=.1)
    row = aggregate_prediction_metrics(prepared, group_columns=['method'], alpha=.1, capacity=10).iloc[0]
    assert row.attempted_n == 3 and row.usable_n == 2 and row.unavailable_n == 1
    assert row.issued_coverage == .5 and row.service_coverage == pytest.approx(1 / 3)
    assert row.interval_score == 22  # (2 + (2 + 20*2))/2
    with pytest.raises(RuntimeError, match='unavailable'):
        require_full_availability(prepared, ['m'], context='toy')
    absent = prepared.iloc[[2]]
    a = aggregate_prediction_metrics(absent, group_columns=['method'], alpha=.1, capacity=10).iloc[0]
    assert np.isnan(a.issued_coverage) and a.service_coverage == 0
    assert np.isnan(a.interval_score)
    np.testing.assert_allclose(cqr_scores(np.array([1., 4.]), np.zeros(2), np.full(2, 2.)), [0, 2])


@pytest.mark.parametrize('corruption', ['partial', 'infinite', 'reversed'])
def test_invalid_prediction_evidence_fails_closed(corruption):
    frame = pd.DataFrame({'y': [1.], 'median': [1.], 'lower': [0.], 'upper': [2.]})
    if corruption == 'partial': frame.loc[0, 'median'] = np.nan
    if corruption == 'infinite': frame.loc[0, 'upper'] = np.inf
    if corruption == 'reversed': frame.loc[0, 'lower'] = 3
    with pytest.raises(ValueError): prepare_prediction_metric_frame(frame, alpha=.1)


def threshold_fixture(ramp_n):
    cfg = deepcopy(CFG)
    cfg['methods']['primary_deployable_candidates'] = ['robust_local_cqr']
    rows = []
    for method in [cfg['methods']['primary_baseline'], 'robust_local_cqr']:
        for group, n in [('ramp', ramp_n), ('non_ramp', 1000)]:
            rows.append(dict(station='CSGS2', horizon_steps=1, threshold=.15, method=method,
                             condition='clean', fold='q1_2020', ramp_group=group, n=n, undercoverage_error=0.0))
    return cfg, pd.DataFrame(rows)


@pytest.mark.parametrize('n,assessable', [(0, False), (79, False), (80, True)])
def test_insufficient_ramp_support_is_not_a_gate_pass(n, assessable):
    cfg, groups = threshold_fixture(n)
    row = ramp_threshold_station_table(groups, cfg, expected_stations=['CSGS2']).iloc[0]
    assert bool(row.ramp_support_available) == assessable
    assert row.candidate_eligible_non_ramp_groups == 1
    if not assessable:
        assert pd.isna(row.candidate_absolute_gate_passed) and pd.isna(row.station_improvement)
        assert row.candidate_absolute_gate_status == 'INSUFFICIENT_RAMP_SUPPORT'
        assert not row.candidate_all_stations_absolute_gate_passed
    else:
        assert row.candidate_absolute_gate_status == 'PASS'
        assert row.ramp_support_status == 'PARTIAL_RAMP_SUPPORT'


def test_missing_station_cannot_satisfy_all_station_gate():
    cfg, groups = threshold_fixture(80)
    row = ramp_threshold_station_table(groups, cfg, expected_stations=['CSGS2', 'CSGS8']).iloc[0]
    assert not row.candidate_all_stations_absolute_gate_passed
    assert row.candidate_all_stations_gate_status == 'INCOMPLETE_STATION_SET'


def natural_fixture(n):
    times = pd.date_range('2020-01-01 05:00', periods=n, freq='15min')
    x = pd.DataFrame({'power_current': np.ones(n), 'power_lag_1': np.full(n, np.nan)}, index=times)
    y = pd.Series(np.ones(n), index=times)
    meta = pd.DataFrame({'target_timestamp': times + pd.Timedelta(minutes=15), 'ramp_fraction': .2}, index=times)
    audit = pd.DataFrame([dict(group=g, source_columns='|'.join(condition_mask_columns(x, g + '_outage')),
                               structurally_absent_optional_channels='', missing_n_among_model_evaluable=n,
                               confirmatory_performance_eligible=True, eligibility_reason='toy')
                          for g in ['irradiance', 'weather', 'recent_power', 'combined']])
    pred = pd.DataFrame(dict(fold='toy', timestamp=times, target_timestamp=meta['target_timestamp'].to_numpy(),
                            horizon_steps=1, horizon_minutes=15, condition='clean', method='base', ramp_group='ramp',
                            y=1., median=1., lower=0., upper=2.))
    kwargs = dict(expected_test_origins=times, station='toy', fold='toy', horizon=1, horizon_minutes=15,
                  capacity=10, alpha=.1, minimum_group_n=80, methods=['base'], baseline_method='base',
                  nominal_ramp_threshold=.1, raw_file='toy.csv', raw_file_sha256='synthetic')
    return (pred, x, y, meta, audit), kwargs


@pytest.mark.parametrize('n', [79, 80])
def test_natural_missingness_eligibility_and_duplicate_family_pooling(n):
    args, kwargs = natural_fixture(n)
    result = natural_missingness_chunk_performance(*args, **kwargs)
    eligible = result[result.performance_eligible]
    if n < 80:
        assert eligible.empty and result.coverage.isna().all()
    else:
        assert len(eligible) == 2  # Same recent-power and combined timestamps.
        summary = natural_missingness_summary(result)
        assert summary.iloc[0].prediction_rows == 80  # Never double count to 160.
        assert summary.iloc[0].eligible_unique_cells == 1
        assert eligible.issued_coverage.eq(1).all() and eligible.service_coverage.eq(1).all()


def test_natural_missingness_rejects_missing_prediction_origin():
    args, kwargs = natural_fixture(80)
    with pytest.raises(RuntimeError, match='exact corrected test origin'):
        natural_missingness_chunk_performance(args[0].iloc[:-1], *args[1:], **kwargs)


def test_rolling_scores_are_released_only_at_target_time():
    origins = pd.date_range('2020-01-01', periods=4, freq='15min')
    targets = origins + pd.Timedelta(hours=1)
    base = rolling_qhat(np.zeros(100), np.full(4, 999.), origins, targets, .1, 100, 100)
    np.testing.assert_array_equal(base, 0)


def test_station_first_not_timestamp_pooled_and_gate_monotonicity():
    groups = pd.read_csv(ROOT / 'outputs/tables/ramp_threshold_group_metrics.csv')
    method = CFG['methods']['primary_baseline']
    result = station_first_tradeoffs(groups, CFG)
    nominal = groups[(groups.horizon_steps == 1) & (groups.threshold == .1) & (groups.method == method) & (groups.n >= 80)]
    expected = nominal.groupby('station')['coverage'].mean().mean()
    actual = result[(result.horizon_steps == 1) & (result.method == method) & (result.stratum == 'all')].iloc[0]
    assert actual.mean_station_coverage == pytest.approx(expected)
    assert abs(expected - np.average(nominal.coverage, weights=nominal.n)) > .01
    gate = gate_sensitivity_table(groups, CFG)
    for _, g in gate.groupby(['horizon_steps', 'method']):
        assert (np.diff(g.sort_values('delta').stations_passing) >= 0).all()


def test_frozen_main_numbers_and_nominal_decisions():
    assert verify_frozen_results()['status'] == 'PASS'
