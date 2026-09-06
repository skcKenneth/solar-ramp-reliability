"""Verify frozen science and artifact integrity, without fitting any model."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from solar_reliability.reproducibility import model_execution_source_sha256
from analyze_results import ramp_threshold_station_table, validate_nominal_group_metrics


def verify_frozen_results(root: Path = ROOT) -> dict:
    fixture = json.loads((root / 'tests/fixtures/frozen_results.json').read_text())
    config_path = root / 'configs/experiment.yaml'
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != fixture['config_sha256']:
        raise AssertionError('Frozen configuration changed')
    if model_execution_source_sha256(root) != fixture['model_execution_source_sha256']:
        raise AssertionError('Model-execution source changed; prediction provenance requires re-evaluation')
    cfg = yaml.safe_load(config_path.read_text())
    tables = root / cfg['outputs']['tables']
    scalar_checks = 0
    for name, frozen in fixture['tables'].items():
        expected = pd.DataFrame(frozen['records'], columns=frozen['columns'])
        expected = pd.read_csv(io.StringIO(expected.to_csv(index=False)))
        actual = pd.read_csv(tables / f'{name}.csv')[frozen['columns']]
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)
        scalar_checks += expected.size
    verdict = json.loads((tables / 'multisite_verdict.json').read_text())
    verdict.pop('runtime_seconds', None)
    assert verdict == fixture['verdict'], 'Nominal verdict changed'
    groups = pd.read_csv(tables / 'ramp_threshold_group_metrics.csv')
    validate_nominal_group_metrics(groups, cfg, root=root)
    thresholds = pd.read_csv(tables / 'ramp_threshold_sensitivity.csv')
    expected_thresholds = ramp_threshold_station_table(groups, cfg, expected_stations=verdict['valid_confirmatory_stations'])
    # Compare the on-disk CSV representation (empty strings/nullable booleans
    # round-trip as NaN); retain missing gate decisions, never coerce to False.
    expected_thresholds = pd.read_csv(io.StringIO(expected_thresholds.to_csv(index=False)))
    pd.testing.assert_frame_equal(thresholds, expected_thresholds, check_dtype=False, rtol=1e-10, atol=1e-12)
    unavailable_support = ~thresholds['ramp_support_available']
    assert thresholds.loc[unavailable_support, 'candidate_absolute_gate_passed'].isna().all()
    assert thresholds.loc[unavailable_support, 'station_improvement'].isna().all()
    assert thresholds.loc[thresholds['is_nominal_threshold'], 'ramp_support_complete'].all()
    availability = pd.read_csv(tables / 'prediction_availability.csv')
    assert (availability['attempted_n'] == availability['usable_n'] + availability['unavailable_n']).all()
    np.testing.assert_allclose(availability['issued_coverage'], availability['covered_n'] / availability['usable_n'].replace(0, np.nan))
    np.testing.assert_allclose(availability['service_coverage'], availability['covered_n'] / availability['attempted_n'])
    primary = [cfg['methods']['primary_baseline'], *cfg['methods']['primary_deployable_candidates']]
    assert availability.loc[availability['method'].isin(primary), 'unavailable_n'].eq(0).all()
    population = pd.read_csv(tables / 'natural_missingness_population.csv')
    assert len(population) == 40
    assert (population['target_available_daylight_n'] == population['model_evaluable_n'] + population['excluded_current_power_missing_n']).all()
    clean = availability[availability['condition'].eq('clean') & availability['method'].eq(primary[0])]
    clean = clean.groupby(['station', 'fold', 'horizon_steps'])['attempted_n'].sum()
    aligned = population.set_index(['station', 'fold', 'horizon_steps'])['model_evaluable_n']
    pd.testing.assert_series_equal(aligned.sort_index(), clean.sort_index(), check_names=False)
    natural = pd.read_csv(tables / 'natural_missingness_performance.csv')
    assert set(natural['method']) == set(primary), 'Do not mix persistence stress tests into natural missingness'
    eligible = natural['performance_eligible']
    assert not natural.loc[~eligible, ['coverage', 'normalized_interval_score']].notna().any().any()
    assert natural.loc[eligible, 'attempted_n'].ge(cfg['calibration']['min_group_n']).all()
    np.testing.assert_allclose(natural['coverage'], natural['issued_coverage'], equal_nan=True)
    np.testing.assert_allclose(natural['coverage'], natural['service_coverage'], equal_nan=True)
    return {'status': 'PASS', 'reference_commit': fixture['source_commit'], 'frozen_tables': len(fixture['tables']),
            'frozen_table_cells': scalar_checks, 'nominal_groups_checked': 3200,
            'threshold_rows_checked': len(thresholds), 'insufficient_support_rows': int(unavailable_support.sum()),
            'population_rows': len(population), 'refitted_jobs': 0,
            'scope': 'frozen-result regression and denominator/population consistency; not full model reproduction'}


def verify_transport(root: Path = ROOT) -> int:
    manifest = json.loads((root / 'OUTPUT_MANIFEST.json').read_text())
    paths = sorted(p.relative_to(root).as_posix() for p in (root / 'outputs').rglob('*') if p.is_file())
    assert paths == [r['path'] for r in manifest['outputs']]
    assert len(paths) == manifest['output_count']
    for record in manifest['outputs']:
        data = (root / record['path']).read_bytes()
        assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
    for record in pd.read_csv(root / 'data/raw/multisite/download_receipt.csv').itertuples():
        path = root / str(record.path).replace('\\', '/')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record.sha256, path
    return len(paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    if args.config != 'configs/experiment.yaml':
        parser.error('This verifier targets the fixed released configuration only')
    result = verify_frozen_results()
    result['output_hashes_checked'] = verify_transport()
    print(json.dumps(result, indent=2))
