"""No-fit descriptions and population accounting; never changes model inputs."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from solar_reliability.data import load_solar_data
from solar_reliability.features import make_supervised
from solar_reliability.splits import leakage_safe_fold_masks


def population_counts(raw: pd.DataFrame, cfg: dict, fold: dict, horizon: int) -> tuple[dict, pd.DatetimeIndex]:
    """Account for exclusions before missing-feature performance is measurable."""
    target_time = raw.index + pd.Timedelta(minutes=int(cfg['data']['sampling_minutes']) * horizon)
    masks = leakage_safe_fold_masks(raw.index, target_time, fold, horizon_steps=horizon,
                                   sampling_minutes=int(cfg['data']['sampling_minutes']))
    daylight = pd.Series(raw.index.hour, index=raw.index).between(*cfg['data']['daylight_hours'])
    scheduled = masks.test & daylight
    target_available = scheduled & raw['power'].shift(-horizon).notna()
    evaluable = target_available & raw['power'].notna()
    origins = pd.DatetimeIndex(raw.index[evaluable])
    counts = {
        'purged_test_daylight_origins_n': int(scheduled.sum()),
        'future_target_missing_n': int((scheduled & ~target_available).sum()),
        'target_available_daylight_n': int(target_available.sum()),
        'excluded_current_power_missing_n': int((target_available & ~evaluable).sum()),
        'model_evaluable_n': int(evaluable.sum()),
        'model_evaluable_origin_sha256': hashlib.sha256(origins.asi8.astype('<i8').tobytes()).hexdigest(),
    }
    return counts, origins


def build_population_audit(root: Path, cfg: dict) -> pd.DataFrame:
    audit = pd.read_csv(root / cfg['data']['station_audit_table'])
    stations = set(audit.loc[audit['status'].eq('valid'), 'site_id']) - set(cfg['project']['development_sites'])
    manifest = pd.read_csv(root / cfg['data']['manifest'])
    records = []
    for record in manifest.loc[manifest['site_id'].isin(stations)].itertuples():
        raw_path = root / cfg['data']['raw_dir'] / record.filename
        raw = load_solar_data(raw_path, nominal_capacity_mw=record.nominal_capacity_mw)
        for h in cfg['forecast']['horizons_steps']:
            x, _, meta = make_supervised(raw, h, lag_steps=cfg['forecast']['lag_steps'],
                                        rolling_windows=cfg['forecast']['rolling_windows'],
                                        daylight_hours=tuple(cfg['data']['daylight_hours']),
                                        nominal_capacity_mw=record.nominal_capacity_mw)
            for fold in cfg['data']['site1_calendar_folds']:
                counts, origins = population_counts(raw, cfg, fold, h)
                mask = leakage_safe_fold_masks(x.index, meta['target_timestamp'], fold,
                                               horizon_steps=h, sampling_minutes=cfg['data']['sampling_minutes'])
                if not origins.equals(pd.DatetimeIndex(x.index[mask.test])):
                    raise RuntimeError('Population audit disagrees with unchanged supervised test selection')
                records.append({'station': record.site_id, 'fold': fold['name'], 'horizon_steps': h,
                                **counts, 'raw_file_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                                'scope': 'post_run_population_accounting_no_predictions_for_excluded_origins',
                                'target_interpolated': False})
    return pd.DataFrame(records).sort_values(['station', 'fold', 'horizon_steps']).reset_index(drop=True)


def expanded_method_details(cfg: dict) -> dict:
    """Explicit account of models.py/run_advanced_study.py, guarded by regression tests."""
    p = cfg['model']
    return {
        'shared_forecaster_scope': {
            'shared': [cfg['methods']['primary_baseline'], *cfg['methods']['primary_deployable_candidates'],
                       'oracle_ramp_mondrian'],
            'not_shared': ['persistence_condition_cqr', 'lgbm_condition_cqr', 'clean_condition_cqr'],
            'fit_unit': 'one degradation-augmented ExtraTreesRegressor per station/fold/horizon',
            'fit_cutoff': 'purged training block only; no calibration or test fitting of base/risk model',
        },
        'base_interval': {
            'construction': 'numpy.quantile of the individual tree point predictions, not leaf-response quantiles',
            'quantiles': [float(cfg['project']['alpha']) / 2, .5, 1 - float(cfg['project']['alpha']) / 2],
            'numpy_quantile_method': 'linear (locked NumPy default)',
            'imputer': 'training-fitted median; add_indicator=True; keep_empty_features=True',
            'indicator_scope': 'columns missing during imputer fitting; explicit family fractions also recomputed',
            'n_estimators': p['n_estimators'], 'max_depth': p['max_depth'],
            'min_samples_leaf': p['min_samples_leaf'], 'max_features': p['max_features'],
            'bootstrap': True, 'max_samples': p['max_samples'], 'max_train_rows': p['max_train_rows'],
            'subsampling': 'deterministic evenly spaced row indices using numpy.linspace, not random sampling',
            'augmentation_fraction_total_across_degraded_conditions': p['augmented_condition_fraction'],
            'augmentation_partition': 'one seeded permutation; selected rows partitioned across the four degraded conditions',
            'ramp_weight': p['ramp_sample_weight'], 'degraded_weight_multiplier': p['degraded_sample_weight'],
            'clipping': 'base endpoints before scores and expanded endpoints to [0, capacity]',
        },
        'risk_model': {
            'type': 'ExtraTreesClassifier', 'training': 'unaugmented subsampled purged training features',
            'label': 'abs(y[t+h]-y[t])/capacity >= nominal threshold for the horizon',
            'imputer': 'separate training-fitted median; add_indicator=True; keep_empty_features=True',
            'n_estimators': p['risk_n_estimators'], 'max_depth': p['risk_max_depth'],
            'min_samples_leaf': p['risk_min_samples_leaf'], 'max_features': p['max_features'],
            'bootstrap': True, 'max_samples': p['max_samples'], 'class_weight': 'balanced',
            'probability': 'class-1 probability from predict_proba; both classes verified in all 40 saved runs',
            'seed': 'job_seed_map(project seed, station, fold, horizon)[ramp_risk_model]',
            'strata': 'four quantile bins of pooled calibration risks across all five conditions',
            'ties': 'numpy.digitize(..., right=True); repeated edges may leave empty bins',
            'shrinkage_k': 100.0, 'empty_bin_fallback': 'condition-specific quantile (weight zero)',
        },
        'sensitivity_scope': {
            'ramp': 'post-run relabeling of saved nominal-threshold predictions using exact raw power differences',
            'refitting': False, 'recalibration': False,
            'unchanged': 'ramp training weights, risk labels/model, calibrators, future-label reference, thresholds/gates',
            'limitation': 'not a comparison of pipelines retrained at alternative ramp definitions',
        },
        'coverage_denominators': {
            'attempted': 'target-evaluable saved prediction rows, including unavailable prediction tuples',
            'issued': 'rows with finite median, lower and upper',
            'issued_coverage': 'covered / issued; undefined when issued=0',
            'service_coverage': 'covered / attempted; unavailable attempts count as uncovered',
            'width_and_interval_score': 'issued predictions only',
            'primary': 'requires complete availability for baseline and three candidates',
            'natural_missingness': 'four primary methods; naturally missing features in clean-condition predictions only',
            'exclusions': 'missing current power or future target not included; no evidence on those origins',
        },
    }


def write_reporting_support(root: Path, cfg: dict, output: Path) -> None:
    method_path = output / 'method_specification.json'
    method = json.loads(method_path.read_text(encoding='utf-8'))
    method['reproduction_details'] = expanded_method_details(cfg)
    method_path.write_text(json.dumps(method, indent=2) + '\n', encoding='utf-8', newline='\n')
    build_population_audit(root, cfg).to_csv(output / 'natural_missingness_population.csv', index=False, lineterminator='\n')
