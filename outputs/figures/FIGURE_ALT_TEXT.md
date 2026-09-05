# Figure alt text and evidence mapping

This deterministic accessibility artifact covers exactly the three corrected core figures. Alt-text values are rounded to three decimals for readability; plotting and validation use the full-precision source values. In the companion GitHub code archive, repository-relative source paths and exact input hashes are recorded in `outputs/tables/visualization_audit.json`.

## Figure 1 - Prespecified confirmation protocol and pre-outcome station screening

**Alt text.** Five protocol stages run from development on CSGS1 only, through prespecified splits, sensor conditions, and seeds, to screening 7 external station records and confirming 5 valid stations over 4 folds and 2 horizons (40 station-fold-horizon evaluation units), followed by prespecified station-level criteria. The included stations and nominal capacities are CSGS2 (130 MW), CSGS5 (110 MW), CSGS6 (35 MW), CSGS7 (30 MW), CSGS8 (30 MW). The excluded records are CSGS3 (30 MW; missing required temperature field); CSGS4 (130 MW; daylight target missingness exceeded the prespecified 20% threshold). Station codes are dataset identifiers, not geographic locations.

**Source and claim mapping.** Protocol settings come from `configs/experiment.yaml` (`project.development_sites`, `project.seed`, `data.site1_calendar_folds`, `conditions`, `forecast.horizons_steps`, `metrics.statistical_replication_units.primary`, and `decision_rules`). Screening marks, capacities, statuses, and recorded reasons come directly from `outputs/tables/station_data_audit.csv` (`site_id`, `nominal_capacity_mw`, `status`, `exclusion_reason`). The included-station set and total of 40 evaluations are cross-checked against `outputs/tables/multisite_verdict.json`.

| Station | Plotted capacity (MW) | Plotted status | Displayed exclusion criterion |
| --- | --- | --- | --- |
| CSGS2 | 130 | included | - |
| CSGS3 | 30 | excluded | missing required temperature field |
| CSGS4 | 130 | excluded | daylight target missingness exceeded the prespecified 20% threshold |
| CSGS5 | 110 | included | - |
| CSGS6 | 35 | included | - |
| CSGS7 | 30 | included | - |
| CSGS8 | 30 | included | - |

## Figure 2 - Station-level worst-group undercoverage

**Alt text.** Two panels compare condition CQR (open circles) with Descriptor CQR (filled diamonds) at 15- and 60-minute horizons for five confirmatory stations. Arrows run from condition CQR to the Descriptor CQR value, which is lower in 10 of 10 station-horizon comparisons. 15 minutes: Descriptor CQR 0.244 to 0.430 versus condition CQR 0.354 to 0.549; 60 minutes: Descriptor CQR 0.107 to 0.363 versus condition CQR 0.187 to 0.415. A dashed line marks the prespecified undercoverage tolerance of 0.10. No Descriptor CQR estimate satisfies it.

**Source and claim mapping.** Every circle, diamond, and arrow endpoint comes from `outputs/tables/station_level_effects.csv`, filtered to `method=robust_local_cqr`, using `station`, `horizon_steps`, `baseline_worst_undercoverage`, and `method_worst_undercoverage`. Horizon labels use `data.sampling_minutes` and `forecast.horizons_steps`; the dashed tolerance uses `decision_rules.absolute_reliability_gate.max_worst_group_undercoverage_each_horizon` from `configs/experiment.yaml`.

| Horizon | Station | Condition CQR | Descriptor CQR |
| --- | --- | --- | --- |
| 15 min | CSGS2 | 0.549 | 0.430 |
| 15 min | CSGS5 | 0.382 | 0.244 |
| 15 min | CSGS6 | 0.411 | 0.314 |
| 15 min | CSGS7 | 0.354 | 0.246 |
| 15 min | CSGS8 | 0.496 | 0.407 |
| 60 min | CSGS2 | 0.283 | 0.162 |
| 60 min | CSGS5 | 0.225 | 0.175 |
| 60 min | CSGS6 | 0.259 | 0.161 |
| 60 min | CSGS7 | 0.187 | 0.107 |
| 60 min | CSGS8 | 0.415 | 0.363 |

## Figure 3 - Station-first width–score trade-off

**Alt text.** Two panels plot change in normalized interval width horizontally and change in normalized interval score vertically, relative to condition CQR at 15 and 60 minutes. The two condition-CQR baselines are at the origin; their absolute values are 15 minutes: normalized width 0.203 and interval score 0.454; 60 minutes: normalized width 0.242 and interval score 0.429. All six candidate estimates have positive width differences and negative interval-score differences relative to condition CQR. The plotted candidate coordinates are 15-minute Descriptor CQR: width +0.028, score -0.049; 15-minute Risk-Mondrian CQR: width +0.019, score -0.035; 15-minute Rolling CQR: width +0.003, score -0.006; 60-minute Descriptor CQR: width +0.028, score -0.027; 60-minute Risk-Mondrian CQR: width +0.017, score -0.014; 60-minute Rolling CQR: width +0.002, score -0.005. Every displayed method and horizon aggregates five stations and has 100% mean station-level prediction availability.

**Source and claim mapping.** Every point, baseline reference value, and availability claim comes from `outputs/tables/method_tradeoffs.csv`, filtered to `stratum=all` and the four displayed methods. The mapped fields are `horizon_steps`, `method`, `stations`, `aggregation_unit`, `mean_normalized_width`, `mean_normalized_interval_score`, `normalized_width_change_vs_baseline`, `normalized_interval_score_change_vs_baseline`, and `mean_station_prediction_availability`. Each delta is recomputed from its method and condition-CQR source means before rendering.

| Horizon | Method | Mean normalized width | Mean normalized interval score | Width change | Interval-score change | Availability |
| --- | --- | --- | --- | --- | --- | --- |
| 15 min | Condition CQR | 0.203 | 0.454 | +0.000 | +0.000 | 100% |
| 15 min | Descriptor CQR | 0.232 | 0.405 | +0.028 | -0.049 | 100% |
| 15 min | Risk-Mondrian CQR | 0.222 | 0.420 | +0.019 | -0.035 | 100% |
| 15 min | Rolling CQR | 0.207 | 0.448 | +0.003 | -0.006 | 100% |
| 60 min | Condition CQR | 0.242 | 0.429 | +0.000 | +0.000 | 100% |
| 60 min | Descriptor CQR | 0.270 | 0.402 | +0.028 | -0.027 | 100% |
| 60 min | Risk-Mondrian CQR | 0.259 | 0.416 | +0.017 | -0.014 | 100% |
| 60 min | Rolling CQR | 0.244 | 0.424 | +0.002 | -0.005 | 100% |

## Validation criteria

Validation fails if the frozen external-station list does not match the audit; any capacity, status, or exclusion reason is missing; the audit and verdict station sets differ; the station-by-horizon or method-by-horizon grids are incomplete; a plotted metric is non-finite; an undercoverage value lies outside [0, 1]; a tradeoff row is not a five-station station-first aggregate with complete availability; a condition-CQR delta is nonzero; or a reported tradeoff delta differs from the corresponding difference of full-precision source means. A violation prevents figure generation.
