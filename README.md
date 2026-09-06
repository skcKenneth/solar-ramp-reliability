# Solar ramp reliability

This repository contains the runnable scientific workflow, its eight source data
files, and the corresponding result tables and figures. The compact release does
not include a paper or document-production files.

## Requirements

- Python 3.11 or newer
- Windows, macOS, or Linux

Create an isolated environment and install the locked dependencies. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `.venv/bin/python` instead.

## Verify the package

```powershell
python -m pytest -q -p no:cacheprovider
python main.py verify
python main.py audit
python main.py smoke
python main.py figures
```

The smoke command checks every cell of the fixed 5-station x 4-fold x 2-horizon
matrix without fitting the models. Unit tests and smoke checks are **not** a full
scientific reproduction. `verify` compares six numerical tables and nominal
decisions with an independent pre-revision reference, checks the 3,200 nominal
groups, coverage denominators, population accounting, and artifact hashes.

## Rebuild all results

```powershell
python main.py run
```

This command audits the data, resumes compatible completed jobs (fitting missing
jobs up to all 40), checks their 120 intermediate
artifacts, aggregates the results, generates the derived CSV/JSON files, and
renders the figures. Temporary model chunks and completion markers are written
under `work/` and are intentionally excluded from the repository ZIP.
To explicitly refit every job, use `python scripts/run_all.py --force`. This is
not needed to inspect or verify the included outputs. Do not alter the frozen
configuration to make a decision gate pass.

If compatible full prediction chunks and their original completion markers are
already present, the no-refit downstream sequence is:

```powershell
python scripts/generate_specifications.py
python main.py analyze
python main.py figures
python main.py verify
```

`analyze` fails if source/configuration/data/runtime hashes do not match the
original full-run markers. Never edit markers to bypass a mismatch. The compact
ZIP intentionally does not include these temporary chunks; on a new machine,
regenerating them requires the full fitting workflow above. All direct scripts
that update outputs should be followed by `python scripts/update_output_manifest.py`
before verification. This canonicalizes text outputs to LF for Git transport;
checksums are not evidence of scientific correctness by themselves.

Included result artifacts are under `outputs/tables/` and `outputs/figures/`.
`OUTPUT_MANIFEST.json` records their byte sizes and SHA-256 digests.

## Interpretation and reproducibility

- Nominal settings remain fixed in `configs/experiment.yaml`. The principal
  baseline and three calibration candidates share the degradation-augmented
  Extra-Trees forecaster, as does the future-label diagnostic. Persistence,
  LightGBM and clean-only Extra-Trees are separate auxiliary baselines.
- `outputs/tables/method_specification.json` specifies empirical tree-prediction
  quantiles, imputation, risk classifier, augmentation, descriptors and calibration.
  The 0.65 augmentation fraction is total across the four degraded conditions.
- Ramp-threshold sensitivity **relabels saved nominal predictions without refitting
  or recalibration**. `INSUFFICIENT_RAMP_SUPPORT` means that no ramp cell reaches
  the unchanged size of 80; it is not a pass or an estimated zero undercoverage.
  `PARTIAL_RAMP_SUPPORT` describes only the eligible subset of fold-condition cells.
- Issued coverage divides covered targets by issued predictions; service coverage
  divides by all attempted rows with evaluable targets. An all-missing prediction
  tuple is unavailable. Width and interval score use issued predictions only.
- Natural-missingness performance uses clean-condition predictions of the four
  primary methods, restricted to daylight test origins with both current and future
  power observed. `natural_missingness_population.csv` reports excluded origins.
  It does not evaluate complete current-power outages. Structurally absent optional
  channels are excluded from the natural masks; identical family selections are
  deduplicated before descriptive pooling.
- The fixed 0.10 absolute-undercoverage tolerance is a permissive empirical audit
  floor (80% coverage against a 90% target), not a theorem or universal service
  standard. Threshold and gate sensitivities do not establish operational adequacy.

## Data and result provenance

The eight bundled CSVs retain source URLs pinned to upstream revision
`58d5a90fe91b27f89a915e19ad519c0b2ab32e1c` in `data/multisite_manifest.csv`;
`data/raw/multisite/download_receipt.csv` records their SHA-256 hashes. The source
dataset is described by Chen and Xu (2022),
[Scientific Data 9, 577](https://doi.org/10.1038/s41597-022-01696-6).
The receipt says `already_present`; it does not establish an original download
date. The code license does not grant additional rights in third-party data.

`RESULT_PROVENANCE.json` records the prior fitted artifact hashes, seeds, software
versions and the unchanged model/configuration hashes checked for this revision.
The revised analysis reused those 40 fitted jobs; it did not retrain them.
The configuration's `frozen_at` field is retained; these execution receipts are
not an independent timestamped preregistration record. The requirements file pins
the direct scientific dependencies, not every transitive dependency; the delivery
verification logs record the newly resolved environment.
`tests/fixtures/frozen_results.json` was extracted from the pre-revision commit
`532dd54eabbf17bd0f0b7af1d08e2ae6ffef2596`, not from the revised outputs.
Five qualifying stations from one dataset, dependent rolling folds, sparse natural
missingness, and structured feature ablations remain limitations.
