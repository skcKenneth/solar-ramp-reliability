# Solar Ramp Reliability: Implementation Package

This archive contains the Python implementation, frozen configurations, tests,
data manifests, and compact generated result tables for the solar-ramp reliability study,
“When Solar Power Changes Fast: Conditional Reliability of Prediction Intervals under Ramp Events and Sensor Degradation.”

It intentionally excludes raw station data files, large prediction caches,
personal paths, Git metadata, and local runtime caches.

## Minimal Verification Without Large Data/Caches

```powershell
python -m pytest -q
python scripts\verify_literature_metadata.py
```

If dependencies are not already available, create an environment with the
packages in `requirements-lock.txt` before running the minimal verification.

## Full Reproduction Path

This compact implementation package excludes raw public station files and large
prediction chunk caches. To run the full test suite, first retrieve the public
data and rebuild or restore the cached confirmatory chunks:

```powershell
python scripts\download_multisite_data.py
python scripts\run_multisite_confirmatory.py --config configs\multisite_confirmatory.yaml
python -m pytest -q
```

## Rebuild Derived Tables and Figures

```powershell
python scripts\audit_multisite_data.py --config configs\multisite_confirmatory.yaml
python scripts\aggregate_multisite_confirmatory.py --config configs\multisite_confirmatory.yaml
python scripts\render_multisite_figures.py --config configs\multisite_confirmatory.yaml
python scripts\build_adma_manuscript_assets.py
```

Full confirmatory prediction reruns require the public raw station files listed
in `data/multisite_manifest.csv` and `data/raw/multisite/download_receipt.csv`.


## Repository note

This GitHub-oriented package contains code, configurations, tests, manifests, and compact generated tables/figures. The manuscript source is distributed separately.
