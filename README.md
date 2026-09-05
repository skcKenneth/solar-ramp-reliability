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
```

On macOS or Linux, use `.venv/bin/python` instead.

## Verify the package

```powershell
python -m pytest -q -p no:cacheprovider
python main.py audit
python main.py smoke
python main.py figures
```

The smoke command checks every cell of the fixed 5-station x 4-fold x 2-horizon
matrix without fitting the models.

## Rebuild all results

```powershell
python main.py run
```

This command audits the data, fits all 40 jobs, checks their 120 intermediate
artifacts, aggregates the results, generates the derived CSV/JSON files, and
renders the figures. Temporary model chunks and completion markers are written
under `work/` and are intentionally excluded from the repository ZIP.

Included result artifacts are under `outputs/tables/` and `outputs/figures/`.
`OUTPUT_MANIFEST.json` records their byte sizes and SHA-256 digests.
