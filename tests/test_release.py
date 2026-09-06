from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_all import build_jobs, valid_confirmatory_stations, validate_frozen_topology


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_job_matrix_has_40_unique_jobs() -> None:
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    stations = valid_confirmatory_stations(cfg)
    jobs = build_jobs(cfg, stations)
    validate_frozen_topology(cfg, stations, jobs)
    assert len(jobs) == len({job.key for job in jobs}) == 40


def test_raw_data_manifest_resolves_all_eight_files() -> None:
    manifest = pd.read_csv(ROOT / "data/multisite_manifest.csv")
    assert len(manifest) == 8
    for filename in manifest["filename"].astype(str):
        assert (ROOT / "data/raw/multisite" / filename).is_file()


def test_output_manifest_covers_every_result() -> None:
    payload = json.loads((ROOT / "OUTPUT_MANIFEST.json").read_text(encoding="utf-8"))
    records = payload["outputs"]
    actual = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "outputs").rglob("*") if path.is_file())
    assert payload["output_count"] == len(records) == len(actual) == 33
    assert [record["path"] for record in records] == actual
    for record in records:
        data = (ROOT / record["path"]).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]


def test_repository_has_no_document_sources_or_personal_metadata() -> None:
    forbidden_suffixes = {".tex", ".bib", ".bst", ".cls", ".doc", ".docx", ".xlsx"}
    assert not [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]


def test_pdf_metadata_is_neutral() -> None:
    pdfs = sorted((ROOT / "outputs/figures").glob("*.pdf"))
    assert len(pdfs) == 3
    for path in pdfs:
        data = path.read_bytes()
        assert bytes.fromhex("2f417574686f72") not in data
        assert bytes.fromhex("41444d41") not in data
        assert bytes.fromhex("63616d6572612d7265616479") not in data.lower()
