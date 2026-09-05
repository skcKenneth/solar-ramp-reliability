from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable


SEED_NAMESPACE = "solar-ramp-reliability/job-seed/v1"
MODEL_SEED_COMPONENTS = (
    "clean_model",
    "lgbm_model",
    "augmentation",
    "robust_model",
    "ramp_risk_model",
    "local_calibrator",
)
RECORDED_DISTRIBUTIONS = (
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
    "PyYAML", "lightgbm", "Pillow", "pytest",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_execution_source_sha256(root: str | Path) -> str:
    root = Path(root).resolve()
    candidates = [
        root / "pyproject.toml",
        root / "requirements-lock.txt",
        root / "scripts" / "run_advanced_study.py",
        root / "scripts" / "run_job.py",
    ]
    candidates.extend((root / "src").rglob("*.py"))
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    if not files:
        raise FileNotFoundError(f"No model-execution source files found under {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def model_execution_source_sha256_candidates(root: str | Path) -> tuple[str, ...]:
    return (model_execution_source_sha256(root),)


def stable_job_seed(
    base_seed: int,
    station: str,
    fold: str,
    horizon_steps: int,
    component: str,
) -> int:
    if int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be positive")
    fields = (str(station).strip(), str(fold).strip(), str(component).strip())
    if any(not field for field in fields):
        raise ValueError("station, fold, and component must be non-empty")
    payload = json.dumps(
        [SEED_NAMESPACE, int(base_seed), fields[0], fields[1], int(horizon_steps), fields[2]],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)
    return seed or 1


def job_seed_map(
    base_seed: int,
    station: str,
    fold: str,
    horizon_steps: int,
    components: Iterable[str] = MODEL_SEED_COMPONENTS,
) -> dict[str, int]:
    return {
        component: stable_job_seed(base_seed, station, fold, horizon_steps, component)
        for component in components
    }


def seed_audit_payload(
    *,
    base_seed: int,
    stations: Iterable[str],
    folds: Iterable[str],
    horizons_steps: Iterable[int],
    config_sha256: str,
    model_source_sha256: str,
) -> dict:
    jobs = [
        {
            "station": str(station),
            "fold": str(fold),
            "horizon_steps": int(horizon),
            "component_seeds": job_seed_map(base_seed, station, fold, horizon),
        }
        for station in stations
        for fold in folds
        for horizon in horizons_steps
    ]
    return {
        "schema_version": 1,
        "status": "DETERMINISTIC_SEED_MAP",
        "algorithm": "uint32_be(first_4_bytes(SHA256(canonical_json_payload))); zero maps to one",
        "namespace": SEED_NAMESPACE,
        "base_seed": int(base_seed),
        "config_sha256": str(config_sha256),
        "model_execution_source_sha256": str(model_source_sha256),
        "components": list(MODEL_SEED_COMPONENTS),
        "job_count": len(jobs),
        "jobs": jobs,
    }


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in RECORDED_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def runtime_environment() -> dict:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": installed_versions(),
    }


def atomic_write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def artifact_record(path: str | Path, *, root: str | Path) -> dict:
    path = Path(path)
    root = Path(root)
    record = {"path": path.relative_to(root).as_posix(), "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return record


def file_matches_sha256_receipt(
    path: str | Path,
    *,
    expected_sha256: object,
    expected_bytes: object | None = None,
) -> bool:
    path = Path(path)
    if not path.is_file() or not isinstance(expected_sha256, str):
        return False
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        return False
    return file_sha256(path).lower() == expected_sha256.lower()


def file_sha256_transport_candidates(path: str | Path) -> tuple[str, ...]:
    return (file_sha256(path),)
