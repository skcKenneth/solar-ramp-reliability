from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class ConfirmatoryJob:
    station: str
    fold: str
    horizon_steps: int

    @property
    def key(self) -> str:
        return f"{self.station}_{self.fold}_h{self.horizon_steps}"


def resolve_repo_path(path: str | Path, *, label: str, root: Path = ROOT) -> Path:
    repository = root.resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else repository / candidate).resolve()
    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the repository: {path}") from exc
    return resolved


def valid_confirmatory_stations(cfg: dict, *, root: Path = ROOT) -> list[str]:
    audit_path = resolve_repo_path(cfg["data"]["station_audit_table"], label="Station audit path", root=root)
    audit = pd.read_csv(audit_path)
    development = {str(value) for value in cfg["project"].get("development_sites", [])}
    valid = audit[(audit["status"] == "valid") & (~audit["site_id"].astype(str).isin(development))]
    stations = sorted(valid["site_id"].astype(str).tolist())
    if not stations:
        raise RuntimeError("Station audit contains no valid evaluation station")
    return stations


def build_jobs(cfg: dict, stations: Iterable[str]) -> list[ConfirmatoryJob]:
    folds = [str(fold["name"]) for fold in cfg["data"]["site1_calendar_folds"]]
    horizons = [int(value) for value in cfg["forecast"]["horizons_steps"]]
    jobs = [ConfirmatoryJob(str(station), fold, horizon) for station in stations for fold in folds for horizon in horizons]
    if len({job.key for job in jobs}) != len(jobs):
        raise RuntimeError("Job matrix contains duplicate keys")
    return jobs


def validate_frozen_topology(cfg: dict, stations: Iterable[str], jobs: Iterable[ConfirmatoryJob]) -> None:
    stations = [str(value) for value in stations]
    folds = [str(fold["name"]) for fold in cfg["data"]["site1_calendar_folds"]]
    horizons = [int(value) for value in cfg["forecast"]["horizons_steps"]]
    jobs = list(jobs)
    if len(stations) != 5 or len(set(stations)) != 5:
        raise RuntimeError(f"Expected five unique evaluation stations; found {stations}")
    if tuple(folds) != ("q1_2020", "q2_2020", "q3_2020", "q4_2020"):
        raise RuntimeError(f"Unexpected fold topology: {folds}")
    if len(horizons) != 2 or set(horizons) != {1, 4}:
        raise RuntimeError(f"Unexpected horizon topology: {horizons}")
    expected = {(station, fold, horizon) for station in stations for fold in folds for horizon in horizons}
    actual = {(job.station, job.fold, job.horizon_steps) for job in jobs}
    if len(jobs) != 40 or actual != expected:
        raise RuntimeError("Expected the complete 5 x 4 x 2 job matrix")


def marker_path(cfg: dict, job: ConfirmatoryJob, *, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return ROOT / cfg["outputs"]["completion_markers"] / f"{job.key}{suffix}.json"


def artifact_paths(cfg: dict, job: ConfirmatoryJob) -> list[Path]:
    chunks = ROOT / cfg["outputs"]["root"] / "stations" / job.station / "cache" / "chunks"
    stem = f"{job.fold}_h{job.horizon_steps}"
    return [
        chunks / f"predictions_{stem}.pkl.gz",
        chunks / f"run_{stem}.csv",
        chunks / f"calibration_{stem}.csv",
    ]


def run_checked(command: list[str], *, label: str) -> float:
    started = time.time()
    result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if result.returncode:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return time.time() - started


def verify_job(cfg: dict, job: ConfirmatoryJob, *, smoke: bool) -> None:
    path = marker_path(cfg, job, smoke=smoke)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_status = "SMOKE_OK" if smoke else "COMPLETE"
    expected_job = {"station": job.station, "fold": job.fold, "horizon_steps": job.horizon_steps}
    if payload.get("schema_version") != 2 or payload.get("status") != expected_status:
        raise RuntimeError(f"Invalid completion marker: {path.relative_to(ROOT)}")
    if payload.get("job") != expected_job:
        raise RuntimeError(f"Marker identity mismatch: {path.relative_to(ROOT)}")
    if not smoke:
        expected = {path.relative_to(ROOT).as_posix() for path in artifact_paths(cfg, job)}
        recorded = {item.get("path") for item in payload.get("artifacts", [])}
        if recorded != expected or not all(path.is_file() for path in artifact_paths(cfg, job)):
            raise RuntimeError(f"Artifact set mismatch for {job.key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed multi-site evaluation matrix.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--downstream", choices=("all", "none"), default="all")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_repo_path(args.config, label="Configuration path")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_seconds = run_checked(
        [sys.executable, str(ROOT / "scripts" / "audit_data.py"), "--config", args.config],
        label="data audit",
    )
    stations = valid_confirmatory_stations(cfg)
    jobs = build_jobs(cfg, stations)
    validate_frozen_topology(cfg, stations, jobs)
    failures: list[str] = []
    job_seconds = 0.0
    for index, job in enumerate(jobs, start=1):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_job.py"),
            "--config", args.config,
            "--station", job.station,
            "--fold", job.fold,
            "--horizon", str(job.horizon_steps),
        ]
        if args.smoke:
            command.append("--smoke")
        if args.force:
            command.append("--force")
        try:
            elapsed = run_checked(command, label=job.key)
            job_seconds += elapsed
            verify_job(cfg, job, smoke=args.smoke)
            print(f"[{index:02d}/40] PASS {job.key} ({elapsed:.2f}s)", flush=True)
        except Exception as exc:
            failures.append(f"{job.key}: {exc}")
            print(f"[{index:02d}/40] FAIL {job.key}: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break
    if failures:
        raise SystemExit("; ".join(failures))

    downstream_seconds = 0.0
    if not args.smoke and args.downstream == "all":
        for label, script in (
            ("aggregation", "aggregate_results.py"),
            ("specifications", "generate_specifications.py"),
            ("analysis", "analyze_results.py"),
            ("figures", "render_figures.py"),
            ("output manifest", "update_output_manifest.py"),
            ("scientific regression", "verify_results.py"),
        ):
            elapsed = run_checked(
                [sys.executable, str(ROOT / "scripts" / script), "--config", args.config],
                label=label,
            )
            downstream_seconds += elapsed
            print(f"PASS {label} ({elapsed:.2f}s)", flush=True)

    expected_artifacts = [] if args.smoke else [path for job in jobs for path in artifact_paths(cfg, job)]
    if not args.smoke and (len(expected_artifacts) != 120 or not all(path.is_file() for path in expected_artifacts)):
        raise RuntimeError("Full run did not produce all 120 intermediate artifacts")
    print(json.dumps({
        "status": "SMOKE_COMPLETE" if args.smoke else "COMPLETE",
        "stations": stations,
        "jobs": len(jobs),
        "intermediate_artifacts": len(expected_artifacts),
        "audit_seconds": round(audit_seconds, 3),
        "job_seconds": round(job_seconds, 3),
        "downstream_seconds": round(downstream_seconds, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
