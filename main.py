from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
MULTISITE_CONFIG = ROOT / "configs" / "multisite_confirmatory.yaml"
ADVANCED_CONFIG = ROOT / "configs" / "advanced.yaml"
DEFAULT_WORKFLOW = "all"


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]


def load_multisite_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def confirmatory_stations(cfg: dict) -> list[str]:
    audit_path = ROOT / cfg["data"]["station_audit_table"]
    if audit_path.exists():
        sys.path.insert(0, str(SCRIPTS))
        from aggregate_multisite_confirmatory import valid_confirmatory_stations

        return valid_confirmatory_stations(cfg)

    development = set(cfg["project"].get("development_sites", []))
    preferred = cfg["project"].get("confirmatory_sites_preferred", [])
    return [station for station in preferred if station not in development]


def confirmatory_jobs(config_path: Path, *, smoke: bool) -> list[Command]:
    cfg = load_multisite_config(config_path)
    config_arg = str(config_path.relative_to(ROOT))
    stations = confirmatory_stations(cfg)
    folds = [fold["name"] for fold in cfg["data"]["site1_calendar_folds"]]
    horizons = [str(horizon) for horizon in cfg["forecast"]["horizons_steps"]]
    jobs: list[Command] = []
    for station in stations:
        for fold in folds:
            for horizon in horizons:
                argv = [
                    sys.executable,
                    str(SCRIPTS / "run_multisite_confirmatory.py"),
                    "--config",
                    config_arg,
                    "--station",
                    station,
                    "--fold",
                    fold,
                    "--horizon",
                    horizon,
                ]
                if smoke:
                    argv.append("--smoke")
                jobs.append(
                    Command(
                        label=f"confirmatory {station} {fold} h{horizon}{' (smoke)' if smoke else ''}",
                        argv=tuple(argv),
                    )
                )
    return jobs


def script_command(script_name: str, *args: str) -> Command:
    return Command(
        label=script_name,
        argv=(sys.executable, str(SCRIPTS / script_name), *args),
    )


def workflow_commands(name: str, *, config_path: Path, smoke: bool) -> list[Command]:
    multisite_config = str(config_path.relative_to(ROOT))
    advanced_config = str(ADVANCED_CONFIG.relative_to(ROOT))

    workflows: dict[str, Callable[[], list[Command]]] = {
        "verify": lambda: [
            Command(label="pytest", argv=(sys.executable, "-m", "pytest", "-q")),
        ],
        "multisite-download": lambda: [script_command("download_multisite_data.py")],
        "multisite-audit": lambda: [
            script_command("audit_multisite_data.py", "--config", multisite_config),
        ],
        "multisite-confirmatory": lambda: confirmatory_jobs(config_path, smoke=smoke),
        "multisite-aggregate": lambda: [
            script_command("aggregate_multisite_confirmatory.py", "--config", multisite_config),
        ],
        "multisite-figures": lambda: [
            script_command("render_multisite_figures.py", "--config", multisite_config),
        ],
        "multisite-rebuild": lambda: [
            *workflow_commands("multisite-audit", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-aggregate", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-figures", config_path=config_path, smoke=smoke),
        ],
        "multisite-full": lambda: [
            *workflow_commands("multisite-download", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-audit", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-confirmatory", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-aggregate", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-figures", config_path=config_path, smoke=smoke),
            Command(label="pytest", argv=(sys.executable, "-m", "pytest", "-q")),
        ],
        "advanced": lambda: [
            script_command("run_advanced_study.py", "--config", advanced_config),
            script_command("aggregate_advanced_study.py", "--config", advanced_config),
            script_command("postprocess_advanced_results.py"),
            script_command("render_advanced_figures.py"),
            script_command("audit_advanced_visualizations.py"),
        ],
        "all": lambda: [
            *workflow_commands("multisite-download", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-audit", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-confirmatory", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-aggregate", config_path=config_path, smoke=smoke),
            *workflow_commands("multisite-figures", config_path=config_path, smoke=smoke),
            *workflow_commands("advanced", config_path=config_path, smoke=smoke),
            Command(label="pytest", argv=(sys.executable, "-m", "pytest", "-q")),
        ],
    }

    if name not in workflows:
        known = ", ".join(sorted(workflows))
        raise ValueError(f"Unknown workflow '{name}'. Known workflows: {known}")
    return workflows[name]()


def run_commands(
    commands: Sequence[Command],
    *,
    dry_run: bool,
    continue_on_error: bool,
) -> int:
    failures = 0
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {command.label}", flush=True)
        print(" ".join(command.argv), flush=True)
        if dry_run:
            continue
        result = subprocess.run(command.argv, cwd=ROOT, check=False)
        if result.returncode != 0:
            failures += 1
            print(f"FAILED: {command.label} (exit {result.returncode})", flush=True)
            if not continue_on_error:
                return result.returncode
    return failures


def list_workflows() -> None:
    print("Workflows:")
    for name in sorted(
        [
            "verify",
            "multisite-download",
            "multisite-audit",
            "multisite-confirmatory",
            "multisite-aggregate",
            "multisite-figures",
            "multisite-rebuild",
            "multisite-full",
            "advanced",
            "all",
        ]
    ):
        print(f"  {name}")
    print("\nScripts:")
    for script in sorted(path.name for path in SCRIPTS.glob("*.py")):
        print(f"  {script}")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(MULTISITE_CONFIG.relative_to(ROOT)),
        help="Multi-site confirmatory config (default: configs/multisite_confirmatory.yaml)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run confirmatory jobs in smoke-test mode instead of full prediction reruns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining commands after a failure.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run solar-ramp-reliability scripts and documented workflows. "
            f"With no subcommand, runs the '{DEFAULT_WORKFLOW}' workflow."
        ),
    )
    add_runtime_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List available workflows and scripts.")
    list_parser.set_defaults(handler=lambda args: list_workflows() or 0)

    run_parser = subparsers.add_parser("run", help="Run a named workflow.")
    add_runtime_arguments(run_parser)
    run_parser.add_argument(
        "workflow",
        help="Workflow name. Use 'list' to see options.",
    )
    run_parser.set_defaults(handler=run_workflow)

    script_parser = subparsers.add_parser("script", help="Run one script from scripts/.")
    add_runtime_arguments(script_parser)
    script_parser.add_argument("script_name", help="Script filename, e.g. audit_multisite_data.py")
    script_parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script.")
    script_parser.set_defaults(handler=run_script)

    parser.set_defaults(handler=run_default)
    return parser


def run_default(args: argparse.Namespace) -> int:
    args.workflow = DEFAULT_WORKFLOW
    return run_workflow(args)


def run_workflow(args: argparse.Namespace) -> int:
    config_path = ROOT / args.config
    commands = workflow_commands(args.workflow, config_path=config_path, smoke=args.smoke)
    if not commands:
        print(f"Workflow '{args.workflow}' has no commands.")
        return 0
    return run_commands(
        commands,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


def run_script(args: argparse.Namespace) -> int:
    script_path = SCRIPTS / args.script_name
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    command = Command(
        label=args.script_name,
        argv=(sys.executable, str(script_path), *args.script_args),
    )
    return run_commands(
        [command],
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exit_code = args.handler(args)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
