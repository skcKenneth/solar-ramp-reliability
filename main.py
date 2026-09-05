from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = "configs/experiment.yaml"
COMMANDS = {
    "audit": ["scripts/audit_data.py"],
    "smoke": ["scripts/run_all.py", "--smoke"],
    "run": ["scripts/run_all.py", "--force"],
    "aggregate": ["scripts/aggregate_results.py"],
    "analyze": ["scripts/analyze_results.py"],
    "figures": ["scripts/render_figures.py"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the solar-ramp reliability workflow.")
    parser.add_argument("command", choices=["list", *COMMANDS])
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(COMMANDS))
        return 0
    script, *extra = COMMANDS[args.command]
    command = [sys.executable, str(ROOT / script), "--config", args.config, *extra]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
