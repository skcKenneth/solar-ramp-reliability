from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.specifications import DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR, generate_specifications
from reporting_support import write_reporting_support


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic method and data specifications.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for name, path in generate_specifications(args.config, args.output_dir).items():
        print(f"{name}: {path}")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    write_reporting_support(ROOT, cfg, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
