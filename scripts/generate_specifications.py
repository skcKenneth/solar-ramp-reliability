from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solar_reliability.specifications import DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR, generate_specifications


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic method and data specifications.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for name, path in generate_specifications(args.config, args.output_dir).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
