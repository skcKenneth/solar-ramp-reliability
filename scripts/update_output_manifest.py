"""Canonical transport checksums, separate from scientific regression checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update_manifest(root: Path = ROOT) -> dict:
    records = []
    for path in sorted((root / 'outputs').rglob('*'), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if path.suffix in {'.csv', '.json', '.md'}:
            canonical = data.replace(b'\r\n', b'\n')
            if canonical != data:
                path.write_bytes(canonical)
            data = canonical
        records.append({'path': path.relative_to(root).as_posix(), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
    payload = {'schema_version': 1, 'output_count': len(records),
               'purpose': 'transport integrity only; use verify_results.py for scientific regression',
               'text_line_endings': 'LF', 'outputs': records}
    (root / 'OUTPUT_MANIFEST.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8', newline='\n')
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiment.yaml', help='accepted for workflow compatibility')
    parser.parse_args()
    print(json.dumps({'output_count': update_manifest()['output_count']}))
