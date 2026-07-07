from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the processed eight-site solar CSV set.")
    parser.add_argument("--manifest", default="data/multisite_manifest.csv")
    parser.add_argument("--output", default="data/raw/multisite")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    records = []
    for row in manifest.itertuples(index=False):
        destination = output / row.filename
        if destination.exists() and not args.overwrite:
            status = "already_present"
        else:
            print(f"downloading {row.site_id}: {row.source_url}", flush=True)
            request = urllib.request.Request(row.source_url, headers={"User-Agent": "solar-ramp-reliability/0.2"})
            with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
                handle.write(response.read())
            status = "downloaded"
        records.append(
            {
                "site_id": row.site_id,
                "nominal_capacity_mw": row.nominal_capacity_mw,
                "path": str(destination.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "status": status,
            }
        )
    pd.DataFrame(records).to_csv(output / "download_receipt.csv", index=False)
    print(f"wrote {output / 'download_receipt.csv'}")


if __name__ == "__main__":
    main()
