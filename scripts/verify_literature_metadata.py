from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


ENTRY_RE = re.compile(r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,]+),(?P<body>.*?)(?=\n@\w+\s*\{|\Z)", re.S)
FIELD_RE = re.compile(r"(?P<field>\w+)\s*=\s*[\{\"](?P<value>.*?)[\}\"]\s*,?", re.S)


def parse_bib(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    for match in ENTRY_RE.finditer(text):
        row = {"entry_type": match.group("type"), "key": match.group("key").strip()}
        for field in FIELD_RE.finditer(match.group("body")):
            value = " ".join(field.group("value").split())
            row[field.group("field").lower()] = value
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BibTeX metadata completeness audit.")
    parser.add_argument("--bib", default="literature/references.bib")
    parser.add_argument("--output", default="literature/metadata_audit.csv")
    args = parser.parse_args()

    rows = parse_bib(ROOT / args.bib)
    verification_path = ROOT / "literature" / "source_verification.csv"
    verification_status: dict[str, str] = {}
    if verification_path.exists():
        with verification_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                verification_status[row["key"]] = row.get("verification_status", "")
    required = ["author", "title", "year"]
    output_rows: list[dict[str, str | bool]] = []
    for row in rows:
        missing = [field for field in required if not row.get(field)]
        venue_present = bool(row.get("journal") or row.get("booktitle") or row.get("publisher") or row.get("howpublished"))
        identifier_present = bool(row.get("doi") or row.get("url") or row.get("eprint"))
        output_rows.append(
            {
                "key": row["key"],
                "entry_type": row["entry_type"],
                "title_present": bool(row.get("title")),
                "author_present": bool(row.get("author")),
                "year_present": bool(row.get("year")),
                "venue_present": venue_present,
                "identifier_present": identifier_present,
                "doi": row.get("doi", ""),
                "missing_required_fields": ";".join(missing),
                "offline_completeness_pass": not missing and venue_present,
                "web_verification_status": verification_status.get(row["key"], "not_verified"),
            }
        )
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()) if output_rows else ["key"])
        writer.writeheader()
        writer.writerows(output_rows)
    failed = [row for row in output_rows if not row["offline_completeness_pass"]]
    print(f"audited {len(output_rows)} entries; offline completeness failures={len(failed)}; wrote {out}")


if __name__ == "__main__":
    main()
