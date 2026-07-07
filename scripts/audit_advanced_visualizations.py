from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "advanced" / "figures"
TABLES = ROOT / "outputs" / "advanced" / "tables"
records = []
for png in sorted(FIGURES.glob("*.png")):
    with Image.open(png) as image:
        width, height = image.size
    pdf = png.with_suffix(".pdf")
    svg = png.with_suffix(".svg")
    status = "pass" if width >= 1200 and height >= 600 and pdf.exists() and svg.exists() else "review"
    records.append(
        {
            "figure": png.name,
            "width_px": width,
            "height_px": height,
            "pdf_vector": pdf.exists(),
            "svg_vector": svg.exists(),
            "png_bytes": png.stat().st_size,
            "status": status,
        }
    )
manifest = {"figures": records, "all_pass": bool(records) and all(r["status"] == "pass" for r in records)}
(TABLES / "advanced_visualization_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
