from __future__ import annotations

from pathlib import Path
import json
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
fig_dir = ROOT / "outputs" / "figures"
records = []
for png in sorted(fig_dir.glob("*.png")):
    with Image.open(png) as image:
        width, height = image.size
    pdf = png.with_suffix(".pdf")
    records.append(
        {
            "figure": png.name,
            "width_px": width,
            "height_px": height,
            "has_vector_pdf": pdf.exists(),
            "png_bytes": png.stat().st_size,
            "pdf_bytes": pdf.stat().st_size if pdf.exists() else 0,
            "status": "pass" if width >= 1200 and height >= 500 and pdf.exists() else "review",
        }
    )
output = ROOT / "outputs" / "tables" / "visualization_audit.json"
output.write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
