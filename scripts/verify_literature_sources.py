from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature" / "source_verification.csv"
DOC = ROOT / "docs" / "25_citation_source_verification.md"

VERIFIED_ROWS = [
    {
        "key": "romano2019cqr",
        "verified_title": "Conformalized Quantile Regression",
        "verified_authors": "Yaniv Romano; Evan Patterson; Emmanuel Candes",
        "verified_year": "2019",
        "verified_venue": "Advances in Neural Information Processing Systems 32 (NeurIPS 2019)",
        "doi": "",
        "source_url": "https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html",
        "source_type": "official_proceedings",
        "verification_status": "verified",
        "notes": "Official NeurIPS page exposes citation title, authors, venue, and publication date.",
    },
    {
        "key": "zaffran2023missing",
        "verified_title": "Conformal Prediction with Missing Values",
        "verified_authors": "Margaux Zaffran; Aymeric Dieuleveut; Julie Josse; Yaniv Romano",
        "verified_year": "2023",
        "verified_venue": "Proceedings of the 40th International Conference on Machine Learning, PMLR 202:40578-40604",
        "doi": "",
        "source_url": "https://proceedings.mlr.press/v202/zaffran23a.html",
        "source_type": "official_proceedings",
        "verification_status": "verified",
        "notes": "Official PMLR page verifies title, authors, pages, venue, and year.",
    },
    {
        "key": "pvconformal2024",
        "verified_title": "Enhancing the reliability of probabilistic PV power forecasts using conformal prediction",
        "verified_authors": "Yvet Renkema; Lennard Visser; Tarek AlSkaif",
        "verified_year": "2024",
        "verified_venue": "Solar Energy Advances 4:100059",
        "doi": "10.1016/j.seja.2024.100059",
        "source_url": "https://doi.org/10.1016/j.seja.2024.100059",
        "source_type": "publisher_doi_crossref",
        "verification_status": "verified",
        "notes": "Crossref record is maintained by Elsevier and links to the publisher landing page.",
    },
    {
        "key": "retzlaff2025coverage",
        "verified_title": "Testing Marginal and Conditional Coverage in Conformal Prediction for Non-Stationary Time Series via Value-at-Risk Backtesting",
        "verified_authors": "Konrad Retzlaff; Filip Schlembach; Dennis Bams; Philippe Dreesen",
        "verified_year": "2025",
        "verified_venue": "Proceedings of the Fourteenth Symposium on Conformal and Probabilistic Prediction with Applications, PMLR 266:725-747",
        "doi": "",
        "source_url": "https://proceedings.mlr.press/v266/retzlaff25a.html",
        "source_type": "official_proceedings",
        "verification_status": "verified",
        "notes": "Official PMLR page verifies title, authors, pages, venue, and year.",
    },
    {
        "key": "moradi2025context",
        "verified_title": "Enhanced Renewable Energy Forecasting using Context-Aware Conformal Prediction",
        "verified_authors": "Alireza Moradi; Mathieu Tanneau; Reza Zandehshahvar; Pascal Van Hentenryck",
        "verified_year": "2025",
        "verified_venue": "arXiv:2510.15780",
        "doi": "",
        "source_url": "https://arxiv.org/abs/2510.15780",
        "source_type": "arxiv_api",
        "verification_status": "verified_preprint",
        "notes": "arXiv API verifies v2 title, authors, and first publication date 2025-10-17.",
    },
    {
        "key": "lanzilao2026ramps",
        "verified_title": "Characterization and forecasting of national-scale solar power ramp events",
        "verified_authors": "Luca Lanzilao; Angela Meyer",
        "verified_year": "2026",
        "verified_venue": "arXiv:2603.26596",
        "doi": "",
        "source_url": "https://arxiv.org/abs/2603.26596",
        "source_type": "arxiv_api",
        "verification_status": "verified_preprint",
        "notes": "arXiv API verifies v1 title, authors, and publication date 2026-03-27.",
    },
    {
        "key": "chen2022stategrid",
        "verified_title": "Solar and wind power data from the Chinese State Grid Renewable Energy Generation Forecasting Competition",
        "verified_authors": "Yongbao Chen",
        "verified_year": "2022",
        "verified_venue": "figshare dataset",
        "doi": "10.6084/m9.figshare.17304221.v4",
        "source_url": "https://doi.org/10.6084/m9.figshare.17304221.v4",
        "source_type": "datacite_figshare",
        "verification_status": "verified",
        "notes": "DataCite verifies title, creator, publisher, year, dataset type, size, and CC0 license.",
    },
    {
        "key": "suresh2026adaptive",
        "verified_title": "Model-Agnostic, Probabilistic, Hour-Ahead Solar PV Forecasting Using Adaptive Conformal Inference",
        "verified_authors": "Vishnu Suresh",
        "verified_year": "2026",
        "verified_venue": "Energies 19(6):1495",
        "doi": "10.3390/en19061495",
        "source_url": "https://doi.org/10.3390/en19061495",
        "source_type": "publisher_doi_crossref",
        "verification_status": "verified",
        "notes": "Crossref record is maintained by MDPI and links to the publisher landing page.",
    },
    {
        "key": "yan2025missingpv",
        "verified_title": "Robust photovoltaic forecasting under severe data missingness via multi-domain collaboration and covariate interaction",
        "verified_authors": "Ke Yan; Jian Liu; Jiazhen Zhang; Fan Yang; Yuan Gao; Yang Du",
        "verified_year": "2025",
        "verified_venue": "Applied Energy 401:126771",
        "doi": "10.1016/j.apenergy.2025.126771",
        "source_url": "https://doi.org/10.1016/j.apenergy.2025.126771",
        "source_type": "publisher_doi_crossref",
        "verification_status": "verified",
        "notes": "Crossref record is maintained by Elsevier and links to the publisher landing page.",
    },
    {
        "key": "zaffran2022adaptive",
        "verified_title": "Adaptive Conformal Predictions for Time Series",
        "verified_authors": "Margaux Zaffran; Olivier Feron; Yannig Goude; Julie Josse; Aymeric Dieuleveut",
        "verified_year": "2022",
        "verified_venue": "Proceedings of the 39th International Conference on Machine Learning, PMLR 162:25834-25866",
        "doi": "",
        "source_url": "https://proceedings.mlr.press/v162/zaffran22a.html",
        "source_type": "official_proceedings",
        "verification_status": "verified",
        "notes": "Official PMLR page verifies title, authors, pages, venue, and year.",
    },
    {
        "key": "xu2023spci",
        "verified_title": "Sequential Predictive Conformal Inference for Time Series",
        "verified_authors": "Chen Xu; Yao Xie",
        "verified_year": "2023",
        "verified_venue": "Proceedings of the 40th International Conference on Machine Learning, PMLR 202:38707-38727",
        "doi": "",
        "source_url": "https://proceedings.mlr.press/v202/xu23r.html",
        "source_type": "official_proceedings",
        "verification_status": "verified",
        "notes": "Official PMLR page verifies title, authors, pages, venue, and year.",
    },
    {
        "key": "lopes2024conforme",
        "verified_title": "ConForME: Multi-horizon Conditional Conformal Time Series Forecasting",
        "verified_authors": "Aloysio Galvao Lopes; Eric Goubault; Sylvie Putot; Laurent Pautet",
        "verified_year": "2024",
        "verified_venue": "Proceedings of the Thirteenth Symposium on Conformal and Probabilistic Prediction with Applications, PMLR 230:345-365",
        "doi": "",
        "source_url": "https://proceedings.mlr.press/v230/galvao-lopes24a.html",
        "source_type": "official_proceedings",
        "verification_status": "verified",
        "notes": "Official PMLR page verifies title, authors, pages, venue, and year.",
    },
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(VERIFIED_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(VERIFIED_ROWS)

    verified = sum(1 for row in VERIFIED_ROWS if row["verification_status"].startswith("verified"))
    md_lines = [
        "# Citation Source Verification",
        "",
        "**Status:** Source-level metadata verification complete for the current manuscript bibliography.",
        "",
        f"- Entries checked: {len(VERIFIED_ROWS)}",
        f"- Entries verified or verified as preprints: {verified}",
        "- Primary/authoritative sources used: official proceedings pages, arXiv API, DataCite figshare metadata, and publisher-maintained DOI/Crossref records.",
        "",
        "## Verification Table",
        "",
        "| Key | Status | Verified source | Notes |",
        "|---|---|---|---|",
    ]
    for row in VERIFIED_ROWS:
        md_lines.append(
            f"| `{row['key']}` | {row['verification_status']} | {row['source_url']} | {row['notes']} |"
        )

    md_lines.extend(
        [
            "",
            "## Remaining Citation Caveats",
            "",
            "- Preprints remain labelled as preprints in BibTeX and manuscript framing.",
            "- Publisher DOI/Crossref records verify metadata but do not by themselves prove every claim made about a cited work; manuscript claim-faithfulness still needs human reading.",
            "- No unverifiable source is used to support a novelty or priority claim.",
        ]
    )
    DOC.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} and {DOC}; verified={verified}/{len(VERIFIED_ROWS)}")


if __name__ == "__main__":
    main()
