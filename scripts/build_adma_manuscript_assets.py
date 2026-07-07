from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "multisite" / "tables"
FIGURES = ROOT / "outputs" / "multisite" / "figures"
PAPER = ROOT / "paper"


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_gate_table() -> None:
    gates = pd.read_csv(TABLES / "candidate_station_gates.csv")
    rows = []
    for row in gates.itertuples(index=False):
        rows.append(
            [
                f"{int(row.horizon_steps) * 15} min",
                row.method.replace("robust_", "").replace("_", " "),
                f"{row.mean_improvement:.3f}",
                f"{int(row.stations_improved_n)}/{int(row.stations)}",
                f"[{row.station_bootstrap_ci_low:.3f}, {row.station_bootstrap_ci_high:.3f}]",
                "pass" if row.comparative_gate_passed else "fail",
                "pass" if row.absolute_reliability_gate_passed else "fail",
            ]
        )
    body = "\n".join(" & ".join(tex_escape(cell) for cell in cells) + r" \\" for cells in rows)
    table = (
        r"\begin{tabular}{lllrrrr}" "\n"
        r"\toprule" "\n"
        r"Horizon & Method & Mean improvement & Stations improved & Station CI & Comparative & Absolute \\" "\n"
        r"\midrule" "\n"
        f"{body}\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
    )
    (PAPER / "tables" / "multisite_candidate_gates.tex").write_text(table, encoding="utf-8")


def write_numbers() -> None:
    verdict = json.loads((TABLES / "multisite_verdict.json").read_text(encoding="utf-8"))
    gates = pd.read_csv(TABLES / "candidate_station_gates.csv")
    local = gates[gates["method"] == "robust_local_cqr"]
    numbers = {
        "verdict": verdict["verdict"],
        "station_count": verdict["station_count"],
        "prediction_chunks": verdict["prediction_chunks"],
        "conditional_metric_rows": verdict["conditional_metric_rows"],
        "phenomenon_station_counts_by_horizon": verdict["phenomenon_station_counts_by_horizon"],
        "local_mean_improvement_by_horizon": {
            str(int(row.horizon_steps)): float(row.mean_improvement) for row in local.itertuples(index=False)
        },
        "local_max_worst_undercoverage_by_horizon": {
            str(int(row.horizon_steps)): float(row.max_method_worst_undercoverage)
            for row in local.itertuples(index=False)
        },
        "method_superiority_claim_allowed": verdict["method_superiority_claim_allowed"],
    }
    (PAPER / "tables" / "multisite_key_numbers.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")


def copy_figures() -> None:
    stale_outputs = {
        "fig01_station_worst_undercoverage.pdf",
        "fig02_station_aware_improvement.pdf",
        "fig03_oracle_conditioning_gap.pdf",
        "fig04_ramp_risk_degradation.pdf",
    }
    for name in stale_outputs:
        path = PAPER / "figures" / name
        if path.exists():
            path.unlink()
    mapping = {
        "fig20_confirmation_overview.pdf": "fig01_confirmation_overview.pdf",
        "fig20_station_worst_undercoverage.pdf": "fig02_station_worst_undercoverage.pdf",
        "fig21_station_aware_improvement.pdf": "fig03_station_aware_improvement.pdf",
        "fig22_oracle_conditioning_gap.pdf": "fig04_oracle_event_diagnostic.pdf",
    }
    for source, target in mapping.items():
        shutil.copyfile(FIGURES / source, PAPER / "figures" / target)
    (PAPER / "figures" / "FIGURE_SOURCES.md").write_text(
        "\n".join(
            [
                "# Figure Sources",
                "",
                "| Paper figure | Source | Table provenance |",
                "|---|---|---|",
                "| fig01_confirmation_overview.pdf | outputs/multisite/figures/fig20_confirmation_overview.pdf | outputs/multisite/tables/station_data_audit.csv; outputs/multisite/tables/multisite_verdict.json; outputs/multisite/tables/candidate_station_gates.csv |",
                "| fig02_station_worst_undercoverage.pdf | outputs/multisite/figures/fig20_station_worst_undercoverage.pdf | outputs/multisite/tables/station_level_effects.csv |",
                "| fig03_station_aware_improvement.pdf | outputs/multisite/figures/fig21_station_aware_improvement.pdf | outputs/multisite/tables/candidate_station_gates.csv |",
                "| fig04_oracle_event_diagnostic.pdf | outputs/multisite/figures/fig22_oracle_conditioning_gap.pdf | outputs/multisite/tables/oracle_gap_summary.csv; outputs/multisite/tables/ramp_risk_diagnostics.csv |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_blind_readme() -> None:
    text = """# Blind Manuscript Asset README

This directory contains draft ADMA manuscript assets generated from repository outputs.

Current status:

- Multi-site decision: `GO_FULL_PAPER_EVALUATION`.
- Method-superiority claim: not allowed.
- Absolute reliability gate: failed for all deployable candidates.
- Figures are copied from `outputs/multisite/figures/` and retain table provenance in `paper/figures/FIGURE_SOURCES.md`.
- This is not a complete submission package until human review clears; official ADMA/Springer instructions were most recently checked on 2026-07-05.

Human authors must verify all policy, citation, anonymity, and manuscript claims before submission.
"""
    (PAPER / "README_BLIND.md").write_text(text, encoding="utf-8")


def main() -> None:
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    write_gate_table()
    write_numbers()
    copy_figures()
    write_blind_readme()
    print(f"Wrote manuscript assets under {PAPER}")


if __name__ == "__main__":
    main()
