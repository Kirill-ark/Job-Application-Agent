"""Write evals/results/latest.json and latest.md from the suites' results."""

import json
from datetime import UTC, datetime
from typing import Any

from evals.common import LABELS, RESULTS

SPLITS = ("all", "tune", "holdout")


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def matching_section(result: dict[str, Any]) -> list[str]:
    acc, bal = result["accuracy"], result["balanced_accuracy"]
    lines = [
        "## Match scoring",
        "",
        f"{result['pairs']} vacancy-CV pairs with hand labels (good / maybe / bad).",
        "Holdout pairs were never used to adjust the scoring.",
        "",
        "| | all | tune | holdout |",
        "|---|---|---|---|",
        "| Accuracy | " + " | ".join(percent(acc[s]) for s in SPLITS) + " |",
        "| Balanced accuracy | " + " | ".join(percent(bal[s]) for s in SPLITS) + " |",
        "",
        "Confusion matrix (rows: hand label, columns: system label):",
        "",
        "| | " + " | ".join(LABELS) + " | recall |",
        "|---|" + "---|" * (len(LABELS) + 1),
    ]
    for expected in LABELS:
        counts = " | ".join(str(result["confusion"][expected][p]) for p in LABELS)
        lines.append(f"| **{expected}** | {counts} | {percent(result['recall'][expected])} |")
    lines += ["", "### Misses", ""]
    if not result["misses"]:
        lines.append("None.")
    else:
        lines += [
            "| Vacancy | CV | Hand | System | Score | Why the hand label |",
            "|---|---|---|---|---|---|",
        ]
        for miss in sorted(result["misses"], key=lambda m: (m["expected"], -m["score"])):
            lines.append(
                f"| {miss['title'][:55]} | {miss['cv']} | {miss['expected']} | {miss['predicted']} "
                f"| {miss['score']} | {miss['why']} |"
            )
    return lines


SECTIONS = {"matching": matching_section}


def write(results: dict[str, Any]) -> None:
    """Merge these suites into the saved results and rewrite both files."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "latest.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    for suite, result in results.items():
        saved[suite] = {**result, "ran_at": datetime.now(UTC).isoformat(timespec="seconds")}
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=1) + "\n")

    lines = [
        "# Eval results",
        "",
        "Written by `python -m evals.run_evals`. Do not edit by hand.",
        "",
    ]
    for suite, section in SECTIONS.items():
        if suite in saved:
            lines += section(saved[suite]) + ["", f"_Ran at {saved[suite]['ran_at']}._", ""]
    (RESULTS / "latest.md").write_text("\n".join(lines))
