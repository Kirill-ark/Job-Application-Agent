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


def letters_section(result: dict[str, Any]) -> list[str]:
    verdicts, rules = result["verdicts"], result["rules"]
    total = result["sentences"]
    lines = [
        "## Cover letter groundedness",
        "",
        f"{result['letters']} letters, {total} sentences ({result['claims']} claims about the "
        "applicant). Each sentence was checked by the rules and by an LLM judge of another "
        "model family than the writer.",
        "",
        "| | sentences | share |",
        "|---|---|---|",
    ]
    for verdict in ("supported", "wrong_citation", "partly", "unsupported"):
        share = verdicts[verdict] / total if total else None
        lines.append(
            f"| Judge: {verdict.replace('_', ' ')} | {verdicts[verdict]} | {percent(share)} |"
        )
    lines += [
        "",
        f"**Hallucination rate: {percent(result['hallucination_rate'])} of sentences** "
        "(judged partly or unsupported). "
        f"Letters without any: {result['letters_without_hallucination']} of {result['letters']}.",
        "",
        "Rules compared with the judge:",
        "",
        f"- flagged by the rules: {rules['flagged']}, of which the judge also found a problem "
        f"in {rules['caught']}",
        f"- false alarms (rules flagged, judge says supported): {len(rules['false_alarms'])}",
        f"- missed by the rules (judge found a problem): {len(rules['missed'])}",
    ]
    if result["judge_failed"]:
        lines += [
            "",
            f"Not judged ({len(result['judge_failed'])}): " + "; ".join(result["judge_failed"]),
        ]

    def table(title: str, rows: list[dict[str, Any]], note: str) -> list[str]:
        if not rows:
            return []
        out = ["", f"### {title}", "", "| Letter | Sentence | " + note + " |", "|---|---|---|"]
        for row in rows:
            detail = row["reason"] if note == "Judge" else "; ".join(row["rule_problems"])
            out.append(f"| {row['case']} | {row['text']} | {detail} |".replace("\n", " "))
        return out

    lines += table("Sentences the judge found not backed by the CV", result["judged_bad"], "Judge")
    lines += table("False alarms of the rules", rules["false_alarms"], "Rules")
    return lines


SECTIONS = {"matching": matching_section, "letters": letters_section}


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
