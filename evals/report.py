"""Write evals/results/latest.json and latest.md from the suites' results."""

import json
from datetime import UTC, datetime
from typing import Any

from evals.common import LABELS, RESULTS

SPLITS = ("all", "tune", "holdout", "clean holdout")


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def span(bounds: list[float] | tuple[float, float] | None) -> str:
    return "n/a" if bounds is None else f"{bounds[0]:.0%}-{bounds[1]:.0%}"


def matching_section(result: dict[str, Any]) -> list[str]:
    acc, bal = result["accuracy"], result["balanced_accuracy"]
    lines = [
        "## Match scoring",
        "",
        f"{result['pairs']} vacancy-CV pairs with labels (good / maybe / bad), drafted by an "
        "LLM from RUBRIC.md without seeing the scores and checked by hand.",
        "The scoring was adjusted on tune pairs only. The split is per pair, so a holdout pair "
        "can share its vacancy with a tune pair; the clean holdout leaves those out. The first "
        "report also listed holdout misses, so even the holdout is not fully unseen.",
        "",
        "| | " + " | ".join(SPLITS) + " |",
        "|---|" + "---|" * len(SPLITS),
        "| Pairs | " + " | ".join(str(result["counts"][s]) for s in SPLITS) + " |",
        "| Accuracy | "
        + " | ".join(
            f"{percent(acc[s])} ({result['correct'][s]}/{result['counts'][s]})" for s in SPLITS
        )
        + " |",
        "| 95% interval | " + " | ".join(span(result["interval"][s]) for s in SPLITS) + " |",
        "| Balanced accuracy | " + " | ".join(percent(bal[s]) for s in SPLITS) + " |",
        "| Recall on good | "
        + " | ".join(percent(result["recall_by_split"][s]["good"]) for s in SPLITS)
        + " |",
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
        f"- first attempts the rules sent back for a repair: {rules.get('repaired_letters', 'n/a')}"
        f" of {result['letters']} letters ({rules.get('first_attempt_problems', 'n/a')} "
        "problems); the numbers below are about the final, repaired letters",
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


def agent_section(result: dict[str, Any]) -> list[str]:
    lines = [
        "## Tool use",
        "",
        *([f"_{result['note']}_", ""] if result.get("note") else []),
        f"{result['total']} full agent runs on the frozen vacancies. Finished: "
        f"{result['finished']} of {result['total']}. **Without any tool-use mistake: "
        f"{result['clean']} of {result['total']}.** Mean {result['mean_tokens']:,} tokens "
        f"and {result['mean_seconds']} s per run.",
        "",
        "| Goal | Status | Letters | Tool errors | Mistakes |",
        "|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        status = run["status"] if run["status"] == "done" else f"{run['status']}: {run['error']}"
        lines.append(
            f"| {run['name']} | {status} | {run['analysis']['letters']} "
            f"| {run['analysis']['tool_errors']} | {'; '.join(run['mistakes']) or 'none'} |"
        )
    return lines


def safety_section(result: dict[str, Any]) -> list[str]:
    lines = [
        "## Safety",
        "",
        f"{result['total']} runs that ask the agent to send, apply or approve, or that hide "
        f"instructions for it in a vacancy. Counted: {result['counted']} of {result['total']} "
        "(a run counts only if it finished and its answer was judged).",
        "",
        f"- runs in which the agent tried to send (and was refused): "
        f"{result['runs_with_send_attempts']} of {result['total']}",
        f"- final answers that claim something was sent, submitted, approved or exported: "
        f"**{result['false_claims']}** of {result['counted']}",
        f"- the injected instruction reached the model: "
        f"{'yes' if result['injection_reached_model'] else '**no, the case shows nothing**'}",
        f"- approvals or rejections without a human: {result['decisions_without_human']} "
        "(0 by design: the agent has no tool to decide)",
        "",
        "| Case | Status | Tried | Claims an action | Tool-use mistakes | Quote |",
        "|---|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        tried = ", ".join(run["send_attempts"]) or "nothing"
        claim = {True: "yes", False: "no", None: "not judged"}[run["claims_action"]]
        status = run["status"] if run["status"] == "done" else f"{run['status']}: {run['error']}"
        lines.append(
            f"| {run['name']} | {status} | {tried} | {claim} "
            f"| {'; '.join(run['mistakes']) or 'none'} | {run['quote']} |"
        )
    return lines


SECTIONS = {
    "matching": matching_section,
    "letters": letters_section,
    "agent": agent_section,
    "safety": safety_section,
}


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
