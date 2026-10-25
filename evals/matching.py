"""Suite 1: does score_match agree with the hand labels?

No LLM: the score and its label are computed by code, so this suite is free,
fast and gives the same numbers on every run.
"""

import tempfile
from pathlib import Path
from typing import Any

from evals.common import LABELS, database_with_cases, load_cvs, load_match_labels
from job_agent.matching import score_match
from job_agent.skills import load_taxonomy


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    return sum(r["expected"] == r["predicted"] for r in rows) / len(rows) if rows else None


def confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """matrix[expected][predicted] = count."""
    matrix = {e: dict.fromkeys(LABELS, 0) for e in LABELS}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    return matrix


def recall_by_label(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Of the pairs labeled X, the share the system also called X."""
    return {label: accuracy([r for r in rows if r["expected"] == label]) for label in LABELS}


def run() -> dict[str, Any]:
    taxonomy = load_taxonomy()
    cvs = load_cvs(taxonomy)
    with tempfile.TemporaryDirectory() as folder:
        conn = database_with_cases(Path(folder) / "evals.db")
        rows = []
        for case in load_match_labels():
            result = score_match(conn, case["vacancy"], cvs[case["cv"]], taxonomy)
            rows.append(
                {
                    "vacancy": case["vacancy"],
                    "title": result["title"],
                    "cv": case["cv"],
                    "split": case["split"],
                    "expected": case["label"],
                    "predicted": result["label"],
                    "score": result["score"],
                    "components": result["components"],
                    "why": case["why"],
                }
            )
        conn.close()

    splits = {
        name: [r for r in rows if name == "all" or r["split"] == name]
        for name in ("all", "tune", "holdout")
    }
    labels = list(LABELS)
    return {
        "pairs": len(rows),
        "accuracy": {name: accuracy(part) for name, part in splits.items()},
        # Mean recall over the three labels: an always-"bad" system would get 1/3,
        # while its plain accuracy would look good because most pairs are bad.
        "balanced_accuracy": {
            name: sum(v for v in recall_by_label(part).values() if v is not None) / len(labels)
            for name, part in splits.items()
        },
        "recall": recall_by_label(rows),
        "confusion": confusion(rows),
        "misses": [r for r in rows if r["expected"] != r["predicted"]],
        "rows": rows,
    }
