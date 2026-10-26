"""Suite 1: does score_match agree with the hand labels?

No LLM: the score and its label are computed by code, so this suite is free,
fast and gives the same numbers on every run.
"""

import math
import tempfile
from pathlib import Path
from typing import Any

from evals.common import LABELS, database_with_cases, load_cvs, load_match_labels
from job_agent.matching import score_match
from job_agent.skills import load_taxonomy


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    return sum(r["expected"] == r["predicted"] for r in rows) / len(rows) if rows else None


def balanced(rows: list[dict[str, Any]]) -> float | None:
    """Mean recall over the labels that occur: an always-"bad" system gets 1/3,
    while its plain accuracy would look good because most pairs are bad."""
    recalls = [v for v in recall_by_label(rows).values() if v is not None]
    return sum(recalls) / len(recalls) if recalls else None


def interval(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    """95% Wilson interval of the accuracy: with 18 pairs, 72% means "somewhere
    between about 50% and 88%", and the report should say so."""
    n = len(rows)
    if not n:
        return None
    p, z = accuracy(rows), 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - spread), min(1.0, centre + spread)


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
    # The split is per pair, and some vacancies have one CV in tune and the
    # other in holdout. Those holdout pairs share the vacancy with a tuned one;
    # the clean ones are on vacancies the tuning never saw.
    tuned_vacancies = {r["vacancy"] for r in splits["tune"]}
    splits["clean holdout"] = [r for r in splits["holdout"] if r["vacancy"] not in tuned_vacancies]
    return {
        "pairs": len(rows),
        "counts": {name: len(part) for name, part in splits.items()},
        "correct": {
            name: sum(r["expected"] == r["predicted"] for r in part)
            for name, part in splits.items()
        },
        "accuracy": {name: accuracy(part) for name, part in splits.items()},
        "interval": {name: interval(part) for name, part in splits.items()},
        "balanced_accuracy": {name: balanced(part) for name, part in splits.items()},
        "recall": recall_by_label(rows),
        "recall_by_split": {name: recall_by_label(part) for name, part in splits.items()},
        "confusion": confusion(rows),
        "misses": [r for r in rows if r["expected"] != r["predicted"]],
        "rows": rows,
    }
