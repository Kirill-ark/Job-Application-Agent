"""score_match: how well one vacancy fits the CV, from 0 to 100.

Hybrid by design:
- The score is computed by plain code from four components, so it is stable,
  testable and explainable.
- The LLM only writes a short explanation of that score. If the LLM is not
  available, a template writes it instead, so scoring never fails.

The weights and thresholds are starting values; the evals in step 10 tune them
against hand-labeled vacancies.
"""

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from job_agent.db.queries import get_score, get_vacancy, save_score
from job_agent.llm import Complete, LLMError
from job_agent.skills import Taxonomy

# Change this when the scoring logic changes, so cached scores are not reused.
SCORING_VERSION = "1"

WEIGHTS = {"skills": 0.6, "level": 0.2, "german": 0.1, "location": 0.1}
GOOD_FROM, MAYBE_FROM = 65, 40

TITLE_SKILL_WEIGHT = 2  # a skill in the title counts double
UNVERIFIED_CREDIT = 0.5  # skill only listed in other_skills
NO_INFORMATION = 0.5  # used when a component cannot be judged

LANGUAGE_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2", "native"]

# Explicit German requirements, strongest first.
GERMAN_REQUIREMENTS = [
    (
        "C1",
        r"(fließend\w*|fliessend\w*|verhandlungssicher\w*|sehr gut\w*|exzellent\w*|muttersprach\w*)"
        r"\s+(deutsch|german)|deutsch\w*\s*\(?(c1|c2)|german\s*\(?(c1|c2)"
        r"|fluent\s+(in\s+)?german|native\s+german|german\s+native",
    ),
    (
        "B2",
        r"gut\w*\s+deutsch|good\s+(command\s+of\s+)?german|deutsch\w*\s*\(?b[12]"
        r"|german\s*\(?b[12]|german\s+(is\s+)?(required|a must)",
    ),
]
GERMAN_WORDS = {"und", "der", "die", "das", "mit", "für", "wir", "sie", "ist", "du", "bei", "auf"}
ENGLISH_WORDS = {"and", "the", "with", "for", "we", "you", "is", "our", "are", "of", "to", "in"}

REASONS_SYSTEM_PROMPT = (
    "You explain a job match score to a student who is looking for a job. "
    "Write 2-3 short sentences in English. Mention the strongest matches and the most "
    "important gaps. Use only the facts in the input. Do not change or restate the "
    "score as a different number, and do not invent skills or requirements."
)


class MatchError(Exception):
    """The vacancy is not known."""


def cv_hash(cv: dict[str, Any]) -> str:
    """Fingerprint of the CV (and of the scoring logic) for the score cache."""
    payload = json.dumps({"cv": cv, "version": SCORING_VERSION}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# Components


def find_skills(vacancy: dict[str, Any], taxonomy: Taxonomy) -> dict[str, int]:
    """Skills from the vocabulary that the vacancy mentions, with their weight."""
    body = " ".join([*vacancy["tags"], vacancy["description"]])
    found: dict[str, int] = {}
    for skill in taxonomy.skills.values():
        names = sorted([skill.name, *skill.aliases], key=len, reverse=True)
        pattern = r"(?<![\w+#])(" + "|".join(re.escape(n) for n in names) + r")(?![\w+#])"
        if re.search(pattern, vacancy["title"], re.IGNORECASE):
            found[skill.name] = TITLE_SKILL_WEIGHT
        elif re.search(pattern, body, re.IGNORECASE):
            found[skill.name] = 1
    return found


def skills_component(
    wanted: dict[str, int], cv_skills: dict[str, dict[str, Any]]
) -> tuple[float, list[dict[str, Any]], list[str]]:
    """Weighted share of the vacancy's skills that the CV has. Returns (value, matched, gaps)."""
    if not wanted:
        return NO_INFORMATION, [], ["The vacancy names no skills we can compare"]
    earned = 0.0
    matched, gaps = [], []
    for skill, weight in sorted(wanted.items(), key=lambda item: (-item[1], item[0])):
        have = cv_skills.get(skill)
        if have and have["verified"]:
            earned += weight
            matched.append({"skill": skill, "evidence": have["evidence"]})
        elif have:
            earned += weight * UNVERIFIED_CREDIT
            gaps.append(f"{skill} (listed in the CV, but not used in a project or job)")
        else:
            gaps.append(skill)
    return earned / sum(wanted.values()), matched, gaps


def level_component(vacancy_level: str, looking_for: list[str]) -> float:
    if vacancy_level in looking_for:
        return 1.0
    return {"entry": 0.6, "unknown": NO_INFORMATION, "experienced": 0.0}.get(vacancy_level, 0.3)


def is_mostly_german(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return sum(w in GERMAN_WORDS for w in words) > sum(w in ENGLISH_WORDS for w in words)


def german_required(description: str) -> str | None:
    """The German level the vacancy asks for, or None if it does not need German."""
    for level, pattern in GERMAN_REQUIREMENTS:
        if re.search(pattern, description, re.IGNORECASE):
            return level
    if is_mostly_german(description):
        return "B2"  # an ad written in German expects at least good German
    return None


def german_component(description: str, languages: list[dict[str, str]]) -> tuple[float, str | None]:
    """Returns (value, gap or None)."""
    required = german_required(description)
    if required is None:
        return 1.0, None
    have = next((s["level"] for s in languages if s["language"].lower() == "german"), None)
    have_rank = LANGUAGE_LEVELS.index(have) if have else -1
    missing = LANGUAGE_LEVELS.index(required) - have_rank
    if missing <= 0:
        return 1.0, None
    gap = f"German {required} expected, the CV has {have or 'no German'}"
    return (0.5 if missing == 1 else 0.0), gap


def location_component(vacancy: dict[str, Any], cv_location: str) -> float:
    if vacancy["remote"]:
        return 1.0
    if not vacancy["location"]:
        return NO_INFORMATION
    return 1.0 if cv_location.lower() in vacancy["location"].lower() else 0.0


def label(score: int) -> str:
    if score >= GOOD_FROM:
        return "good"
    return "maybe" if score >= MAYBE_FROM else "bad"


# Reasons


def template_reasons(score: int, matched: list[dict[str, Any]], gaps: list[str]) -> str:
    """Explanation written without the LLM."""
    parts = [f"{label(score).capitalize()} match ({score}/100)."]
    if matched:
        parts.append("Matches: " + ", ".join(m["skill"] for m in matched[:5]) + ".")
    if gaps:
        parts.append("Gaps: " + "; ".join(gaps[:4]) + ".")
    return " ".join(parts)


def write_reasons(
    result: dict[str, Any], vacancy: dict[str, Any], complete: Complete | None
) -> str:
    fallback = template_reasons(result["score"], result["matched"], result["gaps"])
    if complete is None:
        return fallback
    facts = {
        "vacancy": {
            "title": vacancy["title"],
            "company": vacancy["company"],
            "level": vacancy["level"],
            "location": vacancy["location"],
        },
        "score": result["score"],
        "label": label(result["score"]),
        "components": result["components"],
        "matched_skills": result["matched"],
        "gaps": result["gaps"],
    }
    try:
        return complete(REASONS_SYSTEM_PROMPT, json.dumps(facts, ensure_ascii=False))
    except LLMError:
        return fallback


# The tool


def score_match(
    conn: sqlite3.Connection,
    job_id: str,
    cv: dict[str, Any],
    taxonomy: Taxonomy,
    complete: Complete | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Score one cached vacancy against the CV returned by read_cv.

    Results are cached per vacancy and CV, so asking twice does not call the LLM twice.
    """
    vacancy = get_vacancy(conn, job_id)
    if vacancy is None:
        raise MatchError(f"Unknown vacancy '{job_id}'. Use an id returned by search_jobs.")
    fingerprint = cv_hash(cv)

    cached = get_score(conn, job_id, fingerprint)
    if cached is None:
        skills, matched, gaps = skills_component(find_skills(vacancy, taxonomy), cv["skills"])
        german, german_gap = german_component(vacancy["description"], cv["languages"])
        if german_gap:
            gaps.append(german_gap)
        if vacancy["level"] == "experienced" and "experienced" not in cv["looking_for"]:
            gaps.append("Requires professional experience")
        components = {
            "skills": round(skills, 2),
            "level": level_component(vacancy["level"], cv["looking_for"]),
            "german": german,
            "location": location_component(vacancy, cv["location"]),
        }
        score = round(100 * sum(WEIGHTS[name] * value for name, value in components.items()))
        cached = {
            "vacancy_id": job_id,
            "cv_hash": fingerprint,
            "score": score,
            "components": components,
            "matched": matched,
            "gaps": gaps,
            "created_at": (now or datetime.now(UTC)).isoformat(),
        }
        cached["reasons"] = write_reasons(cached, vacancy, complete)
        save_score(conn, cached)

    return {
        "job_id": job_id,
        "title": vacancy["title"],
        "company": vacancy["company"],
        "score": cached["score"],
        "label": label(cached["score"]),
        "components": cached["components"],
        "matched": cached["matched"],
        "gaps": cached["gaps"],
        "reasons": cached["reasons"],
    }
