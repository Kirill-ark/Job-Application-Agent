"""Vacancies from the Arbeitnow job board: fetching, cleaning, caching and search.

The Arbeitnow API has no search parameters: it returns all jobs, newest first,
300 per page. So we download the newest pages into SQLite at most once an hour
and search that cache locally.

Another job source only needs its own `normalize_<source>` function that returns
the same vacancy dict.
"""

import logging
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Any

import httpx

from job_agent.db.queries import current_vacancies, last_fetched_at, upsert_vacancies

log = logging.getLogger(__name__)

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
PAGES_TO_FETCH = 5  # about 1,500 jobs from the last few days
CACHE_TTL = timedelta(hours=1)  # Arbeitnow updates hourly
SNIPPET_LENGTH = 300

# Query words that mean the same thing. A query word matches any word of its group.
SYNONYM_GROUPS = [
    {"werkstudent", "werkstudentin", "working student"},
    {"ai", "ki", "artificial intelligence", "künstliche intelligenz"},
    {"ml", "machine learning"},
    {"internship", "intern", "praktikum", "praktikant"},
    {"junior", "entry level", "berufseinstieg"},
]

# Checked in this order; the first match wins.
LEVEL_PATTERNS = [
    ("student", r"werkstudent|working student|student"),
    ("intern", r"\bintern\b|internship|praktik"),
    ("entry", r"junior|entry|berufseinstieg|einsteiger|graduate|absolvent"),
    ("experienced", r"senior|\blead\b|principal|head of|experienced|berufserfahren|\bmid\b"),
]

LEVELS = ("student", "intern", "entry", "experienced", "unknown")

# How much a match counts, by where it was found.
WEIGHT_TITLE, WEIGHT_TAGS, WEIGHT_DESCRIPTION = 3, 2, 1
# Added when the vacancy has the wanted level. Only for vacancies that match the
# query, so a level alone never brings in an unrelated job.
LEVEL_BONUS = 2


class JobSourceError(Exception):
    """The job board could not be reached or returned something unexpected."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(" ")  # keep "<li>Python</li><li>SQL</li>" from becoming "PythonSQL"


def html_to_text(html: str) -> str:
    """Strip tags and collapse whitespace."""
    extractor = _TextExtractor()
    extractor.feed(html)
    return " ".join("".join(extractor.parts).split())


def detect_level(title: str, job_types: list[str]) -> str:
    """One seniority level from the title, or from the job types if the title says nothing."""
    for text in (title, " ".join(job_types)):
        for level, pattern in LEVEL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return level
    return "unknown"


def normalize_arbeitnow(raw: dict[str, Any], fetched_at: str) -> dict[str, Any]:
    """Turn one Arbeitnow job into our vacancy dict.

    The URL becomes a link on the page, so only https links are accepted
    (a "javascript:" URL would run code when clicked).
    """
    for key in ("slug", "title", "company_name", "url", "description"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"{key} is missing or not text")
    if not isinstance(raw.get("location"), str):  # empty for remote jobs
        raise ValueError("location is not text")
    for key in ("tags", "job_types"):
        if not isinstance(raw.get(key), list):
            raise ValueError(f"{key} is not a list")
    if not raw["url"].startswith("https://"):
        raise ValueError(f"not an https URL: {raw['url']!r}")
    return {
        "id": raw["slug"],
        "title": raw["title"],
        "company": raw["company_name"],
        "location": raw["location"],
        "remote": bool(raw["remote"]),
        "url": raw["url"],
        "description": html_to_text(raw["description"]),
        "tags": raw["tags"],
        "job_types": raw["job_types"],
        "level": detect_level(raw["title"], raw["job_types"]),
        "posted_at": datetime.fromtimestamp(raw["created_at"], UTC).isoformat(),
        "fetched_at": fetched_at,
    }


def fetch_arbeitnow(client: httpx.Client, pages: int = PAGES_TO_FETCH) -> list[dict[str, Any]]:
    """Download the newest pages of raw jobs."""
    jobs: list[dict[str, Any]] = []
    try:
        for page in range(1, pages + 1):
            response = client.get(ARBEITNOW_URL, params={"page": page}, timeout=20)
            response.raise_for_status()
            body = response.json()
            jobs.extend(body["data"])
            if not body["links"]["next"]:
                break
    except (httpx.HTTPError, ValueError, KeyError) as error:
        raise JobSourceError(f"Arbeitnow request failed: {error}") from error
    return jobs


def refresh_cache(conn: sqlite3.Connection, client: httpx.Client, now: datetime) -> int:
    """Download fresh jobs into the cache. Returns how many were stored."""
    fetched_at = now.isoformat()
    vacancies = []
    for raw in fetch_arbeitnow(client):
        try:
            vacancies.append(normalize_arbeitnow(raw, fetched_at))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            # One broken job must not stop the search for all the others.
            slug = raw.get("slug") if isinstance(raw, dict) else None
            log.warning("Skipping a malformed job %r: %s", slug, error)
    upsert_vacancies(conn, vacancies)
    return len(vacancies)


def _ensure_fresh_cache(conn: sqlite3.Connection, client: httpx.Client, now: datetime) -> None:
    last = last_fetched_at(conn)
    if last and datetime.fromisoformat(last) > now - CACHE_TTL:
        return
    try:
        refresh_cache(conn, client, now)
    except (JobSourceError, sqlite3.Error) as error:
        if last is None:
            raise  # nothing cached, nothing to fall back to
        # The old cache is better than no answer.
        log.warning("Using cached vacancies from %s: %s", last, error)


def query_terms(query: str) -> list[set[str]]:
    """Split a query into terms; each term is the set of words that count as a match.

    Multi-word synonyms ("machine learning") are found first so they stay one term.
    """
    rest = query.lower()
    terms: list[set[str]] = []
    for group in SYNONYM_GROUPS:
        for phrase in sorted(group, key=len, reverse=True):
            pattern = rf"\b{re.escape(phrase)}\b"
            if re.search(pattern, rest):
                terms.append(group)
                rest = re.sub(pattern, " ", rest)
                break
    for word in rest.split():
        group = next((g for g in SYNONYM_GROUPS if word in g), {word})
        if group not in terms:
            terms.append(group)
    return terms


def _contains(text: str, words: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) for word in words)


def relevance(vacancy: dict[str, Any], terms: list[set[str]]) -> int:
    """Sum over terms of the best place each term was found: title, tags or description."""
    tags = " ".join(vacancy["tags"])
    total = 0
    for words in terms:
        if _contains(vacancy["title"], words):
            total += WEIGHT_TITLE
        elif _contains(tags, words):
            total += WEIGHT_TAGS
        elif _contains(vacancy["description"], words):
            total += WEIGHT_DESCRIPTION
    return total


def _in_location(vacancy: dict[str, Any], location: str | None) -> bool:
    if not location:
        return True
    return location.lower() in vacancy["location"].lower() or vacancy["remote"]


def search_jobs(
    conn: sqlite3.Connection,
    query: str,
    location: str | None = None,
    limit: int = 10,
    level: str | None = None,
    client: httpx.Client | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Find cached vacancies matching the query, best first.

    The query says what the job is about ("machine learning"); the wanted
    seniority goes into `level` ("student"), which ranks those vacancies higher.
    Remote vacancies count as being in any location. The result is short on
    purpose: the full description stays in the database for score_match.
    """
    # Models write "Intern" as often as "intern" (seen live); case says nothing here.
    level = level.lower() if level is not None else None
    if level is not None and level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}, not {level!r}")
    now = now or datetime.now(UTC)
    if client is None:
        with httpx.Client() as own_client:
            _ensure_fresh_cache(conn, own_client, now)
    else:
        _ensure_fresh_cache(conn, client, now)

    terms = query_terms(query)
    found = []
    for vacancy in current_vacancies(conn):
        score = relevance(vacancy, terms)
        if score > 0 and _in_location(vacancy, location):
            if level and vacancy["level"] == level:
                score += LEVEL_BONUS
            found.append((score, vacancy))
    found.sort(key=lambda pair: (pair[0], pair[1]["posted_at"]), reverse=True)

    return [
        {
            "id": vacancy["id"],
            "title": vacancy["title"],
            "company": vacancy["company"],
            "location": vacancy["location"],
            "remote": vacancy["remote"],
            "level": vacancy["level"],
            "url": vacancy["url"],
            "posted_at": vacancy["posted_at"],
            "relevance": score,
            "snippet": vacancy["description"][:SNIPPET_LENGTH],
        }
        for score, vacancy in found[:limit]
    ]
