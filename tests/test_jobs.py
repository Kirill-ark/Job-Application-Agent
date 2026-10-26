import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from job_agent.db import connect, init_db
from job_agent.db.queries import get_vacancy
from job_agent.jobs import (
    JobSourceError,
    detect_level,
    fetch_arbeitnow,
    html_to_text,
    normalize_arbeitnow,
    query_terms,
    search_jobs,
)

PAGES = json.loads((Path(__file__).parent / "fixtures" / "arbeitnow_pages.json").read_text())
NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


class FakeArbeitnow:
    """Stands in for the real API. Counts requests; can be switched to failing."""

    def __init__(self, pages: list[dict[str, Any]] = PAGES) -> None:
        self.pages = pages
        self.requests = 0
        self.failing = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if self.failing:
            return httpx.Response(500)
        page = int(request.url.params["page"])
        return httpx.Response(200, json=self.pages[page - 1])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    yield conn
    conn.close()


def ids(results: list[dict[str, Any]]) -> list[str]:
    return [result["id"] for result in results]


# Cleaning and normalizing


def test_html_to_text_strips_tags_and_keeps_words_apart() -> None:
    html = "<p>Du arbeitest mit <strong>Python</strong>.</p><ul><li>Python</li><li>SQL</li></ul>"
    assert html_to_text(html) == "Du arbeitest mit Python. Python SQL"


@pytest.mark.parametrize(
    ("title", "job_types", "level"),
    [
        ("Werkstudent Machine Learning (m/w/d)", [], "student"),
        ("Working Student Data", [], "student"),
        ("Data Analyst", ["hilfstätigkeit / student"], "student"),
        ("Internship - ML Research Engineer", ["Full Time"], "intern"),
        ("Praktikum Data Science", [], "intern"),
        ("Junior Data Analyst", [], "entry"),
        ("Data Analyst", ["berufseinstieg"], "entry"),
        ("Senior ML Engineer", [], "experienced"),
        ("KI-Berater (m/w/d)", ["Unbefristeter arbeitsvertrag", "berufserfahren"], "experienced"),
        ("Backend Engineer", ["Full Time"], "unknown"),
        ("Internal Tools Engineer", [], "unknown"),
    ],
)
def test_detect_level(title: str, job_types: list[str], level: str) -> None:
    assert detect_level(title, job_types) == level


def test_normalize_arbeitnow() -> None:
    vacancy = normalize_arbeitnow(PAGES[0]["data"][0], NOW.isoformat())
    assert vacancy["id"] == "werkstudent-machine-learning-acme-1"
    assert vacancy["company"] == "Acme GmbH"
    assert vacancy["level"] == "student"
    assert vacancy["description"].startswith("Du arbeitest mit Python")
    assert vacancy["posted_at"] == datetime.fromtimestamp(1792800000, UTC).isoformat()
    assert vacancy["fetched_at"] == NOW.isoformat()


# Fetching


def test_fetch_follows_pages_until_there_is_no_next() -> None:
    api = FakeArbeitnow()
    jobs = fetch_arbeitnow(api.client(), pages=5)
    assert len(jobs) == 7
    assert api.requests == 2


def test_fetch_error_is_a_job_source_error() -> None:
    api = FakeArbeitnow()
    api.failing = True
    with pytest.raises(JobSourceError):
        fetch_arbeitnow(api.client())


def test_fetch_unexpected_body_is_a_job_source_error() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="<html>"))
    )
    with pytest.raises(JobSourceError):
        fetch_arbeitnow(client)


# Query parsing


def test_multi_word_synonym_stays_one_term() -> None:
    assert query_terms("werkstudent machine learning") == [
        {"werkstudent", "werkstudentin", "working student"},
        {"ml", "machine learning"},
    ]


def test_unknown_words_are_their_own_terms() -> None:
    assert query_terms("python ai python") == [
        {"ai", "ki", "artificial intelligence", "künstliche intelligenz"},
        {"python"},
    ]


# Search


def test_best_match_comes_first(conn: sqlite3.Connection) -> None:
    results = search_jobs(
        conn, "werkstudent machine learning", "Berlin", client=FakeArbeitnow().client(), now=NOW
    )
    assert ids(results)[0] == "werkstudent-machine-learning-acme-1"
    assert results[0]["relevance"] == 6  # both terms in the title


def test_location_filter_keeps_remote_jobs(conn: sqlite3.Connection) -> None:
    results = search_jobs(
        conn, "machine learning", "Berlin", client=FakeArbeitnow().client(), now=NOW
    )
    assert "senior-machine-learning-engineer-beta-2" not in ids(results)  # Munich
    assert "junior-data-analyst-zeta-6" not in ids(results)  # Hamburg
    assert "backend-engineer-eta-7" in ids(results)  # Berlin, match in description


def test_synonyms_and_word_boundaries(conn: sqlite3.Connection) -> None:
    results = search_jobs(conn, "AI", "Berlin", client=FakeArbeitnow().client(), now=NOW)
    assert set(ids(results)) == {
        "ki-berater-epsilon-5",  # "KI" is a synonym of "AI"
        "ai-research-intern-delta-4",  # remote
    }
    # "email" in the marketing job must not count as "ai"


def test_no_location_means_everywhere(conn: sqlite3.Connection) -> None:
    results = search_jobs(conn, "machine learning", client=FakeArbeitnow().client(), now=NOW)
    assert "senior-machine-learning-engineer-beta-2" in ids(results)


def test_limit_and_short_result(conn: sqlite3.Connection) -> None:
    results = search_jobs(
        conn, "machine learning", limit=2, client=FakeArbeitnow().client(), now=NOW
    )
    assert len(results) == 2
    assert "description" not in results[0]
    assert len(results[0]["snippet"]) <= 300


def test_full_description_stays_in_the_database(conn: sqlite3.Connection) -> None:
    search_jobs(conn, "python", client=FakeArbeitnow().client(), now=NOW)
    vacancy = get_vacancy(conn, "werkstudent-machine-learning-acme-1")
    assert vacancy is not None
    assert "PyTorch" in vacancy["description"]


# Cache


def test_fresh_cache_does_not_call_the_api(conn: sqlite3.Connection) -> None:
    api = FakeArbeitnow()
    search_jobs(conn, "python", client=api.client(), now=NOW)
    search_jobs(conn, "python", client=api.client(), now=NOW + timedelta(minutes=30))
    assert api.requests == 2  # the two pages of the first search only


def test_old_cache_is_refreshed(conn: sqlite3.Connection) -> None:
    api = FakeArbeitnow()
    search_jobs(conn, "python", client=api.client(), now=NOW)
    search_jobs(conn, "python", client=api.client(), now=NOW + timedelta(hours=2))
    assert api.requests == 4


def test_removed_vacancy_is_not_found_after_refresh(conn: sqlite3.Connection) -> None:
    search_jobs(conn, "python", client=FakeArbeitnow().client(), now=NOW)
    without_backend = [PAGES[0], {"data": [], "links": {"next": None}}]
    later = NOW + timedelta(hours=2)
    results = search_jobs(conn, "python", client=FakeArbeitnow(without_backend).client(), now=later)
    assert "backend-engineer-eta-7" not in ids(results)
    assert get_vacancy(conn, "backend-engineer-eta-7") is not None  # kept for old scores


def test_api_down_with_empty_cache_is_an_error(conn: sqlite3.Connection) -> None:
    api = FakeArbeitnow()
    api.failing = True
    with pytest.raises(JobSourceError):
        search_jobs(conn, "python", client=api.client(), now=NOW)


def test_api_down_with_old_cache_uses_the_cache(conn: sqlite3.Connection) -> None:
    api = FakeArbeitnow()
    search_jobs(conn, "python", client=api.client(), now=NOW)
    api.failing = True
    results = search_jobs(conn, "python", client=api.client(), now=NOW + timedelta(hours=2))
    assert "werkstudent-machine-learning-acme-1" in ids(results)


# Level bonus


def test_level_bonus_changes_the_order(conn: sqlite3.Connection) -> None:
    client = FakeArbeitnow().client()
    without = search_jobs(conn, "machine learning", client=client, now=NOW)
    assert ids(without)[0] == "werkstudent-machine-learning-acme-1"  # newer, same relevance
    with_level = search_jobs(conn, "machine learning", level="experienced", client=client, now=NOW)
    assert ids(with_level)[0] == "senior-machine-learning-engineer-beta-2"
    assert with_level[0]["relevance"] == 5  # title match 3 + level bonus 2


def test_level_alone_does_not_bring_in_unrelated_jobs(conn: sqlite3.Connection) -> None:
    results = search_jobs(
        conn,
        "machine learning",
        "Berlin",
        level="student",
        client=FakeArbeitnow().client(),
        now=NOW,
    )
    assert ids(results)[0] == "werkstudent-machine-learning-acme-1"
    assert "werkstudent-marketing-gamma-3" not in ids(results)  # student, but not about ML


def test_unknown_level_is_an_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="level must be one of"):
        search_jobs(conn, "python", level="senior", client=FakeArbeitnow().client(), now=NOW)


def test_level_in_any_case(conn: sqlite3.Connection) -> None:
    # Seen live: the model wrote "Intern", got an error and searched without a city.
    client = FakeArbeitnow().client()
    upper = search_jobs(conn, "machine learning", level="Student", client=client, now=NOW)
    assert ids(upper) == ids(search_jobs(conn, "machine learning", level="student", now=NOW))


@pytest.mark.parametrize(
    "broken",
    [
        {"company_name": None},  # would break NOT NULL for the whole batch
        {"tags": None},  # would break every later search
        {"url": "javascript:alert(1)"},  # would become a clickable link
        {"created_at": None},
    ],
)
def test_a_malformed_job_is_skipped(conn: sqlite3.Connection, broken: dict[str, Any]) -> None:
    pages = json.loads(json.dumps(PAGES))
    pages[0]["data"][0].update(broken)
    pages[0]["data"].append("not a job at all")
    results = search_jobs(conn, "machine learning", client=FakeArbeitnow(pages).client(), now=NOW)
    assert "werkstudent-machine-learning-acme-1" not in ids(results)  # the broken one
    assert results  # the others are found
