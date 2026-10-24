import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from job_agent.db import connect, init_db
from job_agent.db.queries import (
    current_vacancies,
    get_vacancy,
    last_fetched_at,
    upsert_vacancies,
)

NOW = "2026-10-24T10:00:00+00:00"


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    yield conn
    conn.close()


def make_vacancy(**changes: Any) -> dict[str, Any]:
    vacancy = {
        "id": "ml-werkstudent-acme-123",
        "title": "Werkstudent Machine Learning",
        "company": "Acme GmbH",
        "location": "Berlin",
        "remote": False,
        "url": "https://www.arbeitnow.com/jobs/ml-werkstudent-acme-123",
        "description": "Python and PyTorch",
        "tags": ["Python", "Machine Learning"],
        "job_types": ["working student"],
        "level": "student",
        "posted_at": NOW,
        "fetched_at": NOW,
    }
    return {**vacancy, **changes}


def test_init_creates_all_tables(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    assert {row["name"] for row in rows} == {"vacancies", "scores", "drafts", "approvals", "runs"}


def test_init_twice_keeps_data(conn: sqlite3.Connection) -> None:
    upsert_vacancies(conn, [make_vacancy()])
    init_db(conn)
    assert get_vacancy(conn, "ml-werkstudent-acme-123") is not None


def test_connection_settings(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_vacancy_round_trip(conn: sqlite3.Connection) -> None:
    upsert_vacancies(conn, [make_vacancy()])
    assert get_vacancy(conn, "ml-werkstudent-acme-123") == make_vacancy()


def test_upsert_updates_instead_of_duplicating(conn: sqlite3.Connection) -> None:
    upsert_vacancies(conn, [make_vacancy()])
    upsert_vacancies(conn, [make_vacancy(title="Werkstudent AI", remote=True)])
    vacancy = get_vacancy(conn, "ml-werkstudent-acme-123")
    assert vacancy is not None
    assert vacancy["title"] == "Werkstudent AI"
    assert vacancy["remote"] is True
    assert conn.execute("SELECT COUNT(*) FROM vacancies").fetchone()[0] == 1


def test_unknown_vacancy_is_none(conn: sqlite3.Connection) -> None:
    assert get_vacancy(conn, "nope") is None


def test_score_needs_existing_vacancy(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO scores (vacancy_id, cv_hash, score, components, matched, gaps, reasons,"
            " created_at)"
            " VALUES ('nope', 'h', 50, '{}', '[]', '[]', '', ?)",
            (NOW,),
        )


def test_score_must_be_between_0_and_100(conn: sqlite3.Connection) -> None:
    upsert_vacancies(conn, [make_vacancy()])
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO scores (vacancy_id, cv_hash, score, components, matched, gaps, reasons,"
            " created_at)"
            " VALUES ('ml-werkstudent-acme-123', 'h', 150, '{}', '[]', '[]', '', ?)",
            (NOW,),
        )


def test_approval_decision_is_checked(conn: sqlite3.Connection) -> None:
    upsert_vacancies(conn, [make_vacancy()])
    conn.execute(
        "INSERT INTO drafts (vacancy_id, cv_hash, text, cited_facts, created_at, updated_at)"
        " VALUES ('ml-werkstudent-acme-123', 'h', 'Letter', '{}', ?, ?)",
        (NOW, NOW),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO approvals (draft_id, action, decision, decided_at)"
            " VALUES (1, 'export', 'maybe', ?)",
            (NOW,),
        )


def test_current_vacancies_are_from_the_latest_refresh(conn: sqlite3.Connection) -> None:
    later = "2026-10-24T12:00:00+00:00"
    upsert_vacancies(conn, [make_vacancy(), make_vacancy(id="old-job")])
    upsert_vacancies(conn, [make_vacancy(fetched_at=later)])
    assert last_fetched_at(conn) == later
    assert [v["id"] for v in current_vacancies(conn)] == ["ml-werkstudent-acme-123"]


def test_empty_cache_has_no_fetch_time(conn: sqlite3.Connection) -> None:
    assert last_fetched_at(conn) is None
