"""SQL queries. Each function takes an open connection and plain data.

Queries for drafts, approvals and runs are added in the steps
that need them.
"""

import json
import sqlite3
from typing import Any


def upsert_vacancies(conn: sqlite3.Connection, vacancies: list[dict[str, Any]]) -> None:
    """Insert vacancies, or update the ones already cached. One transaction for all."""
    with conn:
        conn.executemany(
            """
            INSERT INTO vacancies (id, title, company, location, remote, url, description,
                                   tags, job_types, level, posted_at, fetched_at)
            VALUES (:id, :title, :company, :location, :remote, :url, :description,
                    :tags, :job_types, :level, :posted_at, :fetched_at)
            ON CONFLICT (id) DO UPDATE SET
                title = excluded.title,
                company = excluded.company,
                location = excluded.location,
                remote = excluded.remote,
                url = excluded.url,
                description = excluded.description,
                tags = excluded.tags,
                job_types = excluded.job_types,
                level = excluded.level,
                posted_at = excluded.posted_at,
                fetched_at = excluded.fetched_at
            """,
            [
                {
                    **vacancy,
                    "remote": int(vacancy["remote"]),
                    "tags": json.dumps(vacancy["tags"]),
                    "job_types": json.dumps(vacancy["job_types"]),
                }
                for vacancy in vacancies
            ],
        )


def _vacancy_from_row(row: sqlite3.Row) -> dict[str, Any]:
    vacancy = dict(row)
    vacancy["remote"] = bool(vacancy["remote"])
    vacancy["tags"] = json.loads(vacancy["tags"])
    vacancy["job_types"] = json.loads(vacancy["job_types"])
    return vacancy


def get_vacancy(conn: sqlite3.Connection, vacancy_id: str) -> dict[str, Any] | None:
    """Return one cached vacancy, or None if it is not in the cache."""
    row = conn.execute("SELECT * FROM vacancies WHERE id = ?", (vacancy_id,)).fetchone()
    return _vacancy_from_row(row) if row else None


def last_fetched_at(conn: sqlite3.Connection) -> str | None:
    """Time of the latest refresh, or None if the cache is empty."""
    return conn.execute("SELECT MAX(fetched_at) FROM vacancies").fetchone()[0]


def current_vacancies(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Vacancies from the latest refresh.

    A vacancy that was not in the latest refresh has been taken down, so it is
    skipped. It stays in the table because scores and drafts may refer to it.
    """
    rows = conn.execute(
        "SELECT * FROM vacancies WHERE fetched_at = (SELECT MAX(fetched_at) FROM vacancies)"
    ).fetchall()
    return [_vacancy_from_row(row) for row in rows]


def get_score(conn: sqlite3.Connection, vacancy_id: str, cv_hash: str) -> dict[str, Any] | None:
    """Return the cached match result for this vacancy and CV, or None."""
    row = conn.execute(
        "SELECT * FROM scores WHERE vacancy_id = ? AND cv_hash = ?", (vacancy_id, cv_hash)
    ).fetchone()
    if row is None:
        return None
    score = dict(row)
    for column in ("components", "matched", "gaps"):
        score[column] = json.loads(score[column])
    return score


def save_score(conn: sqlite3.Connection, score: dict[str, Any]) -> None:
    """Store a match result, replacing an older one for the same vacancy and CV."""
    with conn:
        conn.execute(
            """
            INSERT INTO scores (vacancy_id, cv_hash, score, components, matched, gaps,
                                reasons, created_at)
            VALUES (:vacancy_id, :cv_hash, :score, :components, :matched, :gaps,
                    :reasons, :created_at)
            ON CONFLICT (vacancy_id, cv_hash) DO UPDATE SET
                score = excluded.score,
                components = excluded.components,
                matched = excluded.matched,
                gaps = excluded.gaps,
                reasons = excluded.reasons,
                created_at = excluded.created_at
            """,
            {
                **score,
                "components": json.dumps(score["components"]),
                "matched": json.dumps(score["matched"]),
                "gaps": json.dumps(score["gaps"]),
            },
        )
