"""SQL queries. Each function takes an open connection and plain data.

Queries for scores, drafts, approvals and runs are added in the steps
that need them.
"""

import json
import sqlite3
from typing import Any


def upsert_vacancy(conn: sqlite3.Connection, vacancy: dict[str, Any]) -> None:
    """Insert a vacancy, or update it if one with the same id is already cached."""
    with conn:
        conn.execute(
            """
            INSERT INTO vacancies (id, title, company, location, remote, url, description,
                                   tags, job_types, posted_at, fetched_at)
            VALUES (:id, :title, :company, :location, :remote, :url, :description,
                    :tags, :job_types, :posted_at, :fetched_at)
            ON CONFLICT (id) DO UPDATE SET
                title = excluded.title,
                company = excluded.company,
                location = excluded.location,
                remote = excluded.remote,
                url = excluded.url,
                description = excluded.description,
                tags = excluded.tags,
                job_types = excluded.job_types,
                posted_at = excluded.posted_at,
                fetched_at = excluded.fetched_at
            """,
            {
                **vacancy,
                "remote": int(vacancy["remote"]),
                "tags": json.dumps(vacancy["tags"]),
                "job_types": json.dumps(vacancy["job_types"]),
            },
        )


def get_vacancy(conn: sqlite3.Connection, vacancy_id: str) -> dict[str, Any] | None:
    """Return one cached vacancy, or None if it is not in the cache."""
    row = conn.execute("SELECT * FROM vacancies WHERE id = ?", (vacancy_id,)).fetchone()
    if row is None:
        return None
    vacancy = dict(row)
    vacancy["remote"] = bool(vacancy["remote"])
    vacancy["tags"] = json.loads(vacancy["tags"])
    vacancy["job_types"] = json.loads(vacancy["job_types"])
    return vacancy
