"""SQL queries. Each function takes an open connection and plain data.

Queries for runs are added in the step that needs them.
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


def save_draft(conn: sqlite3.Connection, draft: dict[str, Any]) -> int:
    """Store a new cover letter draft and return its id. Older drafts are kept."""
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO drafts (vacancy_id, cv_hash, text, cited_facts, issues,
                                created_at, updated_at)
            VALUES (:vacancy_id, :cv_hash, :text, :cited_facts, :issues,
                    :created_at, :created_at)
            """,
            {
                **draft,
                "cited_facts": json.dumps(draft["cited_facts"], ensure_ascii=False),
                "issues": json.dumps(draft["issues"], ensure_ascii=False),
            },
        )
    return cursor.lastrowid


def get_draft(conn: sqlite3.Connection, draft_id: int) -> dict[str, Any] | None:
    """Return one draft, or None if there is no draft with this id."""
    row = conn.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone()
    if row is None:
        return None
    draft = dict(row)
    for column in ("cited_facts", "issues"):
        draft[column] = json.loads(draft[column])
    return draft


def set_draft_status(
    conn: sqlite3.Connection, draft_id: int, status: str, now: str, edited_text: str | None = None
) -> None:
    """Change a draft's status; keep the earlier edit unless a new one is given."""
    conn.execute(
        """
        UPDATE drafts SET status = ?, edited_text = COALESCE(?, edited_text), updated_at = ?
        WHERE id = ?
        """,
        (status, edited_text, now, draft_id),
    )


def add_approval_event(conn: sqlite3.Connection, event: dict[str, Any]) -> int:
    """Append one row to the approval log and return its id."""
    cursor = conn.execute(
        """
        INSERT INTO approvals (request_id, draft_id, action, event, comment, created_at)
        VALUES (:request_id, :draft_id, :action, :event, :comment, :created_at)
        """,
        {"request_id": None, "comment": None, **event},
    )
    return cursor.lastrowid


# Every request with its decision, if there is one.
_REQUESTS = """
    SELECT r.id, r.draft_id, r.action, r.event, r.comment, r.created_at,
           d.event AS decision, d.comment AS decision_comment, d.created_at AS decided_at
    FROM approvals r
    LEFT JOIN approvals d ON d.request_id = r.id
    WHERE r.request_id IS NULL
"""


def _request_from_row(row: sqlite3.Row) -> dict[str, Any]:
    request = dict(row)
    if request["event"] == "refused":
        request["status"] = "refused"
    else:
        request["status"] = request["decision"] or "pending"
    return request


def get_approval_request(conn: sqlite3.Connection, request_id: int) -> dict[str, Any] | None:
    """One request with its current status: pending, approved, rejected or refused."""
    row = conn.execute(_REQUESTS + " AND r.id = ?", (request_id,)).fetchone()
    return _request_from_row(row) if row else None


def approval_requests(
    conn: sqlite3.Connection, draft_id: int | None = None
) -> list[dict[str, Any]]:
    """All requests, oldest first, optionally for one draft."""
    if draft_id is None:
        rows = conn.execute(_REQUESTS + " ORDER BY r.id").fetchall()
    else:
        rows = conn.execute(_REQUESTS + " AND r.draft_id = ? ORDER BY r.id", (draft_id,)).fetchall()
    return [_request_from_row(row) for row in rows]
