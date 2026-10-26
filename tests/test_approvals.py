import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from job_agent.approvals import (
    REFUSAL,
    ApprovalError,
    decide,
    export_letter,
    request_approval,
)
from job_agent.db import connect, init_db
from job_agent.db.queries import approval_requests, get_draft, save_draft, upsert_vacancies

NOW = "2026-10-24T12:00:00+00:00"


def pending_requests(conn: sqlite3.Connection) -> list[dict]:
    return [r for r in approval_requests(conn) if r["status"] == "pending"]


LETTER = "Dear team,\n\nI reached 0.93 recall. [p1.f2]\n\nBest regards, Test Person"


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    upsert_vacancies(
        conn,
        [
            {
                "id": "job",
                "title": "Working Student ML",
                "company": "Acme GmbH",
                "location": "Berlin",
                "remote": False,
                "url": "https://example.com/job",
                "description": "Python",
                "tags": [],
                "job_types": [],
                "level": "student",
                "posted_at": NOW,
                "fetched_at": NOW,
            }
        ],
    )
    yield conn
    conn.close()


def make_draft(conn: sqlite3.Connection, issues: list[dict[str, str]] | None = None) -> int:
    return save_draft(
        conn,
        {
            "vacancy_id": "job",
            "cv_hash": "h",
            "text": LETTER,
            "cited_facts": {"p1.f2": "Reached 0.93 recall"},
            "issues": issues or [],
            "created_at": NOW,
        },
    )


def approve_ready(conn: sqlite3.Connection, draft_id: int, **changes: str) -> None:
    request = request_approval(conn, "mark_ready", draft_id)
    decide(conn, request["request_id"], "approved", **changes)


def log_size(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0]


# Requests


def test_request_waits_for_the_human(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    result = request_approval(conn, "mark_ready", draft_id)
    assert result["status"] == "pending"
    assert "Waiting for the user's decision" in result["message"]
    assert get_draft(conn, draft_id)["status"] == "draft"  # nothing changed yet
    assert [r["id"] for r in pending_requests(conn)] == [result["request_id"]]


def test_asking_twice_returns_the_same_request(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    first = request_approval(conn, "mark_ready", draft_id)
    second = request_approval(conn, "mark_ready", draft_id)
    assert second["request_id"] == first["request_id"]
    assert log_size(conn) == 1


@pytest.mark.parametrize("action", ["send", "email", "submit_application"])
def test_sending_is_refused_and_logged(conn: sqlite3.Connection, action: str) -> None:
    draft_id = make_draft(conn)
    result = request_approval(conn, action, draft_id)
    assert result["status"] == "refused"
    assert result["message"] == REFUSAL
    assert pending_requests(conn) == []  # nothing for the human to approve
    row = conn.execute("SELECT action, event FROM approvals").fetchone()
    assert (row["action"], row["event"]) == (action, "refused")


def test_refused_request_cannot_be_approved(conn: sqlite3.Connection) -> None:
    request = request_approval(conn, "send", make_draft(conn))
    with pytest.raises(ApprovalError, match="already refused"):
        decide(conn, request["request_id"], "approved")


def test_unknown_draft(conn: sqlite3.Connection) -> None:
    with pytest.raises(ApprovalError, match="draft_cover_letter"):
        request_approval(conn, "mark_ready", 99)


def test_issues_are_shown_and_logged(conn: sqlite3.Connection) -> None:
    issues = [{"sentence": "I know Django.", "problem": "mentions skills ...: Django"}]
    result = request_approval(conn, "mark_ready", make_draft(conn, issues))
    assert result["issues"] == issues
    comment = conn.execute("SELECT comment FROM approvals").fetchone()[0]
    assert comment == "The draft has 1 unresolved groundedness issue(s)."


# Decisions


def test_approve_mark_ready(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    request = request_approval(conn, "mark_ready", draft_id)
    result = decide(conn, request["request_id"], "approved", comment="Looks good")
    assert result["status"] == "approved"
    assert result["decision_comment"] == "Looks good"
    assert get_draft(conn, draft_id)["status"] == "approved"
    assert pending_requests(conn) == []


def test_reject_mark_ready(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    request = request_approval(conn, "mark_ready", draft_id)
    decide(conn, request["request_id"], "rejected", comment="Too generic")
    assert get_draft(conn, draft_id)["status"] == "rejected"
    with pytest.raises(ApprovalError, match="Write a new draft"):
        request_approval(conn, "mark_ready", draft_id)


def test_cannot_decide_twice(conn: sqlite3.Connection) -> None:
    request = request_approval(conn, "mark_ready", make_draft(conn))
    decide(conn, request["request_id"], "approved")
    with pytest.raises(ApprovalError, match="already approved"):
        decide(conn, request["request_id"], "rejected")


def test_unknown_request_and_decision(conn: sqlite3.Connection) -> None:
    with pytest.raises(ApprovalError, match="Unknown approval request"):
        decide(conn, 42, "approved")
    request = request_approval(conn, "mark_ready", make_draft(conn))
    with pytest.raises(ApprovalError, match="Unknown decision"):
        decide(conn, request["request_id"], "maybe")  # type: ignore[arg-type]


def test_edit_is_saved_with_the_approval(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    approve_ready(conn, draft_id, edited_text="My own final letter.")
    draft = get_draft(conn, draft_id)
    assert draft["edited_text"] == "My own final letter."
    assert draft["text"] == LETTER  # the generated version stays for the history


def test_edit_only_with_approval(conn: sqlite3.Connection) -> None:
    request = request_approval(conn, "mark_ready", make_draft(conn))
    with pytest.raises(ApprovalError, match="edited_text"):
        decide(conn, request["request_id"], "rejected", edited_text="x")
    with pytest.raises(ApprovalError, match="empty"):
        decide(conn, request["request_id"], "approved", edited_text="  ")


# Export


def test_export_needs_an_approved_draft(conn: sqlite3.Connection) -> None:
    with pytest.raises(ApprovalError, match="mark_ready"):
        request_approval(conn, "export", make_draft(conn))


def test_export_needs_its_own_approval(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    approve_ready(conn, draft_id)
    with pytest.raises(ApprovalError, match="not been approved"):
        export_letter(conn, draft_id)
    request = request_approval(conn, "export", draft_id)
    with pytest.raises(ApprovalError, match="not been approved"):
        export_letter(conn, draft_id)  # still pending
    decide(conn, request["request_id"], "approved")
    assert export_letter(conn, draft_id) == (
        "Dear team,\n\nI reached 0.93 recall.\n\nBest regards, Test Person"
    )


def test_export_uses_the_users_edit(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    approve_ready(conn, draft_id, edited_text="My own final letter.")
    request = request_approval(conn, "export", draft_id)
    decide(conn, request["request_id"], "approved")
    assert export_letter(conn, draft_id) == "My own final letter."


def test_rejected_export_gives_nothing(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    approve_ready(conn, draft_id)
    request = request_approval(conn, "export", draft_id)
    decide(conn, request["request_id"], "rejected")
    with pytest.raises(ApprovalError, match="not been approved"):
        export_letter(conn, draft_id)


# The log


def test_log_only_grows(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    approve_ready(conn, draft_id)
    request_approval(conn, "send", draft_id)
    events = [r["event"] for r in conn.execute("SELECT event FROM approvals ORDER BY id")]
    assert events == ["requested", "approved", "refused"]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE approvals SET event = 'approved' WHERE event = 'refused'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM approvals")


def test_database_forbids_a_second_decision(conn: sqlite3.Connection) -> None:
    request = request_approval(conn, "mark_ready", make_draft(conn))
    decide(conn, request["request_id"], "approved")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO approvals (request_id, draft_id, action, event, created_at)"
            " VALUES (?, 1, 'mark_ready', 'rejected', ?)",
            (request["request_id"], NOW),
        )


def test_two_decisions_at_once_are_an_approval_error(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft_id = make_draft(conn)
    request = request_approval(conn, "mark_ready", draft_id)
    decide(conn, request["request_id"], "approved")
    # The second click read the request before the first one was saved.
    import job_agent.approvals as approvals

    real = approvals.get_approval_request
    monkeypatch.setattr(
        approvals, "get_approval_request", lambda c, i: {**real(c, i), "status": "pending"}
    )
    with pytest.raises(ApprovalError, match="already decided"):
        decide(conn, request["request_id"], "rejected")


def test_refused_action_names_are_cut_short(conn: sqlite3.Connection) -> None:
    draft_id = make_draft(conn)
    result = request_approval(conn, "send " + "my whole CV text " * 20, draft_id)
    assert result["status"] == "refused"
    assert len(result["action"]) == 30
