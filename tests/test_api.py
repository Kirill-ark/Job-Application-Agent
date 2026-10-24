"""The web API, with a fake agent that fills in the run like the real one."""

import sqlite3
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from job_agent.api.app import FAILED_RUN, create_app
from job_agent.approvals import request_approval
from job_agent.db import connect
from job_agent.db.queries import (
    finish_run,
    save_draft,
    save_trace,
    upsert_vacancies,
)

NOW = "2026-10-25T10:00:00+00:00"
LETTER = "Dear team,\n\nI reached 0.93 recall. [p1.f2]\n\nBest regards, Alex Example"


def vacancy(job_id: str) -> dict[str, Any]:
    return {
        "id": job_id,
        "title": "Working Student ML",
        "company": "Acme GmbH",
        "location": "Berlin",
        "remote": False,
        "url": f"https://example.com/{job_id}",
        "description": "Python",
        "tags": [],
        "job_types": [],
        "level": "student",
        "posted_at": NOW,
        "fetched_at": NOW,
    }


async def fake_agent(goal: str, run_id: int, conn: sqlite3.Connection) -> None:
    job_id = f"job-{run_id}"
    upsert_vacancies(conn, [vacancy(job_id)])
    draft_id = save_draft(
        conn,
        {
            "vacancy_id": job_id,
            "cv_hash": "h",
            "text": LETTER,
            "cited_facts": {"p1.f2": "Reached 0.93 recall on the held-out test set"},
            "issues": [],
            "created_at": NOW,
        },
    )
    request_approval(conn, "mark_ready", draft_id)
    llm = {"kind": "llm", "prompt_tokens": 90, "completion_tokens": 10, "duration_ms": 5}
    head = {"job_id": job_id, "title": "Working Student ML", "company": "Acme GmbH"}
    save_trace(
        conn,
        run_id,
        [
            {**llm, "text": None, "tool_calls": ["score_match"]},
            {
                "kind": "tool",
                "tool": "score_match",
                "arguments": {"job_id": job_id},
                "ok": True,
                "duration_ms": 5,
                "result": {**head, "score": 80, "label": "good", "gaps": []},
            },
            {
                "kind": "tool",
                "tool": "draft_cover_letter",
                "arguments": {"job_id": job_id},
                "ok": True,
                "duration_ms": 5,
                "result": {**head, "draft_id": draft_id, "grounded": True, "approval": "pending"},
            },
            {**llm, "text": "Done", "tool_calls": []},
        ],
    )
    finish_run(conn, run_id, "done", datetime.now(UTC).isoformat(), answer="Done")


async def crashing_agent(goal: str, run_id: int, conn: sqlite3.Connection) -> None:
    raise RuntimeError("the tool server did not start")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def client(db_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(db_path, agent_runner=fake_agent)) as client:
        yield client


def start(client: TestClient, goal: str = "Find ML jobs in Berlin") -> tuple[int, dict[str, str]]:
    response = client.post("/api/runs", json={"goal": goal})
    assert response.status_code == 201
    body = response.json()
    return body["run_id"], {"X-Run-Token": body["token"]}


def wait_until_finished(client: TestClient, run_id: int, auth: dict[str, str]) -> dict[str, Any]:
    for _ in range(100):
        run = client.get(f"/api/runs/{run_id}", headers=auth).json()
        if run["status"] != "running":
            return run
        time.sleep(0.02)
    raise AssertionError("the run did not finish")


def draft_url(run_id: int, draft_id: int, action: str = "") -> str:
    return f"/api/runs/{run_id}/drafts/{draft_id}" + (f"/{action}" if action else "")


# Page and runs


def test_page_and_static_files(client: TestClient) -> None:
    assert "Job Application Agent" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok"}


def test_run_shows_scores_and_drafts(client: TestClient) -> None:
    run_id, auth = start(client)
    run = wait_until_finished(client, run_id, auth)
    assert run["status"] == "done"
    assert run["goal"] == "Find ML jobs in Berlin"
    assert run["answer"] == "Done"
    assert run["scores"][0]["url"] == f"https://example.com/job-{run_id}"
    assert run["drafts"][0]["approval"] == "pending"
    assert len(run["trace"]) == 4


def test_goal_is_validated(client: TestClient) -> None:
    assert client.post("/api/runs", json={"goal": "x"}).status_code == 422
    assert client.post("/api/runs", json={"goal": "x" * 501}).status_code == 422


def test_run_needs_its_token(client: TestClient) -> None:
    run_id, _ = start(client)
    assert client.get(f"/api/runs/{run_id}").status_code == 422  # no token at all
    wrong = {"X-Run-Token": "guessed"}
    assert client.get(f"/api/runs/{run_id}", headers=wrong).status_code == 404
    assert client.get("/api/runs/999", headers=wrong).status_code == 404  # same answer


def test_other_runs_drafts_are_hidden(client: TestClient) -> None:
    first, first_auth = start(client)
    second, second_auth = start(client)
    wait_until_finished(client, first, first_auth)
    other_draft = wait_until_finished(client, second, second_auth)["drafts"][0]["draft_id"]
    # The first run's token, the second run's draft: not found, either way round.
    assert client.get(draft_url(first, other_draft), headers=first_auth).status_code == 404
    assert client.get(draft_url(second, other_draft), headers=first_auth).status_code == 404


def test_crashed_agent_fails_the_run(db_path: Path) -> None:
    with TestClient(create_app(db_path, agent_runner=crashing_agent)) as client:
        run_id, auth = start(client)
        run = wait_until_finished(client, run_id, auth)
    assert run["status"] == "failed"
    assert run["error"] == FAILED_RUN  # no internal details for the visitor


# Letters


@pytest.fixture
def finished(client: TestClient) -> tuple[int, int, dict[str, str]]:
    run_id, auth = start(client)
    draft_id = wait_until_finished(client, run_id, auth)["drafts"][0]["draft_id"]
    return run_id, draft_id, auth


def test_draft_view(client: TestClient, finished: tuple[int, int, dict[str, str]]) -> None:
    run_id, draft_id, auth = finished
    draft = client.get(draft_url(run_id, draft_id), headers=auth).json()
    assert draft["text"] == LETTER
    assert (
        draft["plain_text"] == "Dear team,\n\nI reached 0.93 recall.\n\nBest regards, Alex Example"
    )
    assert draft["cited_facts"] == {"p1.f2": "Reached 0.93 recall on the held-out test set"}
    assert draft["vacancy"]["url"] == f"https://example.com/job-{run_id}"
    assert [r["status"] for r in draft["approvals"]] == ["pending"]


def test_approve_with_edit_then_export(
    client: TestClient, db_path: Path, finished: tuple[int, int, dict[str, str]]
) -> None:
    run_id, draft_id, auth = finished
    assert client.post(draft_url(run_id, draft_id, "export"), headers=auth).status_code == 409

    response = client.post(
        draft_url(run_id, draft_id, "decision"),
        headers=auth,
        json={"decision": "approved", "edited_text": "My final letter."},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert response.json()["edited_text"] == "My final letter."

    export = client.post(draft_url(run_id, draft_id, "export"), headers=auth)
    assert export.json() == {"text": "My final letter."}

    conn = connect(db_path)
    events = [
        (row["action"], row["event"])
        for row in conn.execute("SELECT action, event FROM approvals ORDER BY id")
    ]
    conn.close()
    assert events == [
        ("mark_ready", "requested"),
        ("mark_ready", "approved"),
        ("export", "requested"),
        ("export", "approved"),
    ]


def test_reject_and_no_second_decision(
    client: TestClient, finished: tuple[int, int, dict[str, str]]
) -> None:
    run_id, draft_id, auth = finished
    url = draft_url(run_id, draft_id, "decision")
    rejected = client.post(
        url, headers=auth, json={"decision": "rejected", "comment": "Too generic"}
    )
    assert rejected.json()["status"] == "rejected"
    again = client.post(url, headers=auth, json={"decision": "approved"})
    assert again.status_code == 409


def test_edit_is_ignored_on_reject(
    client: TestClient, finished: tuple[int, int, dict[str, str]]
) -> None:
    run_id, draft_id, auth = finished
    response = client.post(
        draft_url(run_id, draft_id, "decision"),
        headers=auth,
        json={"decision": "rejected", "edited_text": "something"},
    )
    assert response.status_code == 200
    assert response.json()["edited_text"] is None
