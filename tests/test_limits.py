"""Public demo protection: run limits, the demo run and erasing old content."""

import asyncio
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from job_agent.api.app import create_app
from job_agent.api.limits import Limits
from job_agent.db import connect, init_db
from job_agent.db.queries import create_run, get_draft, get_run, set_demo_run
from job_agent.retention import DELETED, fail_interrupted_runs, redact_old_runs
from tests.test_api import draft_url, fake_agent, start, wait_until_finished

T0 = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


def make_client(db_path: Path, limits: Limits, clock: Clock, agent: Any = fake_agent) -> TestClient:
    return TestClient(create_app(db_path, agent_runner=agent, limits=limits, clock=clock))


@pytest.fixture
def clock() -> Clock:
    return Clock()


def refused(client: TestClient) -> dict[str, str]:
    response = client.post("/api/runs", json={"goal": "Find ML jobs"})
    assert response.status_code == 429
    return response.json()["detail"]


# Limits


def test_runs_per_ip(tmp_path: Path, clock: Clock) -> None:
    limits = Limits(runs_per_ip_per_day=2, runs_per_day=100, salt="s")
    with make_client(tmp_path / "t.db", limits, clock) as client:
        for _ in range(2):
            run_id, auth = start(client)
            wait_until_finished(client, run_id, auth)
        assert refused(client)["reason"] == "ip_limit"
        clock.now += timedelta(hours=25)  # a day later the IP may run again
        start(client)


def test_runs_per_day_for_everyone(tmp_path: Path, clock: Clock) -> None:
    limits = Limits(runs_per_ip_per_day=100, runs_per_day=1, salt="s")
    with make_client(tmp_path / "t.db", limits, clock) as client:
        run_id, auth = start(client)
        wait_until_finished(client, run_id, auth)
        detail = refused(client)
    assert detail["reason"] == "daily_limit"
    assert "limit for today" in detail["message"]


def test_one_run_at_a_time(tmp_path: Path, clock: Clock) -> None:
    release = asyncio.Event()

    async def slow_agent(goal: str, run_id: int, conn: sqlite3.Connection) -> None:
        await release.wait()

    limits = Limits(runs_per_ip_per_day=100, runs_per_day=100, salt="s")
    with make_client(tmp_path / "t.db", limits, clock, slow_agent) as client:
        start(client)
        assert refused(client)["reason"] == "busy"
        clock.now += timedelta(minutes=11)  # a run stuck that long no longer blocks
        start(client)
        client.portal.call(release.set)


def test_ip_is_stored_only_as_a_hash(tmp_path: Path, clock: Clock) -> None:
    limits = Limits(salt="secret")
    with make_client(tmp_path / "t.db", limits, clock) as client:
        run_id, _ = start(client)
    conn = connect(tmp_path / "t.db")
    stored = get_run(conn, run_id)["ip_hash"]
    conn.close()
    assert stored == limits.hash_ip("testclient")
    assert "testclient" not in stored
    assert Limits(salt="other").hash_ip("testclient") != stored


@pytest.mark.parametrize(("trust_proxy", "status"), [(False, 429), (True, 201)])
def test_forwarded_ip_only_behind_a_trusted_proxy(
    tmp_path: Path, clock: Clock, trust_proxy: bool, status: int
) -> None:
    limits = Limits(runs_per_ip_per_day=1, runs_per_day=100, salt="s")
    app = create_app(
        tmp_path / "t.db",
        agent_runner=fake_agent,
        limits=limits,
        clock=clock,
        trust_proxy=trust_proxy,
    )
    with TestClient(app) as client:
        run_id, auth = start(client)
        wait_until_finished(client, run_id, auth)
        # Another visitor behind the proxy. Without trust_proxy the header is
        # ignored (anyone could fake it), so this counts as the same IP.
        other = {"X-Forwarded-For": "203.0.113.7"}
        assert (
            client.post("/api/runs", json={"goal": "Find ML jobs"}, headers=other).status_code
            == status
        )


# The demo run


def test_no_demo_run_yet(tmp_path: Path, clock: Clock) -> None:
    with make_client(tmp_path / "t.db", Limits(salt="s"), clock) as client:
        assert client.get("/api/demo").status_code == 404


def test_demo_run_is_open_to_everyone(tmp_path: Path, clock: Clock) -> None:
    with make_client(tmp_path / "t.db", Limits(salt="s"), clock) as client:
        run_id, auth = start(client)
        wait_until_finished(client, run_id, auth)
        conn = connect(tmp_path / "t.db")
        set_demo_run(conn, run_id)
        conn.close()
        demo = client.get("/api/demo").json()  # no token needed
    assert demo["run_id"] == run_id
    assert len(demo["trace"]) == 4
    draft_id = demo["drafts"][0]["draft_id"]
    assert demo["letters"][str(draft_id)]["plain_text"].startswith("Dear team")


def test_demo_runs_do_not_count_against_limits(tmp_path: Path, clock: Clock) -> None:
    limits = Limits(runs_per_ip_per_day=1, runs_per_day=1, salt="s")
    with make_client(tmp_path / "t.db", limits, clock) as client:
        run_id, auth = start(client)
        wait_until_finished(client, run_id, auth)
        conn = connect(tmp_path / "t.db")
        set_demo_run(conn, run_id)
        conn.close()
        start(client)


# Erasing old content


@pytest.fixture
def old_run(tmp_path: Path, clock: Clock) -> Iterator[tuple[sqlite3.Connection, int, int]]:
    with make_client(tmp_path / "t.db", Limits(salt="s"), clock) as client:
        run_id, auth = start(client)
        draft_id = wait_until_finished(client, run_id, auth)["drafts"][0]["draft_id"]
        client.post(
            draft_url(run_id, draft_id, "decision"), headers=auth, json={"decision": "approved"}
        )
    conn = connect(tmp_path / "t.db")
    yield conn, run_id, draft_id
    conn.close()


def test_old_runs_are_erased_but_the_log_stays(
    old_run: tuple[sqlite3.Connection, int, int],
) -> None:
    conn, run_id, draft_id = old_run
    log_before = conn.execute("SELECT * FROM approvals").fetchall()

    assert redact_old_runs(conn, T0 + timedelta(hours=23)) == 0  # not yet
    assert redact_old_runs(conn, T0 + timedelta(hours=25)) == 1

    run = get_run(conn, run_id)
    assert (run["goal"], run["trace"], run["answer"], run["ip_hash"]) == (DELETED, [], None, None)
    draft = get_draft(conn, draft_id)
    assert (draft["text"], draft["cited_facts"], draft["edited_text"]) == (DELETED, {}, None)
    assert draft["status"] == "approved"  # what happened is still known
    assert conn.execute("SELECT * FROM approvals").fetchall() == log_before
    assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 0
    assert redact_old_runs(conn, T0 + timedelta(hours=48)) == 0  # done only once


def test_demo_run_is_kept(old_run: tuple[sqlite3.Connection, int, int]) -> None:
    conn, run_id, draft_id = old_run
    set_demo_run(conn, run_id)
    assert redact_old_runs(conn, T0 + timedelta(days=30)) == 0
    assert get_draft(conn, draft_id)["text"] != DELETED


def test_interrupted_runs_are_failed_on_start(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    run_id = create_run(conn, "goal", T0.isoformat())
    fail_interrupted_runs(conn, T0)
    run = get_run(conn, run_id)
    conn.close()
    assert (run["status"], run["error"]) == ("failed", "Interrupted by a server restart")
