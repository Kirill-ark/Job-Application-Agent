"""Public demo protection: run limits, the demo run and erasing old content."""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from job_agent.api.app import create_app, demo_view
from job_agent.api.limits import Limits
from job_agent.db import connect, init_db
from job_agent.db.queries import (
    add_parse,
    create_run,
    get_draft,
    get_run,
    save_draft,
    save_score,
    set_demo_run,
)
from job_agent.retention import (
    CV_FOLDER_PREFIX,
    DELETED,
    fail_interrupted_runs,
    redact_old_runs,
    remove_left_cv_files,
)
from tests.test_api import LETTER, draft_url, fake_agent, start, wait_until_finished

T0 = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


def make_client(db_path: Path, limits: Limits, clock: Clock, agent: Any = fake_agent) -> TestClient:
    app = create_app(
        db_path,
        agent_runner=agent,
        limits=limits,
        clock=clock,
        demo_path=db_path.parent / "demo.json",
    )
    return TestClient(app)


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

    async def slow_agent(goal: str, run_id: int, conn: sqlite3.Connection, cv_path: Path) -> None:
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


@pytest.mark.parametrize(
    ("trusted_proxies", "forwarded", "status"),
    [
        (0, "203.0.113.7", 429),  # no proxy: the header is ignored, same IP as before
        (1, "203.0.113.7", 201),  # the proxy says it is another visitor
        # A visitor who writes the header themselves: the proxy adds the real IP
        # on the right, and only that one counts.
        (1, "198.51.100.1, testclient", 429),
    ],
)
def test_forwarded_ip_only_from_a_trusted_proxy(
    tmp_path: Path, clock: Clock, trusted_proxies: int, forwarded: str, status: int
) -> None:
    limits = Limits(runs_per_ip_per_day=1, runs_per_day=100, salt="s")
    app = create_app(
        tmp_path / "t.db",
        agent_runner=fake_agent,
        limits=limits,
        clock=clock,
        trusted_proxies=trusted_proxies,
    )
    with TestClient(app) as client:
        first = {"X-Forwarded-For": "testclient"}  # what the proxy sends for this visitor
        run_id, auth = start(client, headers=first)
        wait_until_finished(client, run_id, auth)
        response = client.post(
            "/api/runs", json={"goal": "Find ML jobs"}, headers={"X-Forwarded-For": forwarded}
        )
        assert response.status_code == status


# The demo run


def test_no_demo_run_yet(tmp_path: Path, clock: Clock) -> None:
    with make_client(tmp_path / "t.db", Limits(salt="s"), clock) as client:
        assert client.get("/api/demo").status_code == 404


def test_demo_run_is_open_to_everyone(tmp_path: Path, clock: Clock) -> None:
    with make_client(tmp_path / "t.db", Limits(salt="s"), clock) as client:
        run_id, auth = start(client)
        wait_until_finished(client, run_id, auth)
        conn = connect(tmp_path / "t.db")
        (tmp_path / "demo.json").write_text(json.dumps(demo_view(conn, run_id)))  # as demo.py does
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
    job_id = get_draft(conn, draft_id)["vacancy_id"]
    for cv_hash, created_at in [("old", T0), ("fresh", T0 + timedelta(hours=24))]:
        save_score(conn, score_row(job_id, cv_hash, created_at.isoformat()))

    assert redact_old_runs(conn, T0 + timedelta(hours=23)) == 0  # not yet
    assert redact_old_runs(conn, T0 + timedelta(hours=25)) == 1

    run = get_run(conn, run_id)
    assert (run["goal"], run["trace"], run["answer"], run["ip_hash"]) == (DELETED, [], None, None)
    draft = get_draft(conn, draft_id)
    assert (draft["text"], draft["cited_facts"], draft["edited_text"]) == (DELETED, {}, None)
    assert draft["status"] == "approved"  # what happened is still known
    assert conn.execute("SELECT * FROM approvals").fetchall() == log_before
    kept = [row[0] for row in conn.execute("SELECT cv_hash FROM scores")]
    assert kept == ["fresh"]  # the old score is gone, the one from the last day stays
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


def test_a_stopped_run_frees_the_agent(tmp_path: Path, clock: Clock) -> None:
    release = asyncio.Event()

    async def slow_agent(goal: str, run_id: int, conn: sqlite3.Connection, cv_path: Path) -> None:
        await release.wait()

    limits = Limits(runs_per_ip_per_day=100, runs_per_day=100, salt="s")
    with make_client(tmp_path / "t.db", limits, clock, slow_agent) as client:
        run_id, auth = start(client)
        assert refused(client)["reason"] == "busy"
        client.post(f"/api/runs/{run_id}/cancel", headers=auth)
        start(client)  # no need to wait 10 minutes
        client.portal.call(release.set)


def score_row(job_id: str, cv_hash: str, created_at: str) -> dict[str, Any]:
    return {
        "vacancy_id": job_id,
        "cv_hash": cv_hash,
        "score": 50,
        "components": "{}",
        "matched": "[]",
        "gaps": "[]",
        "reasons": "",
        "created_at": created_at,
    }


def test_a_draft_no_trace_shows_is_erased_too(
    old_run: tuple[sqlite3.Connection, int, int],
) -> None:
    # A run stopped while a letter was written: the draft exists, the trace has no step.
    conn, _, draft_id = old_run
    job_id = get_draft(conn, draft_id)["vacancy_id"]
    orphan = save_draft(
        conn,
        {
            "vacancy_id": job_id,
            "cv_hash": "h",
            "text": LETTER,
            "cited_facts": {},
            "issues": [],
            "created_at": T0.isoformat(),
        },
    )
    redact_old_runs(conn, T0 + timedelta(hours=25))
    assert get_draft(conn, orphan)["text"] == DELETED


def test_old_cv_parses_are_forgotten(tmp_path: Path) -> None:
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    add_parse(conn, "ip", T0.isoformat())
    redact_old_runs(conn, T0 + timedelta(hours=25))
    assert conn.execute("SELECT COUNT(*) FROM cv_parses").fetchone()[0] == 0


def test_left_cv_folders_are_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    left = tmp_path / f"{CV_FOLDER_PREFIX}abc"
    left.mkdir()
    (left / "cv.yaml").write_text("name: Someone")
    other = tmp_path / "other-folder"
    other.mkdir()
    remove_left_cv_files()
    assert not left.exists() and other.exists()


# Reading CVs


def test_cv_reads_have_a_daily_cap_for_everyone(tmp_path: Path, clock: Clock) -> None:
    limits = Limits(parses_per_ip_per_day=10, parses_per_day=2, salt="s")
    with make_client(tmp_path / "t.db", limits, clock) as client:
        for ip in ("1.1.1.1", "2.2.2.2"):
            add_parse(connect(tmp_path / "t.db"), limits.hash_ip(ip), T0.isoformat())
        refused = client.post("/api/cv/parse", json={"text": "x" * 60})
    assert refused.status_code == 429
    assert refused.json()["detail"]["reason"] == "parse_daily_limit"


# A run that takes too long


def test_a_run_that_takes_too_long_is_stopped(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("job_agent.api.app.MAX_RUN_TIME", timedelta(seconds=0.2))

    async def hanging_agent(goal: str, run_id: int, conn: sqlite3.Connection, cv: Path) -> None:
        await asyncio.sleep(30)

    with make_client(tmp_path / "t.db", Limits(salt="s"), clock, hanging_agent) as client:
        run_id, auth = start(client)
        run = wait_until_finished(client, run_id, auth)
    assert run["status"] == "failed"
    assert run["error"] == "The run took too long and was stopped"


def test_an_ipv6_user_counts_per_network() -> None:
    limits = Limits(salt="s")
    same_network = limits.hash_ip("2001:db8:1:2::1") == limits.hash_ip("2001:db8:1:2:ffff::9")
    other_network = limits.hash_ip("2001:db8:1:2::1") == limits.hash_ip("2001:db8:1:3::1")
    assert same_network and not other_network
    assert limits.hash_ip("203.0.113.7") != limits.hash_ip("203.0.113.8")  # IPv4 per address


def test_a_second_forwarded_header_line_counts(tmp_path: Path, clock: Clock) -> None:
    # A proxy may add its own X-Forwarded-For line instead of extending the visitor's.
    from starlette.requests import Request

    from job_agent.api.app import client_ip

    scope = {
        "type": "http",
        "headers": [
            (b"x-forwarded-for", b"198.51.100.1"),  # written by the visitor
            (b"x-forwarded-for", b"203.0.113.7"),  # added by the proxy
        ],
        "client": ("10.0.0.1", 1234),
    }
    assert client_ip(Request(scope), trusted_proxies=1) == "203.0.113.7"
