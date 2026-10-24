"""The web API and the page.

    POST /api/runs                                 start the agent; returns run_id and a token
    GET  /api/runs/{run_id}                        progress: status, trace, scores, drafts
    GET  /api/runs/{run_id}/drafts/{draft_id}      one letter with its cited facts and issues
    POST /api/runs/{run_id}/drafts/{draft_id}/decision   the human approves or rejects
    POST /api/runs/{run_id}/drafts/{draft_id}/export     the human takes the final text

The agent runs in the background; the page asks for progress every second.

Every run gets a random token that only the browser that started it knows
(the database keeps a hash). Without it the API answers 404, so nobody can
read other people's letters by trying run ids. Drafts are reached through
their run, and a draft belongs to a run when the run's trace created it.

This is the only place that calls `decide` and `export_letter`: approvals
come from a person clicking a button, never from the agent.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from job_agent.agent.loop import run_summary
from job_agent.agent.runner import run_with_server
from job_agent.approvals import ApprovalError, approval_requests, decide, export_letter
from job_agent.approvals import request_approval as add_request
from job_agent.cv import DEFAULT_CV_PATH
from job_agent.db import connect, default_db_path, init_db
from job_agent.db.queries import create_run, finish_run, get_draft, get_run, get_vacancy
from job_agent.letters import strip_citations
from job_agent.llm import make_chat

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
FAILED_RUN = "The agent could not run. Please try again later."

# (goal, run_id, connection) -> runs the agent and fills in the run.
AgentRunner = Callable[[str, int, sqlite3.Connection], Awaitable[Any]]


class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=500)


class Decision(BaseModel):
    decision: Literal["approved", "rejected"]
    comment: str | None = Field(default=None, max_length=1000)
    edited_text: str | None = Field(default=None, max_length=10_000)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def run_draft_ids(run: dict[str, Any]) -> set[int]:
    """Drafts created by this run, according to its trace."""
    return {
        step["result"]["draft_id"]
        for step in run["trace"]
        if step["kind"] == "tool" and step["tool"] == "draft_cover_letter" and step["ok"]
    }


def create_app(
    db_path: Path | None = None,
    cv_path: Path = DEFAULT_CV_PATH,
    agent_runner: AgentRunner | None = None,
) -> FastAPI:
    """Build the app. Tests pass a fake `agent_runner` and a temporary database."""
    db_path = db_path or default_db_path()
    conn = connect(db_path)
    init_db(conn)
    conn.close()

    async def real_agent(goal: str, run_id: int, conn: sqlite3.Connection) -> None:
        await run_with_server(goal, cv_path, db_path, conn, make_chat(), run_id)

    runner = agent_runner or real_agent
    app = FastAPI(title="Job Application Agent")
    app.state.tasks = set()  # keeps background runs alive until they finish
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def database() -> Iterator[sqlite3.Connection]:
        conn = connect(db_path, any_thread=True)
        try:
            yield conn
        finally:
            conn.close()

    Db = Annotated[sqlite3.Connection, Depends(database)]
    Token = Annotated[str, Header(alias="X-Run-Token")]

    def owned_run(conn: sqlite3.Connection, run_id: int, token: str) -> dict[str, Any]:
        run = get_run(conn, run_id)
        # The same 404 for "no such run" and "wrong token": ids reveal nothing.
        if run is None or not run["token_hash"]:
            raise HTTPException(404, "Run not found")
        if not hmac.compare_digest(run["token_hash"], hash_token(token)):
            raise HTTPException(404, "Run not found")
        return run

    def owned_draft(
        conn: sqlite3.Connection, run_id: int, draft_id: int, token: str
    ) -> dict[str, Any]:
        run = owned_run(conn, run_id, token)
        draft = get_draft(conn, draft_id)
        if draft is None or draft_id not in run_draft_ids(run):
            raise HTTPException(404, "Draft not found")
        return draft

    def draft_view(conn: sqlite3.Connection, draft: dict[str, Any]) -> dict[str, Any]:
        vacancy = get_vacancy(conn, draft["vacancy_id"])
        return {
            "draft_id": draft["id"],
            "status": draft["status"],
            "text": draft["text"],  # with [p1.f2] citations, for the hover hints
            "plain_text": strip_citations(draft["text"]),
            "edited_text": draft["edited_text"],
            "cited_facts": draft["cited_facts"],
            "issues": draft["issues"],
            "vacancy": {k: vacancy[k] for k in ("id", "title", "company", "location", "url")},
            "approvals": approval_requests(conn, draft["id"]),
        }

    async def run_in_background(goal: str, run_id: int) -> None:
        conn = connect(db_path)
        try:
            await runner(goal, run_id, conn)
        except Exception:
            # Also a tool server that does not start, or no LLM key.
            logger.exception("Run %s crashed", run_id)
        finally:
            run = get_run(conn, run_id)
            if run and run["status"] == "running":
                finish_run(conn, run_id, "failed", _now(), error=FAILED_RUN)
            conn.close()

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/runs", status_code=201)
    async def start_run(request: RunRequest, conn: Db) -> dict[str, Any]:
        token = secrets.token_urlsafe(24)
        goal = request.goal.strip()
        run_id = create_run(conn, goal, _now(), token_hash=hash_token(token))
        task = asyncio.create_task(run_in_background(goal, run_id))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return {"run_id": run_id, "token": token}

    @app.get("/api/runs/{run_id}")
    def show_run(run_id: int, conn: Db, token: Token) -> dict[str, Any]:
        run = owned_run(conn, run_id, token)
        summary = run_summary(conn, run_id)
        for score in summary["scores"]:
            vacancy = get_vacancy(conn, score["job_id"])
            score["url"] = vacancy["url"] if vacancy else None
        return {**summary, "goal": run["goal"], "trace": run["trace"]}

    @app.get("/api/runs/{run_id}/drafts/{draft_id}")
    def show_draft(run_id: int, draft_id: int, conn: Db, token: Token) -> dict[str, Any]:
        return draft_view(conn, owned_draft(conn, run_id, draft_id, token))

    @app.post("/api/runs/{run_id}/drafts/{draft_id}/decision")
    def decide_draft(
        run_id: int, draft_id: int, body: Decision, conn: Db, token: Token
    ) -> dict[str, Any]:
        owned_draft(conn, run_id, draft_id, token)
        pending = [
            r
            for r in approval_requests(conn, draft_id)
            if r["action"] == "mark_ready" and r["status"] == "pending"
        ]
        if not pending:
            raise HTTPException(409, "This draft has no pending approval request")
        edited = body.edited_text if body.decision == "approved" else None
        try:
            decide(conn, pending[0]["id"], body.decision, body.comment, edited)
        except ApprovalError as error:
            raise HTTPException(409, str(error)) from error
        return draft_view(conn, get_draft(conn, draft_id))

    @app.post("/api/runs/{run_id}/drafts/{draft_id}/export")
    def export_draft(run_id: int, draft_id: int, conn: Db, token: Token) -> dict[str, str]:
        owned_draft(conn, run_id, draft_id, token)
        # The click on "Export" is the person's approval: the log gets both
        # the request and the decision.
        try:
            request = add_request(conn, "export", draft_id)
            decide(conn, request["request_id"], "approved", "Exported by the user in the UI")
            return {"text": export_letter(conn, draft_id)}
        except ApprovalError as error:
            raise HTTPException(409, str(error)) from error

    return app
