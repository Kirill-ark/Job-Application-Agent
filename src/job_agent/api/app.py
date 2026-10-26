"""The web API and the page.

    POST /api/runs                                 start the agent; returns run_id and a token
    GET  /api/runs/{run_id}                        progress: status, trace, scores, drafts
    GET  /api/runs/{run_id}/drafts/{draft_id}      one letter with its cited facts and issues
    POST /api/runs/{run_id}/drafts/{draft_id}/decision   the human approves or rejects
    POST /api/runs/{run_id}/drafts/{draft_id}/export     the human takes the final text
    POST /api/runs/{run_id}/cancel                 the human stops a run that is going
    GET  /api/demo                                 the saved demo run, shown when a limit is hit
    POST /api/cv/parse                             a pasted CV as the agent will see it
    GET  /api/health                               {"status": "ok"}

The agent runs in the background; the page asks for progress every second.

Every run gets a random token that only the browser that started it knows
(the database keeps a hash). Without it the API answers 404, so nobody can
read other people's letters by trying run ids. Drafts are reached through
their run, and a draft belongs to a run when the run's trace created it.

This is the only place that calls `decide` and `export_letter`: approvals
come from a person clicking a button, never from the agent.

Public demo protection lives in limits.py (who may start a run) and
retention.py (visitors' content is erased after 24 hours).
"""

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import sqlite3
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from job_agent import REPO_ROOT
from job_agent.agent.loop import run_summary
from job_agent.agent.runner import run_with_server
from job_agent.api.limits import MAX_RUN_TIME, MESSAGES, Limits
from job_agent.approvals import ApprovalError, approval_requests, decide, export_letter
from job_agent.approvals import request_approval as add_request
from job_agent.cv import CV, DEFAULT_CV_PATH, cv_data
from job_agent.cv_parser import (
    CV_SCHEMA,
    MAX_CV_CHARS,
    CVParseError,
    check_parsed,
    drop_used_skills,
    move_listed_skills,
    parse_cv_text,
)
from job_agent.db import connect, default_db_path, init_db
from job_agent.db.queries import (
    add_parse,
    create_run,
    created_drafts,
    finish_run,
    get_draft,
    get_run,
    get_vacancy,
    save_trace,
)
from job_agent.letters import strip_citations
from job_agent.llm import DEFAULT_LETTER_MODEL, Complete, LLMError, make_chat, make_complete
from job_agent.retention import (
    CV_FOLDER_PREFIX,
    fail_interrupted_runs,
    redact_old_runs,
    remove_left_cv_files,
)
from job_agent.skills import load_taxonomy

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEMO_PATH = REPO_ROOT / "data" / "demo_run.json"
FAILED_RUN = "The agent could not run. Please try again later."
STOPPED = "Stopped by the user"

TOO_LONG = "The run took too long and was stopped"
REDACT_EVERY = 3600  # seconds; old content is also erased on start and on every new run

# Sent with every response. The page uses no inline scripts or styles, so a
# strict policy costs nothing, and it may not be shown inside another site.
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}

# (goal, run_id, connection, CV file) -> runs the agent and fills in the run.
AgentRunner = Callable[[str, int, sqlite3.Connection, Path], Awaitable[Any]]


class RunRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=500)
    cv: dict[str, Any] | None = None  # a confirmed parsed CV; None = the example CV


# A parsed CV is sent back with the run. It goes into every letter prompt, so
# its size is capped like the pasted text (as JSON it is somewhat longer).
MAX_CV_JSON_CHARS = 2 * MAX_CV_CHARS
MAX_BODY_BYTES = 100_000


class ParseRequest(BaseModel):
    text: str = Field(min_length=50, max_length=MAX_CV_CHARS)


class Decision(BaseModel):
    # No free-text comment: the approval log is kept forever, a visitor's words are not.
    decision: Literal["approved", "rejected"]
    edited_text: str | None = Field(default=None, max_length=10_000)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def run_view(conn: sqlite3.Connection, run: dict[str, Any]) -> dict[str, Any]:
    summary = run_summary(conn, run["id"])
    for score in summary["scores"]:
        vacancy = get_vacancy(conn, score["job_id"])
        score["url"] = vacancy["url"] if vacancy else None
    return {
        **summary,
        "goal": run["goal"],
        "trace": run["trace"],
        "started_at": run["started_at"],
        "finished_at": run["finished_at"],
    }


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


def demo_view(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """A run with its letters, as the page replays it (saved by demo.py)."""
    run = get_run(conn, run_id)
    letters = {d: draft_view(conn, get_draft(conn, d)) for d in created_drafts(run["trace"])}
    return {**run_view(conn, run), "letters": letters}


# Helpers of the routes below


def owned_run(conn: sqlite3.Connection, run_id: int, token: str) -> dict[str, Any]:
    """The run, if `token` is its token. The same 404 for "no such run" and
    "wrong token", so trying ids reveals nothing."""
    run = get_run(conn, run_id)
    if run is None or not run["token_hash"]:
        raise HTTPException(404, "Run not found")
    if not hmac.compare_digest(run["token_hash"], hash_token(token)):
        raise HTTPException(404, "Run not found")
    return run


def owned_draft(conn: sqlite3.Connection, run_id: int, draft_id: int, token: str) -> dict[str, Any]:
    """The draft, if the run of `token` created it."""
    run = owned_run(conn, run_id, token)
    draft = get_draft(conn, draft_id)
    if draft is None or draft_id not in created_drafts(run["trace"]):
        raise HTTPException(404, "Draft not found")
    return draft


_proxy_logged: list[bool] = []  # filled after the first request


def client_ip(request: Request, trusted_proxies: int) -> str:
    """The visitor's IP. Behind `trusted_proxies` proxies it is the entry of
    X-Forwarded-For that the farthest of them added; entries left of it were
    sent by the visitor and can be anything."""
    # All header lines: a proxy may add its own line instead of extending the visitor's.
    header = ",".join(request.headers.getlist("x-forwarded-for"))
    forwarded = [ip.strip() for ip in header.split(",")]
    if trusted_proxies and not _proxy_logged:
        # Once, to check TRUSTED_PROXIES after a deploy. No addresses, only the count.
        logger.warning(
            "X-Forwarded-For has %d entries (TRUSTED_PROXIES=%d)", len(forwarded), trusted_proxies
        )
        _proxy_logged.append(True)
    if trusted_proxies and len(forwarded) >= trusted_proxies and forwarded[-1]:
        return forwarded[-trusted_proxies]
    return request.client.host if request.client else "unknown"


def has_body(request: Request) -> bool:
    headers = request.headers
    return headers.get("content-length", "0") != "0" or "transfer-encoding" in headers


async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Middleware: refuse what no page request looks like, add the security headers.

    A request with a body must say it is JSON: a form or a "no-cors" request
    from another site cannot, so other sites cannot start runs from their
    visitors' browsers.
    """
    size = request.headers.get("content-length")
    if size and (not size.isdigit() or int(size) > MAX_BODY_BYTES):
        response: Response = JSONResponse({"detail": "The request is too large"}, 413)
    elif has_body(request) and not request.headers.get("content-type", "").startswith(
        "application/json"
    ):
        response = JSONResponse({"detail": "Send JSON"}, 415)
    else:
        response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response


class LimitBody:
    """ASGI middleware: read at most MAX_BODY_BYTES of a request body.

    The Content-Length check in `guard` is not enough: a chunked request has no
    Content-Length. So the body is read here first, refused once it is too
    large, and then handed to the app as if it came from the network.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body, more = b"", True
        while more:
            message = await receive()
            if message["type"] != "http.request":  # the client went away
                return
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > MAX_BODY_BYTES:
                response = JSONResponse({"detail": "The request is too large"}, 413)
                response.headers.update(SECURITY_HEADERS)
                return await response(scope, receive, send)
        replayed = False

        async def replay() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()  # after the body: wait for a disconnect

        await self.app(scope, replay, send)


@functools.cache
def parse_complete() -> Complete:
    """The LLM that reads pasted CVs; made on first use, so tests need no API key."""
    model = os.environ.get("GROQ_LETTER_MODEL") or DEFAULT_LETTER_MODEL
    return make_complete(4000, CV_SCHEMA, model=model)


def real_parse_llm(system: str, user: str) -> str:
    return parse_complete()(system, user)


def create_app(
    db_path: Path | None = None,
    cv_path: Path = DEFAULT_CV_PATH,
    agent_runner: AgentRunner | None = None,
    limits: Limits | None = None,
    trusted_proxies: int = 0,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    parse_llm: Complete | None = None,
    demo_path: Path = DEMO_PATH,
) -> FastAPI:
    """Build the app. Tests pass a fake `agent_runner`, `parse_llm`, limits, a clock and a
    temporary database.

    `trusted_proxies`: how many proxies in front of the app add themselves to
    X-Forwarded-For (3 on Render; see client_ip). 0 means no proxy: the IP of
    the connection is used.
    """
    db_path = db_path or default_db_path()
    limits = limits or Limits.from_env()
    conn = connect(db_path)
    init_db(conn)
    fail_interrupted_runs(conn, clock())  # a restart ends every run that was going
    redact_old_runs(conn, clock())
    remove_left_cv_files()
    conn.close()

    async def real_agent(
        goal: str, run_id: int, conn: sqlite3.Connection, run_cv_path: Path
    ) -> None:
        chat = make_chat()
        try:
            await run_with_server(goal, run_cv_path, db_path, conn, chat, run_id)
        finally:
            await chat.aclose()

    runner = agent_runner or real_agent
    parse = parse_llm or real_parse_llm
    taxonomy = load_taxonomy()

    async def redact_regularly() -> None:
        while True:
            await asyncio.sleep(REDACT_EVERY)
            conn = connect(db_path)
            try:
                redact_old_runs(conn, clock())
            except sqlite3.Error:
                logger.exception("Erasing old runs failed")
            finally:
                conn.close()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(redact_regularly())
        yield
        task.cancel()

    app = FastAPI(title="Job Application Agent", lifespan=lifespan)
    app.state.tasks = {}  # run_id -> background task: keeps it alive, and lets it be stopped
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.middleware("http")(guard)
    app.add_middleware(LimitBody)

    def database() -> Iterator[sqlite3.Connection]:
        conn = connect(db_path, any_thread=True)
        try:
            yield conn
        finally:
            conn.close()

    Db = Annotated[sqlite3.Connection, Depends(database)]
    Token = Annotated[str, Header(alias="X-Run-Token")]

    async def run_in_background(goal: str, run_id: int, cv: CV | None) -> None:
        conn = connect(db_path)
        # A visitor's CV lives in a temporary file for this run only.
        folder = Path(tempfile.mkdtemp(prefix=CV_FOLDER_PREFIX)) if cv else None
        error = FAILED_RUN  # used only if the run did not finish itself
        limit = None
        try:
            run_cv_path = cv_path
            if folder and cv:
                run_cv_path = folder / "cv.yaml"
                run_cv_path.write_text(yaml.safe_dump(cv.model_dump()), encoding="utf-8")
            # Ends the run itself, not only its claim on the one-run-at-a-time
            # lock (which other runs stop respecting after MAX_RUN_TIME).
            async with asyncio.timeout(MAX_RUN_TIME.total_seconds()) as limit:
                await runner(goal, run_id, conn, run_cv_path)
        except TimeoutError:
            # Our limit, or a timeout inside the run (a network call, say).
            logger.warning("Run %s timed out", run_id)
            error = TOO_LONG if limit.expired() else FAILED_RUN
        except Exception:
            # Also a tool server that does not start, or no LLM key.
            logger.exception("Run %s crashed", run_id)
        finally:
            if folder:
                shutil.rmtree(folder, ignore_errors=True)
            run = get_run(conn, run_id)
            if run and run["status"] == "running":
                finish_run(conn, run_id, "failed", _now(), error=error)
            conn.close()

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/runs", status_code=201)
    async def start_run(body: RunRequest, request: Request, conn: Db) -> dict[str, Any]:
        now = clock()
        redact_old_runs(conn, now)
        ip_hash = limits.hash_ip(client_ip(request, trusted_proxies))
        # No await between the check and create_run, so two requests cannot both pass.
        reason = limits.refusal(conn, ip_hash, now)
        if reason:
            raise HTTPException(429, {"reason": reason, "message": MESSAGES[reason]})
        cv = None
        if body.cv is not None:
            if len(json.dumps(body.cv)) > MAX_CV_JSON_CHARS:
                raise HTTPException(422, "The CV is too long.")
            try:
                cv = CV.model_validate(body.cv)
            except ValidationError as error:
                raise HTTPException(422, "The CV is not valid. Please read it again.") from error
        token = secrets.token_urlsafe(24)
        goal = body.goal.strip()
        run_id = create_run(
            conn, goal, now.isoformat(), ip_hash, hash_token(token), own_cv=cv is not None
        )
        task = asyncio.create_task(run_in_background(goal, run_id, cv))
        app.state.tasks[run_id] = task
        task.add_done_callback(lambda _: app.state.tasks.pop(run_id, None))
        return {"run_id": run_id, "token": token}

    @app.post("/api/cv/parse")
    async def parse_cv(body: ParseRequest, request: Request, conn: Db) -> dict[str, Any]:
        """Read a pasted CV with the LLM. Nothing is stored: the browser keeps the result."""
        now = clock()
        ip_hash = limits.hash_ip(client_ip(request, trusted_proxies))
        # No await between the check and add_parse, so a burst of requests cannot
        # all pass the check (the LLM call itself runs in a thread below).
        reason = limits.parse_refusal(conn, ip_hash, now)
        if reason:
            raise HTTPException(429, {"reason": reason, "message": MESSAGES[reason]})
        # Counted before the call: a failed read used tokens too.
        add_parse(conn, ip_hash, now.isoformat())
        try:
            parsed = await run_in_threadpool(parse_cv_text, body.text, parse)
            cv = drop_used_skills(move_listed_skills(parsed, body.text), taxonomy)
        except LLMError as error:  # no API key, Groq down or out of budget
            raise HTTPException(503, "Reading CVs is not available right now.") from error
        except CVParseError as error:
            raise HTTPException(422, str(error)) from error
        data = cv_data(cv, taxonomy)
        return {
            "cv": cv.model_dump(),
            "facts": data["facts"],
            "verified_skills": [s for s, info in data["skills"].items() if info["verified"]],
            "unverified_skills": [s for s, info in data["skills"].items() if not info["verified"]],
            "warnings": check_parsed(cv, body.text),
        }

    @app.get("/api/runs/{run_id}")
    def show_run(run_id: int, conn: Db, token: Token) -> dict[str, Any]:
        return run_view(conn, owned_run(conn, run_id, token))

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: int, conn: Db, token: Token) -> dict[str, Any]:
        """Stop a run. What it already made (scores, drafts) is kept.

        The run is finished and the stop is added to its trace before the task
        is cancelled. There is no await in between, so the agent cannot write
        its trace again after this.
        """
        run = owned_run(conn, run_id, token)
        if run["status"] == "running":
            save_trace(conn, run_id, [*run["trace"], {"kind": "stopped", "duration_ms": 0}])
            finish_run(conn, run_id, "failed", _now(), error=STOPPED)
            task = app.state.tasks.get(run_id)
            if task:
                task.cancel()  # also closes the run's tool server
        return run_view(conn, get_run(conn, run_id))

    @app.get("/api/demo")
    def show_demo() -> dict[str, Any]:
        """The saved demo run with its letters, open to everyone (it uses the example CV).

        It is a file in the repository, not a database row: free hosting wipes
        the disk on every restart, and the demo must still be there.
        """
        if not demo_path.exists():
            raise HTTPException(404, "No demo run saved")
        return json.loads(demo_path.read_text(encoding="utf-8"))

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
            decide(conn, pending[0]["id"], body.decision, edited_text=edited)
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
