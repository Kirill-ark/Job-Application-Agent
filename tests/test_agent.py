"""The agent loop with a scripted LLM and the real MCP server in memory."""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from job_agent.agent.loop import MAX_LETTERS, MAX_STEPS, run_agent
from job_agent.cv import DEFAULT_CV_PATH
from job_agent.db import connect, init_db
from job_agent.db.queries import get_run
from job_agent.llm import ChatReply, LLMError, ToolCall
from job_agent.mcp_server.server import build_server
from tests.test_jobs import FakeArbeitnow
from tests.test_mcp_server import JOB, fake_letter

pytestmark = pytest.mark.anyio

OTHER_JOB = "ai-research-intern-delta-4"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def server(db_path: Path) -> MCPServer:
    return build_server(
        db_path, DEFAULT_CV_PATH, letter_llm=fake_letter, http_client=FakeArbeitnow().client()
    )


def calls(*pairs: tuple[str, dict[str, Any]]) -> ChatReply:
    """A reply that asks for these tool calls."""
    return ChatReply(
        content=None,
        tool_calls=[
            ToolCall(f"call_{i}", name, json.dumps(args)) for i, (name, args) in enumerate(pairs)
        ],
        prompt_tokens=100,
        completion_tokens=10,
    )


def final(text: str) -> ChatReply:
    return ChatReply(content=text, prompt_tokens=100, completion_tokens=10)


class ScriptedLLM:
    """Plays back replies in order and keeps every request it got."""

    def __init__(self, *replies: ChatReply | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []
        self.offered: list[set[str]] = []  # the tool names of every request

    async def __call__(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ChatReply:
        self.requests.append([dict(m) for m in messages])
        self.tools = tools
        self.offered.append({t["function"]["name"] for t in tools})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


SEARCH = ("search_jobs", {"query": "machine learning", "location": "Berlin", "level": "student"})


async def run(server: MCPServer, conn: sqlite3.Connection, llm: ScriptedLLM, **kw: Any) -> Any:
    async with Client(server) as client:
        return await run_agent("Find ML jobs", client, llm, conn, **kw)


async def test_full_run(server: MCPServer, conn: sqlite3.Connection) -> None:
    llm = ScriptedLLM(
        calls(("read_cv", {}), SEARCH),
        calls(("score_match", {"job_id": JOB}), ("score_match", {"job_id": OTHER_JOB})),
        calls(("draft_cover_letter", {"job_id": JOB})),
        final("One letter is waiting for your approval."),
    )
    summary = await run(server, conn, llm)

    assert summary["status"] == "done"
    assert summary["answer"] == "One letter is waiting for your approval."
    assert [s["job_id"] for s in summary["scores"]] == [JOB, OTHER_JOB]  # best first
    assert summary["drafts"] == [
        {
            "draft_id": 1,
            "job_id": JOB,
            "title": "Werkstudent Machine Learning (m/w/d)",
            "company": "Acme GmbH",
            "status": "draft",
            "grounded": True,
            "approval": "pending",
        }
    ]
    assert summary["llm_calls"] == 4
    assert summary["tokens"] == 4 * 110

    # The LLM saw the MCP tools and got every result back with its call id.
    assert {t["function"]["name"] for t in llm.tools} >= {"search_jobs", "request_approval"}
    last_request = llm.requests[-1]
    tool_messages = [m for m in last_request if m["role"] == "tool"]
    assert len(tool_messages) == 5
    assert tool_messages[0]["tool_call_id"] == "call_0"
    assert '"name":"Alex Example"' in tool_messages[0]["content"]  # compact JSON


async def test_trace_is_saved_after_every_step(server: MCPServer, conn: sqlite3.Connection) -> None:
    seen: list[int] = []

    def on_step(step: dict[str, Any]) -> None:
        run = get_run(conn, 1)
        seen.append(len(run["trace"]))

    await run(server, conn, ScriptedLLM(calls(SEARCH), final("Done")), on_step=on_step)
    assert seen == [1, 2, 3]  # llm, tool, llm: each was in the database when reported
    trace = get_run(conn, 1)["trace"]
    assert [s["kind"] for s in trace] == ["llm", "tool", "llm"]
    assert trace[1]["arguments"]["query"] == "machine learning"
    assert trace[1]["result"]["count"] >= 1


async def test_step_limit(server: MCPServer, conn: sqlite3.Connection) -> None:
    summary = await run(server, conn, ScriptedLLM(calls(("read_cv", {}))))
    assert summary["status"] == "failed"
    assert summary["error"] == f"Stopped after {MAX_STEPS} steps"
    assert summary["llm_calls"] == MAX_STEPS


async def test_letter_limit(server: MCPServer, conn: sqlite3.Connection) -> None:
    draft = ("draft_cover_letter", {"job_id": JOB})
    llm = ScriptedLLM(calls(SEARCH), calls(*[draft] * (MAX_LETTERS + 1)), final("Done"))
    summary = await run(server, conn, llm)
    assert len(summary["drafts"]) == MAX_LETTERS
    last_tool_message = [m for m in llm.requests[-1] if m["role"] == "tool"][-1]
    assert last_tool_message["content"] == "ERROR: Letter limit reached: at most 3 letters per run"


async def test_tool_error_goes_back_to_the_llm(server: MCPServer, conn: sqlite3.Connection) -> None:
    llm = ScriptedLLM(
        calls(SEARCH),
        calls(("score_match", {"job_id": "made-up-id"})),
        final("Done"),
    )
    summary = await run(server, conn, llm)
    assert summary["status"] == "done"
    error = [m for m in llm.requests[2] if m["role"] == "tool"][1]["content"]
    assert error.startswith("ERROR: Error executing tool score_match: Unknown vacancy")


async def test_id_tools_are_offered_only_after_a_search(
    server: MCPServer, conn: sqlite3.Connection
) -> None:
    # Seen in the evals: the model skipped the search and scored made-up ids j1..j8.
    llm = ScriptedLLM(
        calls(("read_cv", {})),
        calls(("score_match", {"job_id": "j1"})),  # called anyway, not offered
        calls(SEARCH),
        final("Done"),
    )
    await run(server, conn, llm)
    assert "score_match" not in llm.offered[0] | llm.offered[1] | llm.offered[2]
    assert {"read_cv", "search_jobs", "request_approval"} <= llm.offered[0]
    assert {"score_match", "draft_cover_letter"} <= llm.offered[3]
    error = [m for m in llm.requests[2] if m["role"] == "tool"][1]["content"]
    assert error == "ERROR: Call search_jobs first: vacancy ids come only from its results"


async def test_unknown_tool_and_bad_arguments(server: MCPServer, conn: sqlite3.Connection) -> None:
    bad_json = ChatReply(content=None, tool_calls=[ToolCall("c1", "read_cv", "{not json")])
    llm = ScriptedLLM(calls(("send_email", {"to": "boss@example.com"})), bad_json, final("Done"))
    await run(server, conn, llm)
    errors = [m["content"] for m in llm.requests[-1] if m["role"] == "tool"]
    assert errors[0].startswith("ERROR: Unknown tool 'send_email'")
    assert errors[1] == "ERROR: The arguments are not a JSON object"


async def test_too_many_failed_steps_in_a_row(server: MCPServer, conn: sqlite3.Connection) -> None:
    summary = await run(server, conn, ScriptedLLM(calls(("score_match", {"job_id": "x"}))))
    assert summary["status"] == "failed"
    assert summary["error"] == "Stopped after 3 steps in a row in which every tool call failed"
    assert summary["llm_calls"] == 3


async def test_one_step_of_made_up_ids_can_be_corrected(
    server: MCPServer, conn: sqlite3.Connection
) -> None:
    # Seen live: the model skipped the search and invented 16 ids in one step.
    made_up = calls(*[("score_match", {"job_id": f"a{i}"}) for i in range(16)])
    llm = ScriptedLLM(
        made_up, calls(SEARCH), calls(("score_match", {"job_id": JOB})), final("Done")
    )
    summary = await run(server, conn, llm)
    assert summary["status"] == "done"
    assert [s["job_id"] for s in summary["scores"]] == [JOB]


async def test_llm_failure(server: MCPServer, conn: sqlite3.Connection) -> None:
    summary = await run(server, conn, ScriptedLLM(LLMError("LLM request failed: 429")))
    assert summary["status"] == "failed"
    assert summary["error"] == "LLM request failed: 429"


async def test_summary_trusts_tools_not_the_llm(
    server: MCPServer, conn: sqlite3.Connection
) -> None:
    llm = ScriptedLLM(final("I scored 10 jobs and wrote 3 letters for you."))
    summary = await run(server, conn, llm)
    assert summary["scores"] == []
    assert summary["drafts"] == []


async def test_tool_results_are_shown_in_full_only_once(
    server: MCPServer, conn: sqlite3.Connection
) -> None:
    llm = ScriptedLLM(
        calls(SEARCH),
        calls(("score_match", {"job_id": JOB})),
        calls(("draft_cover_letter", {"job_id": JOB})),
        final("Done"),
    )
    await run(server, conn, llm)

    def tool_contents(request: list[dict[str, Any]]) -> list[str]:
        return [m["content"] for m in request if m["role"] == "tool"]

    # Request 2 is the first to see the search: in full.
    assert '"snippet"' in tool_contents(llm.requests[1])[0]
    # Request 3: the search is short now, the new score is in full.
    search, score = tool_contents(llm.requests[2])
    assert json.loads(search) == {
        "jobs": [{k: j[k] for k in ("id", "title", "company")} for j in json.loads(search)["jobs"]]
    }
    assert '"snippet"' not in search
    assert '"components"' not in score and '"reasons"' in score
    # Request 4: the score is short too, the letter is in full.
    _, score, letter = tool_contents(llm.requests[3])
    assert '"reasons"' not in score and '"score"' in score
    assert '"letter"' in letter
    # The trace keeps everything in full.
    search_step = [s for s in get_run(conn, 1)["trace"] if s.get("tool") == "search_jobs"][0]
    assert "snippet" in search_step["result"]["jobs"][0]
