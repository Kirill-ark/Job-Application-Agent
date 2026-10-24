"""The MCP server, tested through a real MCP client connected in memory."""

import json
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from job_agent.cv import DEFAULT_CV_PATH
from job_agent.mcp_server.server import build_server
from tests.test_jobs import FakeArbeitnow

pytestmark = pytest.mark.anyio

JOB = "werkstudent-machine-learning-acme-1"
LETTER = {
    "paragraphs": [
        [{"text": "Dear Acme team,", "kind": "other", "cites": []}],
        [
            {
                "text": "My pneumonia classifier reached 0.93 recall on the held-out test set.",
                "kind": "claim",
                "cites": ["p1.f2"],
            }
        ],
        [{"text": "Best regards, Alex Example", "kind": "other", "cites": []}],
    ]
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def fake_reasons(system: str, user: str) -> str:
    return "Good fit."


def fake_letter(system: str, user: str) -> str:
    return json.dumps(LETTER)


@pytest.fixture
def server(tmp_path: Path) -> MCPServer:
    return build_server(
        tmp_path / "test.db",
        DEFAULT_CV_PATH,
        reasons_llm=fake_reasons,
        letter_llm=fake_letter,
        http_client=FakeArbeitnow().client(),
    )


async def call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    return result.structured_content


async def error_text(client: Client, tool: str, **arguments: Any) -> str:
    result = await client.call_tool(tool, arguments)
    assert result.is_error
    return result.content[0].text


async def test_exactly_the_five_agent_tools(server: MCPServer) -> None:
    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {
        "search_jobs",
        "read_cv",
        "score_match",
        "draft_cover_letter",
        "request_approval",
    }  # no decide, no export: those are the human's
    assert tools["read_cv"].annotations.read_only_hint is True
    assert tools["search_jobs"].input_schema["required"] == ["query"]


async def test_read_cv(server: MCPServer) -> None:
    async with Client(server) as client:
        cv = await call(client, "read_cv")
    assert cv["name"] == "Alex Example"
    assert "PyTorch" in cv["verified_skills"]
    assert cv["unverified_skills"] == ["AWS", "Kubernetes"]
    assert cv["facts"]["p1.f2"] == "Reached 0.93 recall on the held-out test set"


async def test_search_jobs(server: MCPServer) -> None:
    async with Client(server) as client:
        result = await call(
            client, "search_jobs", query="machine learning", location="Berlin", level="student"
        )
    assert result["jobs"][0]["id"] == JOB
    assert result["count"] == len(result["jobs"])


async def test_bad_level_is_explained(server: MCPServer) -> None:
    async with Client(server) as client:
        text = await error_text(client, "search_jobs", query="ml", level="senior")
    assert "level must be one of" in text


async def test_whole_flow_up_to_the_approval(server: MCPServer) -> None:
    async with Client(server) as client:
        await call(client, "search_jobs", query="machine learning")
        match = await call(client, "score_match", job_id=JOB)
        draft = await call(client, "draft_cover_letter", job_id=JOB)
        approval = await call(
            client, "request_approval", action="mark_ready", draft_id=draft["draft_id"]
        )
    assert match["label"] in ("good", "maybe", "bad")
    assert match["reasons"] == "Good fit."
    assert draft["grounded"] is True
    assert "[p1.f2]" not in draft["letter"]
    assert approval["status"] == "pending"


async def test_unknown_job_is_explained(server: MCPServer) -> None:
    async with Client(server) as client:
        text = await error_text(client, "score_match", job_id="nope")
    assert "Use an id returned by search_jobs" in text


async def test_sending_is_refused(server: MCPServer) -> None:
    async with Client(server) as client:
        await call(client, "search_jobs", query="machine learning")
        draft = await call(client, "draft_cover_letter", job_id=JOB)
        result = await call(client, "request_approval", action="send", draft_id=draft["draft_id"])
    assert result["status"] == "refused"


async def test_no_letters_without_llm(tmp_path: Path) -> None:
    server = build_server(tmp_path / "t.db", DEFAULT_CV_PATH, http_client=FakeArbeitnow().client())
    async with Client(server) as client:
        await call(client, "search_jobs", query="machine learning")
        text = await error_text(client, "draft_cover_letter", job_id=JOB)
        match = await call(client, "score_match", job_id=JOB)
    assert "LLM is not configured" in text
    assert match["reasons"].startswith(match["label"].capitalize())  # template reasons


async def test_unknown_draft_is_explained(server: MCPServer) -> None:
    async with Client(server) as client:
        text = await error_text(client, "request_approval", action="mark_ready", draft_id=-1)
    assert "Unknown draft" in text


async def test_crash_details_stay_on_the_server(tmp_path: Path) -> None:
    def broken_llm(system: str, user: str) -> str:
        raise RuntimeError("secret detail /Users/someone/.env")

    server = build_server(
        tmp_path / "t.db",
        DEFAULT_CV_PATH,
        letter_llm=broken_llm,
        http_client=FakeArbeitnow().client(),
    )
    async with Client(server) as client:
        await call(client, "search_jobs", query="machine learning")
        text = await error_text(client, "draft_cover_letter", job_id=JOB)
    assert text == "Error executing tool draft_cover_letter"
