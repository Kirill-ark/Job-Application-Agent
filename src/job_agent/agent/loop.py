"""The agent loop.

    1. The LLM gets the user's goal and the tools of the MCP server.
    2. It answers either with tool calls or with a final message.
    3. We run the tool calls through MCP and give the LLM the results.
    4. Repeat until the final message, or until a limit stops the run.

Every step goes into the run's trace in SQLite right away, so the UI can show
progress while the run is going.

The model has no memory, so every request carries the whole conversation.
To keep that small, a tool result is shown in full only once: after the model
has answered to it, it is replaced by a short version (see SHORT_FIELDS).
The full results stay in the trace.

Limits are enforced here in code, not only asked for in the prompt:
MAX_STEPS LLM calls, MAX_LETTERS letters, MAX_FAILED_STEPS steps in a row in which
every tool call failed. The same goes for the order of the work: the tools that
take a vacancy id are offered only after a search has returned some ids, and
they only accept ids that a search of this run returned.
"""

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from mcp import Client

from job_agent.approvals import approval_requests
from job_agent.db.queries import (
    create_run,
    finish_run,
    get_draft,
    get_run,
    save_trace,
)
from job_agent.llm import Chat, ChatReply, InvalidToolCall, LLMError, ToolCall

logger = logging.getLogger(__name__)

MAX_STEPS = 10
MAX_LETTERS = 3
# Counted per step, not per call: one step with 16 made-up ids is one mistake the
# model can correct after reading the errors, not 16 (found in a live run).
MAX_FAILED_STEPS = 3
# Tools that need a vacancy id. Before the first search the model is not shown
# them: asked nicely in the prompt, it still skipped the search and scored
# made-up ids like j1..j8 (found by the evals).
NEED_SEARCH = {"score_match", "draft_cover_letter"}

# What stays of a tool result in the conversation after the model has seen it.
SHORT_FIELDS = {
    "read_cv": ["name", "location", "looking_for"],
    "search_jobs": ["id", "title", "company"],  # per vacancy
    "score_match": ["job_id", "title", "company", "score", "label", "gaps"],
    "draft_cover_letter": ["draft_id", "job_id", "grounded", "approval"],
}

SYSTEM_PROMPT = """You help a student find a job in Germany and prepare cover letters.
Use the tools. Work in as few steps as possible:
call independent tools together in ONE step (for example score_match for
several vacancies at once).

Plan:
1. read_cv, to know the candidate's location and the levels they look for.
2. search_jobs once or twice (limit 8) with a short topic query, the city, and the level.
   Always search before scoring: vacancy ids come only from search_jobs results.
   Never make up an id.
3. score_match for EVERY vacancy the searches returned (each id once), all in one
   step. Do not skip any by its title: titles often hide a good fit, and scoring
   is cheap.
4. draft_cover_letter for the best matches (label good, else the best maybe),
   at most 3, all in one step. Never for "bad" matches. Each draft goes to the
   user's approval queue by itself.

Rules:
- You cannot send, email or submit anything. The user reviews and sends letters.
- Only report what the tools returned. Never invent vacancies, scores or drafts.
- Final answer: 2-4 sentences of plain text, no Markdown, no tables, no lists.
  The page already shows the scores table and the drafts, so do not repeat them:
  say which vacancy fits best and why, the most common gap, and how many letters
  wait for the user's approval. Do not ask questions: the user cannot reply to you.
  Write it in the language of the user's goal."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def mcp_tools_for_llm(tools: list[Any]) -> list[dict[str, Any]]:
    """MCP tool descriptions in the format of OpenAI-style tool calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def before_search(llm_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tools to offer while no search has returned vacancy ids yet."""
    return [t for t in llm_tools if t["function"]["name"] not in NEED_SEARCH]


def shorten(tool: str, result: dict[str, Any]) -> str:
    """The short version of a tool result, for requests after the first one."""
    fields = SHORT_FIELDS.get(tool)
    if fields is None:
        short = result
    elif tool == "search_jobs":
        short = {"jobs": [{k: job[k] for k in fields} for job in result["jobs"]]}
    else:
        short = {k: result[k] for k in fields if k in result}
    return to_json(short)


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def assistant_message(reply: ChatReply) -> dict[str, Any]:
    """The LLM's reply as a message for the next request."""
    message: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in reply.tool_calls
        ]
    return message


class Run:
    """One agent run: its trace and counters."""

    def __init__(self, conn: sqlite3.Connection, run_id: int, on_step: Callable | None) -> None:
        self.conn = conn
        self.run_id = run_id
        self.on_step = on_step
        self.trace: list[dict[str, Any]] = []
        self.letters = 0
        self.found: set[str] = set()  # vacancy ids the searches of this run returned
        self.failed_steps = 0  # steps in a row in which every call failed

    @property
    def searched(self) -> bool:
        return bool(self.found)

    def record(self, step: dict[str, Any]) -> None:
        self.trace.append(step)
        try:
            save_trace(self.conn, self.run_id, self.trace)
        except sqlite3.OperationalError:
            # The database is busy (the tool server writes the vacancy cache in
            # one go). The full trace is saved again with the next step.
            logger.warning("Could not save the trace of run %s yet", self.run_id)
        if self.on_step:
            self.on_step(step)


async def call_tool(client: Client, run: Run, names: set[str], call: ToolCall) -> tuple[str, str]:
    """Run one tool call. Returns the text the LLM gets now and its short version for later."""
    started = time.monotonic()
    step: dict[str, Any] = {"kind": "tool", "tool": call.name, "arguments": call.arguments}
    result = None
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        arguments = None
    if arguments is not None:
        step["arguments"] = arguments

    if call.name not in names:
        error = f"Unknown tool '{call.name}'. Use one of: {', '.join(sorted(names))}"
    elif not isinstance(arguments, dict):
        error = "The arguments are not a JSON object"
    elif call.name in NEED_SEARCH and not run.searched:
        error = "Call search_jobs first: vacancy ids come only from its results"
    elif call.name in NEED_SEARCH and arguments.get("job_id") not in run.found:
        error = (
            f"Unknown vacancy {arguments.get('job_id')!r}: use an id exactly as search_jobs "
            "returned it in this run"
        )
    elif call.name == "draft_cover_letter" and run.letters >= MAX_LETTERS:
        error = f"Letter limit reached: at most {MAX_LETTERS} letters per run"
    else:
        if call.name == "draft_cover_letter":
            run.letters += 1  # counted when tried: a failed letter cost tokens too
        try:
            result = await client.call_tool(call.name, arguments)
        except Exception as failure:  # the tool server broke; the model may still finish
            logger.exception("Tool %s failed in run %s", call.name, run.run_id)
            result, error = None, f"The tool failed: {type(failure).__name__}"
        else:
            error = None
            if result.is_error:
                error = result.content[0].text if result.content else "The tool failed"

    step["duration_ms"] = round((time.monotonic() - started) * 1000)
    if error:
        run.record({**step, "ok": False, "error": error})
        return f"ERROR: {error}", f"ERROR: {error}"
    data = result.structured_content
    if call.name == "search_jobs":
        run.found |= {job["id"] for job in data["jobs"]}
    run.record({**step, "ok": True, "result": data})
    return to_json(data), shorten(call.name, data)


async def run_agent(
    goal: str,
    client: Client,
    chat: Chat,
    conn: sqlite3.Connection,
    run_id: int | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the agent for one goal with the tools of a connected MCP client.

    `run_id` is a run the caller already created (the web API does, to answer
    the browser at once); without it a new run is created.

    Returns the run summary (see `run_summary`).
    """
    run = Run(conn, run_id or create_run(conn, goal, _now()), on_step)
    tools = (await client.list_tools()).tools
    names = {tool.name for tool in tools}
    llm_tools = mcp_tools_for_llm(tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    status, answer, error = "failed", None, f"Stopped after {MAX_STEPS} steps"
    seen_once: list[tuple[dict[str, Any], str]] = []  # (tool message, its short version)
    for _ in range(MAX_STEPS):
        started = time.monotonic()
        try:
            offered = llm_tools if run.searched else before_search(llm_tools)
            reply = await chat(messages, offered)
        except InvalidToolCall as invalid:
            # Like a failed tool call: the model reads why and tries again.
            run.record({"kind": "invalid_call", "error": str(invalid), "duration_ms": 0})
            messages.append(
                {"role": "user", "content": f"Your tool call was rejected: {invalid}. Fix it."}
            )
            run.failed_steps += 1
            if run.failed_steps >= MAX_FAILED_STEPS:
                error = f"Stopped after {MAX_FAILED_STEPS} invalid or failed steps in a row"
                break
            continue
        except LLMError as llm_error:
            error = str(llm_error)
            break
        # The model has now seen the last tool results in full: shorten them.
        for message, short in seen_once:
            message["content"] = short
        seen_once = []
        run.record(
            {
                "kind": "llm",
                "text": reply.content,
                "tool_calls": [call.name for call in reply.tool_calls],
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "duration_ms": round((time.monotonic() - started) * 1000),
            }
        )
        messages.append(assistant_message(reply))
        if not reply.tool_calls:
            status, answer, error = "done", reply.content, None
            break
        all_failed = True
        for call in reply.tool_calls:
            full, short = await call_tool(client, run, names, call)
            all_failed = all_failed and full.startswith("ERROR: ")
            message = {"role": "tool", "tool_call_id": call.id, "content": full}
            messages.append(message)
            seen_once.append((message, short))
        run.failed_steps = run.failed_steps + 1 if all_failed else 0
        if run.failed_steps >= MAX_FAILED_STEPS:
            error = (
                f"Stopped after {MAX_FAILED_STEPS} steps in a row in which every tool call failed"
            )
            break

    finish_run(conn, run.run_id, status, _now(), answer, error)
    return run_summary(conn, run.run_id)


def run_summary(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """What the run produced, taken from the trace and the database.

    The LLM's final message is only a retelling; the scores and drafts shown
    to the user come from real tool results, so the model cannot claim work it
    did not do.
    """
    run = get_run(conn, run_id)
    if run is None:
        raise ValueError(f"Unknown run {run_id}")
    tool_steps = [s for s in run["trace"] if s["kind"] == "tool" and s["ok"]]
    scores = {s["result"]["job_id"]: s["result"] for s in tool_steps if s["tool"] == "score_match"}
    drafts = []
    for step in tool_steps:
        if step["tool"] != "draft_cover_letter":
            continue
        draft = get_draft(conn, step["result"]["draft_id"])
        requests = approval_requests(conn, draft["id"])
        drafts.append(
            {
                "draft_id": draft["id"],
                "job_id": draft["vacancy_id"],
                "title": step["result"]["title"],
                "company": step["result"]["company"],
                "status": draft["status"],
                "grounded": not draft["issues"],
                "approval": requests[-1]["status"] if requests else None,
            }
        )
    llm_steps = [s for s in run["trace"] if s["kind"] == "llm"]
    return {
        "run_id": run_id,
        "status": run["status"],
        "answer": run["answer"],
        "error": run["error"],
        "scores": sorted(scores.values(), key=lambda s: -s["score"]),
        "drafts": drafts,
        "llm_calls": len(llm_steps),
        "tokens": sum(s["prompt_tokens"] + s["completion_tokens"] for s in llm_steps),
    }
