"""Run the agent from the terminal: python -m job_agent.agent "your goal"."""

import json
import sys
import time
from typing import Any

import anyio
from dotenv import load_dotenv

from job_agent.agent.runner import run_with_server
from job_agent.cv import DEFAULT_CV_PATH
from job_agent.db import REPO_ROOT, connect, default_db_path, init_db
from job_agent.llm import make_chat

DEFAULT_GOAL = (
    "Find machine learning working student or internship jobs in Berlin for me "
    "and prepare cover letters for the best ones."
)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    goal = " ".join(sys.argv[1:]) or DEFAULT_GOAL
    conn = connect()
    init_db(conn)
    started = time.monotonic()

    def show(step: dict[str, Any]) -> None:
        elapsed = time.monotonic() - started
        if step["kind"] == "llm":
            calls = ", ".join(step["tool_calls"]) or "final answer"
            tokens = step["prompt_tokens"] + step["completion_tokens"]
            print(f"[{elapsed:5.0f}s] LLM -> {calls} ({tokens} tokens)")
        else:
            outcome = "ok" if step["ok"] else f"ERROR {step['error']}"
            args = json.dumps(step["arguments"], ensure_ascii=False)
            print(f"[{elapsed:5.0f}s]   {step['tool']}({args}) {outcome}")

    summary = anyio.run(
        run_with_server, goal, DEFAULT_CV_PATH, default_db_path(), conn, make_chat(), None, show
    )
    print(f"\nStatus: {summary['status']}  {summary['error'] or ''}")
    print(f"LLM calls: {summary['llm_calls']}, agent tokens: {summary['tokens']}")
    print(f"Drafts: {summary['drafts']}\n")
    print(summary["answer"] or "")


if __name__ == "__main__":
    main()
