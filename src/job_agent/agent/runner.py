"""Start an MCP server for one run and run the agent with it.

Each run gets its own server process over stdio, serving that run's CV.
"""

import os
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from job_agent.agent.loop import run_agent
from job_agent.llm import Chat


def server_parameters(cv_path: Path, db_path: Path) -> StdioServerParameters:
    env = {
        **os.environ,  # GROQ_API_KEY and the model settings
        "CV_PATH": str(cv_path.resolve()),
        "DATABASE_PATH": str(db_path.resolve()),
        "REASONS_WITH_LLM": "0",  # the agent explains the scores itself; saves tokens
    }
    return StdioServerParameters(
        command=sys.executable, args=["-m", "job_agent.mcp_server"], env=env
    )


async def run_with_server(
    goal: str,
    cv_path: Path,
    db_path: Path,
    conn: sqlite3.Connection,
    chat: Chat,
    ip_hash: str | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    async with Client(server_parameters(cv_path, db_path)) as client:
        return await run_agent(goal, client, chat, conn, ip_hash, on_step)
