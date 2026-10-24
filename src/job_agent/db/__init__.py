"""SQLite connection and schema setup.

The FastAPI app and the MCP server are separate processes that share one
database file, so every connection uses WAL mode and waits for locks
instead of failing with "database is locked".
"""

import os
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def default_db_path() -> Path:
    return REPO_ROOT / os.environ.get("DATABASE_PATH", "data/job_agent.db")


def connect(path: Path | None = None, any_thread: bool = False) -> sqlite3.Connection:
    """Open the database with the settings every connection needs.

    `any_thread` lets the connection move between threads. FastAPI needs it:
    it may open a request's connection in one thread and use it in another.
    One request still uses its connection one call at a time, which is safe.
    """
    path = path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=not any_thread)
    conn.row_factory = sqlite3.Row  # rows can be read by column name
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignores REFERENCES without this
    conn.execute("PRAGMA journal_mode = WAL")  # readers do not block the writer
    conn.execute("PRAGMA busy_timeout = 5000")  # wait up to 5 s for a lock
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables. Safe to call on every start."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
