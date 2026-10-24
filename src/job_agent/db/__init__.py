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


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the database with the settings every connection needs."""
    path = path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row  # rows can be read by column name
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignores REFERENCES without this
    conn.execute("PRAGMA journal_mode = WAL")  # readers do not block the writer
    conn.execute("PRAGMA busy_timeout = 5000")  # wait up to 5 s for a lock
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables. Safe to call on every start."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
