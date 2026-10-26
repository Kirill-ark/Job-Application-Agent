"""SQLite connection and schema setup.

The FastAPI app and the MCP server are separate processes that share one
database file, so every connection uses WAL mode and waits for locks
instead of failing with "database is locked".
"""

import os
import sqlite3
from pathlib import Path

from job_agent import REPO_ROOT

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


# Columns added after a table was first created. CREATE TABLE IF NOT EXISTS
# skips existing tables, so older databases get these with ALTER TABLE.
# Add a line here whenever schema.sql gets a new column.
ADDED_COLUMNS = [
    ("drafts", "issues", "TEXT NOT NULL DEFAULT '[]'"),
    ("runs", "answer", "TEXT"),
    ("runs", "error", "TEXT"),
    ("runs", "token_hash", "TEXT"),
    ("runs", "own_cv", "INTEGER NOT NULL DEFAULT 0 CHECK (own_cv IN (0, 1))"),
]


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and add missing columns. Safe to call on every start."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    for table, column, definition in ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    conn.commit()
