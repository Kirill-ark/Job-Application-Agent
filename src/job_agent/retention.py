"""What happens to visitors' data after a run.

Letters, the trace and the goal can hold personal data from a visitor's CV.
After KEEP_FOR they are erased. The approval log is append-only and is never
deleted: its rows keep only ids, actions, decisions and times, so the log
stays complete while the content is gone. The saved demo run is kept.
"""

import json
import sqlite3
from datetime import datetime, timedelta

from job_agent.agent.loop import created_drafts

KEEP_FOR = timedelta(hours=24)
DELETED = "[deleted]"


def redact_old_runs(conn: sqlite3.Connection, now: datetime) -> int:
    """Erase the content of runs older than KEEP_FOR. Returns how many runs."""
    cutoff = (now - KEEP_FOR).isoformat()
    rows = conn.execute(
        "SELECT id, trace FROM runs WHERE is_demo = 0 AND started_at < ? AND goal != ?",
        (cutoff, DELETED),
    ).fetchall()
    demo_drafts = {
        draft_id
        for (trace,) in conn.execute("SELECT trace FROM runs WHERE is_demo = 1")
        for draft_id in _draft_ids(trace)
    }
    with conn:
        for run_id, trace in rows:
            for draft_id in _draft_ids(trace) - demo_drafts:
                conn.execute(
                    """
                    UPDATE drafts SET text = ?, edited_text = NULL, cited_facts = '{}',
                                      issues = '[]'
                    WHERE id = ?
                    """,
                    (DELETED, draft_id),
                )
            conn.execute(
                """
                UPDATE runs SET goal = ?, trace = '[]', answer = NULL, ip_hash = NULL
                WHERE id = ?
                """,
                (DELETED, run_id),
            )
        # Scores are only a cache; old ones may come from a visitor's CV.
        conn.execute("DELETE FROM scores WHERE created_at < ?", (cutoff,))
    return len(rows)


def fail_interrupted_runs(conn: sqlite3.Connection, now: datetime) -> None:
    """Runs left 'running' by a restart will never finish: mark them failed."""
    with conn:
        conn.execute(
            """
            UPDATE runs SET status = 'failed', error = 'Interrupted by a server restart',
                            finished_at = ?
            WHERE status = 'running'
            """,
            (now.isoformat(),),
        )


def _draft_ids(trace_json: str) -> set[int]:
    return set(created_drafts(json.loads(trace_json)))
