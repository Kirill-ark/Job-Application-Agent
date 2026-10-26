"""What happens to visitors' data after a run.

Letters, the trace and the goal can hold personal data from a visitor's CV.
After KEEP_FOR they are erased. The approval log is append-only and is never
deleted: its rows keep only ids, actions, decisions and times, so the log
stays complete while the content is gone. The saved demo run is kept.

Drafts are erased by their own age, not through the run that made them: a run
stopped while a letter was being written has a draft its trace does not show.
"""

import json
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from job_agent.db.queries import created_drafts

KEEP_FOR = timedelta(hours=24)
DELETED = "[deleted]"
CV_FOLDER_PREFIX = "job-agent-cv-"  # temporary folders of visitors' CVs


def redact_old_runs(conn: sqlite3.Connection, now: datetime) -> int:
    """Erase the content of runs and drafts older than KEEP_FOR. Returns how many runs."""
    cutoff = (now - KEEP_FOR).isoformat()
    run_ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM runs WHERE is_demo = 0 AND started_at < ? AND goal != ?",
            (cutoff, DELETED),
        )
    ]
    demo_drafts = {
        draft_id
        for (trace,) in conn.execute("SELECT trace FROM runs WHERE is_demo = 1")
        for draft_id in created_drafts(json.loads(trace))
    }
    old_drafts = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM drafts WHERE created_at < ? AND text != ?", (cutoff, DELETED)
        )
        if row[0] not in demo_drafts
    ]
    with conn:
        for draft_id in old_drafts:
            conn.execute(
                """
                UPDATE drafts SET text = ?, edited_text = NULL, cited_facts = '{}',
                                  issues = '[]'
                WHERE id = ?
                """,
                (DELETED, draft_id),
            )
        for run_id in run_ids:
            conn.execute(
                """
                UPDATE runs SET goal = ?, trace = '[]', answer = NULL, ip_hash = NULL
                WHERE id = ?
                """,
                (DELETED, run_id),
            )
        # Scores are only a cache; old ones may come from a visitor's CV.
        conn.execute("DELETE FROM scores WHERE created_at < ?", (cutoff,))
        conn.execute("DELETE FROM cv_parses WHERE created_at < ?", (cutoff,))
    return len(run_ids)


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


def remove_left_cv_files() -> None:
    """Delete visitors' CV folders that a killed server could not clean up."""
    for folder in Path(tempfile.gettempdir()).glob(CV_FOLDER_PREFIX + "*"):
        shutil.rmtree(folder, ignore_errors=True)
