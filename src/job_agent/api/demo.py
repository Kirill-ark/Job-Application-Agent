"""Save a finished run as the demo run: python -m job_agent.api.demo RUN_ID.

The page replays the demo run when a limit is reached or the LLM is down.
It is written to data/demo_run.json, which is committed: the server's disk
is wiped on every restart, the repository is not.
Without RUN_ID, lists the recent finished runs to choose from.
"""

import json
import sys

from dotenv import load_dotenv

from job_agent import REPO_ROOT
from job_agent.agent.loop import run_summary
from job_agent.api.app import DEMO_PATH, demo_view
from job_agent.db import connect, init_db
from job_agent.db.queries import get_run, recent_finished_runs, set_demo_run


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    conn = connect()
    init_db(conn)
    if len(sys.argv) < 2:
        for row in recent_finished_runs(conn):
            summary = run_summary(conn, row["id"])
            print(
                f"{row['id']:>4}  {row['started_at'][:16]}  {len(summary['scores'])} scores, "
                f"{len(summary['drafts'])} drafts  {row['goal'][:60]}"
            )
        return
    run_id = int(sys.argv[1])
    run = get_run(conn, run_id)
    if run and run["own_cv"]:
        sys.exit(
            f"Run {run_id} used a visitor's CV: the demo is public, so it must use the example"
        )
    summary = run_summary(conn, run_id)
    if summary["status"] != "done" or not summary["drafts"]:
        sys.exit(f"Run {run_id} is not a finished run with drafts")
    set_demo_run(conn, run_id)  # keeps it in the local database too (see retention.py)
    DEMO_PATH.write_text(json.dumps(demo_view(conn, run_id), ensure_ascii=False, indent=1) + "\n")
    print(f"Run {run_id} is now the demo run, saved to {DEMO_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
