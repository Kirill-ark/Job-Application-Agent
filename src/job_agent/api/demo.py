"""Save a finished run as the demo run: python -m job_agent.api.demo RUN_ID.

The page replays the demo run when a limit is reached or the LLM is down.
Without RUN_ID, lists the recent finished runs to choose from.
"""

import sys

from dotenv import load_dotenv

from job_agent.agent.loop import run_summary
from job_agent.db import REPO_ROOT, connect, init_db
from job_agent.db.queries import set_demo_run


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    conn = connect()
    init_db(conn)
    if len(sys.argv) < 2:
        rows = conn.execute(
            "SELECT id, started_at, goal FROM runs WHERE status = 'done' ORDER BY id DESC LIMIT 10"
        ).fetchall()
        for row in rows:
            summary = run_summary(conn, row["id"])
            print(
                f"{row['id']:>4}  {row['started_at'][:16]}  {len(summary['scores'])} scores, "
                f"{len(summary['drafts'])} drafts  {row['goal'][:60]}"
            )
        return
    run_id = int(sys.argv[1])
    summary = run_summary(conn, run_id)
    if summary["status"] != "done" or not summary["drafts"]:
        sys.exit(f"Run {run_id} is not a finished run with drafts")
    set_demo_run(conn, run_id)
    print(f"Run {run_id} is now the demo run")


if __name__ == "__main__":
    main()
