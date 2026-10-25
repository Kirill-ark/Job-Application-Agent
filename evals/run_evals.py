"""Run the evals: python -m evals.run_evals [--suite matching|all].

Results go to evals/results/latest.md (tables for the README) and latest.json.
"""

import argparse

from dotenv import load_dotenv

from evals import matching, report
from job_agent.db import REPO_ROOT

SUITES = {"matching": matching.run}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")

    chosen = SUITES if args.suite == "all" else {args.suite: SUITES[args.suite]}
    results = {}
    for name, run in chosen.items():
        print(f"Running {name}...")
        results[name] = run()
    report.write(results)
    print("Wrote evals/results/latest.md")


if __name__ == "__main__":
    main()
