"""Run the evals: python -m evals.run_evals [--suite matching|letters|agent|safety|all].

matching needs nothing. The others call Groq (GROQ_API_KEY): letters takes a few
minutes, agent and safety about ten each (full agent runs within the free limits).

Results go to evals/results/latest.md (tables for the README) and latest.json.
"""

import argparse

from dotenv import load_dotenv

from evals import agent, letters, matching, report
from job_agent.db import REPO_ROOT

SUITES = {
    "matching": matching.run,
    "letters": letters.run,
    "agent": agent.run_tool_use,
    "safety": agent.run_safety,
}


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
