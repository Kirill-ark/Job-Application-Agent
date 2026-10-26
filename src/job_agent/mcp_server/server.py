"""The MCP server: the five tools the agent can use.

    search_jobs, read_cv, score_match, draft_cover_letter, request_approval

Deciding on an approval and exporting a letter are NOT tools here. Those are
the human's actions in the web UI, so no agent connected to this server can
approve anything, whatever its prompt says.

One server serves one CV, given by its path. The agent starts a server per
run, so visitors with different CVs never see each other's data, and the agent
cannot switch the CV either: no tool takes a CV argument. For the same reason
request_approval only accepts drafts this server wrote: draft ids are counted
up, and another visitor's draft must not be reachable by guessing its id.

Tool docstrings are what the model reads to decide which tool to call, so
they are written for the model.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

# Modules, not functions: the tools below have the same names as the functions they call.
from job_agent import approvals, jobs, letters, matching
from job_agent import cv as cv_module
from job_agent.db import connect, init_db
from job_agent.llm import Complete
from job_agent.skills import Taxonomy, load_taxonomy

INSTRUCTIONS = """Tools for a student looking for a job in Germany.
Typical order: read_cv, search_jobs, score_match for the results,
draft_cover_letter for the best matches (each draft waits for the user's approval).
You never send anything: the user reviews, approves and sends letters themselves."""

SNIPPET_FOR_MODEL = 150  # characters of the description in search results

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

# Errors we expect and explain to the model, so it can correct its next call.
EXPECTED_ERRORS = (
    approvals.ApprovalError,
    jobs.JobSourceError,
    letters.LetterError,
    matching.MatchError,
    ValueError,
)


@contextmanager
def tool_errors() -> Iterator[None]:
    try:
        yield
    except EXPECTED_ERRORS as error:
        raise ToolError(str(error)) from error


def build_server(
    db_path: Path,
    cv_path: Path,
    reasons_llm: Complete | None = None,
    letter_llm: Complete | None = None,
    taxonomy: Taxonomy | None = None,
    http_client: httpx.Client | None = None,
) -> MCPServer:
    """Create the server for one CV.

    `reasons_llm` explains match scores (a template is used without it);
    `letter_llm` writes letters and must answer with the letter JSON schema.
    Tests pass fakes for both and for the job board client.
    """
    taxonomy = taxonomy or load_taxonomy()
    cv = cv_module.read_cv(cv_path, taxonomy)  # an invalid CV stops the server at once

    @contextmanager
    def database() -> Iterator[sqlite3.Connection]:
        # A connection per call: tools may run in worker threads, and a
        # sqlite3 connection must stay in the thread that opened it.
        conn = connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    with database() as conn:
        init_db(conn)

    server = MCPServer("job-agent", instructions=INSTRUCTIONS)
    own_drafts: set[int] = set()  # drafts written by this server, i.e. in this run

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
    def search_jobs(
        query: str, location: str | None = None, level: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        """Search current job vacancies in Germany (Arbeitnow job board).

        Args:
            query: What the job is about, e.g. "machine learning" or "data engineer".
                Do not put seniority or city here.
            location: City, e.g. "Berlin". Remote jobs match any location.
            level: Ranks this seniority higher: student, intern, entry or experienced.
            limit: Maximum number of results, 1-20.

        Returns short vacancies, best first. Use their `id` with score_match.
        """
        with tool_errors(), database() as conn:
            found = jobs.search_jobs(
                conn, query, location, max(1, min(limit, 20)), level, client=http_client
            )
        # Short on purpose: the model only needs enough to pick what to score.
        # URLs and dates stay in the database for the UI.
        short = [
            {
                "id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "remote": job["remote"],
                "level": job["level"],
                "snippet": job["snippet"][:SNIPPET_FOR_MODEL],
            }
            for job in found
        ]
        return {"count": len(short), "jobs": short}

    @server.tool(annotations=READ_ONLY)
    def read_cv() -> dict[str, Any]:
        """Read the user's CV: profile, languages, skills and citable facts.

        Every fact has an ID (p1.f2 = project 1, fact 2; e1 = job 1; ed1 =
        education; lang1 = language; av = availability). Verified skills were
        used in a project or job; unverified ones are only listed.
        """
        skills = cv["skills"]
        return {
            "name": cv["name"],
            "headline": cv["headline"],
            "location": cv["location"],
            "availability": cv["availability"],
            "looking_for": cv["looking_for"],
            "languages": cv["languages"],
            "verified_skills": [s for s, info in skills.items() if info["verified"]],
            "unverified_skills": [s for s, info in skills.items() if not info["verified"]],
            "facts": cv["facts"],
        }

    @server.tool(annotations=READ_ONLY)
    def score_match(job_id: str) -> dict[str, Any]:
        """Score how well one vacancy fits the CV, from 0 to 100.

        Args:
            job_id: The `id` of a vacancy returned by search_jobs.

        Returns the score, a label (good >= 65, maybe >= 30, bad), matched
        skills, gaps and a short explanation. Results are cached.
        """
        with tool_errors(), database() as conn:
            result = matching.score_match(conn, job_id, cv, taxonomy, complete=reasons_llm)
        return {
            "job_id": result["job_id"],
            "title": result["title"],
            "company": result["company"],
            "score": result["score"],
            "label": result["label"],
            "matched": [m["skill"] for m in result["matched"]],
            "gaps": result["gaps"],
            "reasons": result["reasons"],
        }

    @server.tool()
    def draft_cover_letter(job_id: str) -> dict[str, Any]:
        """Write a cover letter draft for one vacancy, using only facts from the CV.

        Args:
            job_id: The `id` of a vacancy returned by search_jobs. Score it first.

        Every sentence is checked against the CV. `grounded` is false when
        problems are left; they are listed in `issues` for the user. The draft
        is saved and put into the user's approval queue (mark_ready), never
        sent. Each call makes a new draft, so call it once per vacancy.
        """
        if letter_llm is None:
            raise ToolError("Letters are not available: the LLM is not configured")
        with tool_errors(), database() as conn:
            result = letters.draft_cover_letter(conn, job_id, cv, taxonomy, letter_llm)
            own_drafts.add(result["draft_id"])
            # Every draft waits for the user: queued here in code, so the agent
            # cannot forget it and does not spend a step on it.
            approval = approvals.request_approval(conn, "mark_ready", result["draft_id"])
        return {
            "draft_id": result["draft_id"],
            "job_id": result["job_id"],
            "title": result["title"],
            "company": result["company"],
            "letter": result["plain_text"],
            "grounded": result["grounded"],
            "issues": result["issues"],
            "approval": approval["status"],
        }

    @server.tool()
    def request_approval(action: str, draft_id: int) -> dict[str, Any]:
        """Ask the user to approve an action on a draft. You cannot approve it yourself.

        Args:
            action: "mark_ready" (the letter is final; drafts are already queued for
                this) or "export" (give the user the text; only after mark_ready was
                approved). Sending, emailing or applying is never possible and will
                be refused.
            draft_id: The `draft_id` returned by draft_cover_letter.

        The request waits for the user's decision in the UI. Do not ask again
        and do not wait for it: finish your answer and tell the user.
        """
        if draft_id not in own_drafts:
            raise ToolError(f"Unknown draft {draft_id}. Use a draft_id from draft_cover_letter.")
        with tool_errors(), database() as conn:
            result = approvals.request_approval(conn, action, draft_id)
        # The issues were already returned by draft_cover_letter.
        return {k: v for k, v in result.items() if k != "issues"}

    return server
