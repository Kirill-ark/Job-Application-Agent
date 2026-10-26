"""The human-in-the-loop gate.

The agent can only ASK: `request_approval` is an MCP tool and writes a
pending request. Only the human DECIDES: `decide` is called by the web API
when the user clicks a button, and it is never exposed to the agent. So even
an agent that was tricked (for example by instructions hidden in a vacancy)
cannot approve anything: it has no tool for that.

Actions:
  mark_ready - the letter is final; the draft becomes 'approved'
  export     - hand out the letter text; only for an approved draft
Anything else ('send', 'email', ...) is refused at once and logged. The agent
never sends anything; the human sends the letter themselves.

Every request and decision is a new row in the `approvals` table, which the
database keeps append-only.
"""

import sqlite3
from datetime import UTC, datetime
from typing import Any, Literal

from job_agent.db.queries import (
    add_approval_event,
    approval_requests,
    get_approval_request,
    get_draft,
    set_draft_status,
)
from job_agent.letters import strip_citations

ACTIONS = ("mark_ready", "export")
REFUSAL = (
    "Refused: the agent never sends anything or contacts anyone. "
    "Allowed actions are mark_ready and export; the user sends the letter themselves."
)


class ApprovalError(Exception):
    """The request or decision is not allowed (unknown draft, decided twice, ...)."""


def _now(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).isoformat()


def _summary(request: dict[str, Any], message: str, issues: list[Any]) -> dict[str, Any]:
    return {
        "request_id": request["id"],
        "draft_id": request["draft_id"],
        "action": request["action"],
        "status": request["status"],
        "message": message,
        "issues": issues,
    }


def request_approval(
    conn: sqlite3.Connection, action: str, draft_id: int, now: datetime | None = None
) -> dict[str, Any]:
    """Ask the human to approve an action on a draft. Never decides anything itself.

    Asking again while the same request is pending returns that request.
    """
    draft = get_draft(conn, draft_id)
    if draft is None:
        raise ApprovalError(f"Unknown draft {draft_id}. Use a draft_id from draft_cover_letter.")

    if action not in ACTIONS:
        with conn:
            request_id = add_approval_event(
                conn,
                {
                    "draft_id": draft_id,
                    # The log is kept forever and is never erased, so the agent's
                    # free text (it could hold CV data) is not stored, only "other".
                    "action": "other",
                    "event": "refused",
                    "comment": REFUSAL,
                    "created_at": _now(now),
                },
            )
        return _summary(get_approval_request(conn, request_id), REFUSAL, [])

    if draft["status"] == "rejected":
        raise ApprovalError(f"Draft {draft_id} was rejected. Write a new draft instead.")
    if action == "mark_ready" and draft["status"] == "approved":
        raise ApprovalError(f"Draft {draft_id} is already approved.")
    if action == "export" and draft["status"] != "approved":
        raise ApprovalError(f"Draft {draft_id} must be approved (mark_ready) before export.")

    for request in approval_requests(conn, draft_id):
        if request["action"] == action and request["status"] == "pending":
            return _summary(request, "Already waiting for the user's decision.", draft["issues"])

    comment = None
    if draft["issues"]:
        comment = f"The draft has {len(draft['issues'])} unresolved groundedness issue(s)."
    with conn:
        request_id = add_approval_event(
            conn,
            {
                "draft_id": draft_id,
                "action": action,
                "event": "requested",
                "comment": comment,
                "created_at": _now(now),
            },
        )
    message = "Waiting for the user's decision in the UI. Do not ask again."
    return _summary(get_approval_request(conn, request_id), message, draft["issues"])


def decide(
    conn: sqlite3.Connection,
    request_id: int,
    decision: Literal["approved", "rejected"],
    comment: str | None = None,
    edited_text: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record the human's decision. Called by the web API only, never by the agent.

    `edited_text` is the user's final version of the letter (mark_ready only).
    """
    if decision not in ("approved", "rejected"):
        raise ApprovalError(f"Unknown decision '{decision}'")
    request = get_approval_request(conn, request_id)
    if request is None:
        raise ApprovalError(f"Unknown approval request {request_id}")
    if request["status"] != "pending":
        raise ApprovalError(f"Request {request_id} is already {request['status']}")
    if edited_text is not None and not (
        decision == "approved" and request["action"] == "mark_ready"
    ):
        raise ApprovalError("edited_text can only be given when approving mark_ready")
    if edited_text is not None and not edited_text.strip():
        raise ApprovalError("edited_text is empty")

    timestamp = _now(now)
    try:
        with conn:  # the log row and the draft status change together or not at all
            add_approval_event(
                conn,
                {
                    "request_id": request_id,
                    "draft_id": request["draft_id"],
                    "action": request["action"],
                    "event": decision,
                    "comment": comment,
                    "created_at": timestamp,
                },
            )
            if request["action"] == "mark_ready":
                set_draft_status(conn, request["draft_id"], decision, timestamp, edited_text)
    except sqlite3.IntegrityError as error:
        # Two clicks at once: both passed the check above, the database lets one win.
        raise ApprovalError(f"Request {request_id} is already decided") from error
    return get_approval_request(conn, request_id)


def export_letter(conn: sqlite3.Connection, draft_id: int) -> str:
    """The final letter text without [p1.f2] citations, after an approved export.

    The user's edit wins over the generated text.
    """
    draft = get_draft(conn, draft_id)
    if draft is None:
        raise ApprovalError(f"Unknown draft {draft_id}")
    approved = any(
        r["action"] == "export" and r["status"] == "approved"
        for r in approval_requests(conn, draft_id)
    )
    if draft["status"] != "approved" or not approved:
        raise ApprovalError(f"Export of draft {draft_id} has not been approved by the user")
    return draft["edited_text"] or strip_citations(draft["text"])
