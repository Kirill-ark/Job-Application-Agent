"""Limits that keep the public demo inside Groq's free tier.

- One run at a time: Groq allows 8,000 tokens per minute, and two runs at
  once would slow each other down to a crawl.
- A few runs per IP per day, and a cap for everyone together. A run makes at
  most 10 agent steps and 3 letters, so counting runs also bounds LLM calls.
- The same for reading pasted CVs, which uses the letter model's budget.

IPs are never stored: only an HMAC of the IP with a secret salt, which cannot
be turned back into the IP without the salt.
"""

import hashlib
import hmac
import ipaddress
import logging
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from job_agent.db.queries import count_parses_since, count_runs_since, running_since

logger = logging.getLogger(__name__)

DAY = timedelta(hours=24)
# A run older than this no longer blocks others, even if it never finished.
MAX_RUN_TIME = timedelta(minutes=10)

MESSAGES = {
    "busy": "Someone else is using the agent right now. One run at a time keeps it "
    "within the free LLM limits. Please try again in a minute.",
    "ip_limit": "You have used today's runs of the demo.",
    "daily_limit": "The demo has reached its limit for today.",
    "parse_limit": "You have read enough CVs for today.",
    "parse_daily_limit": "Reading CVs has reached its limit for today.",
}


@dataclass
class Limits:
    runs_per_ip_per_day: int = 3
    # Groq's free tier gives the agent model 200,000 tokens per day and a run costs
    # 10,000-23,000, so about 10 runs a day for everyone fit (found by the evals).
    runs_per_day: int = 10
    # A CV read costs up to 2 x 7,000 tokens of the letter model, which also writes
    # every run's letters (~15,000-20,000 a run): 5 reads leave room for 10 runs.
    parses_per_ip_per_day: int = 3
    parses_per_day: int = 5
    salt: str = ""

    @classmethod
    def from_env(cls) -> "Limits":
        salt = os.environ.get("IP_HASH_SALT", "")
        if not salt:
            logger.warning("IP_HASH_SALT is not set: using a random one, limits reset on restart")
            salt = secrets.token_hex(16)

        def setting(name: str, default: int) -> int:
            return int(os.environ.get(name) or default)

        return cls(
            runs_per_ip_per_day=setting("RUNS_PER_IP_PER_DAY", cls.runs_per_ip_per_day),
            runs_per_day=setting("RUNS_PER_DAY", cls.runs_per_day),
            parses_per_ip_per_day=setting("PARSES_PER_IP_PER_DAY", cls.parses_per_ip_per_day),
            parses_per_day=setting("PARSES_PER_DAY", cls.parses_per_day),
            salt=salt,
        )

    def hash_ip(self, ip: str) -> str:
        """HMAC of the visitor's IP. An IPv6 user owns a whole /64 network and
        could take a new address for every run, so IPv6 counts per /64."""
        try:
            address = ipaddress.ip_address(ip)
        except ValueError:
            address = None
        if address is not None and address.version == 6:
            ip = str(ipaddress.ip_network(f"{ip}/64", strict=False))
        return hmac.new(self.salt.encode(), ip.encode(), hashlib.sha256).hexdigest()

    def refusal(self, conn: sqlite3.Connection, ip_hash: str, now: datetime) -> str | None:
        """Why a new run may not start now ('busy', 'ip_limit', 'daily_limit'), or None."""
        if running_since(conn, (now - MAX_RUN_TIME).isoformat()) is not None:
            return "busy"
        day_ago = (now - DAY).isoformat()
        if count_runs_since(conn, day_ago, ip_hash) >= self.runs_per_ip_per_day:
            return "ip_limit"
        if count_runs_since(conn, day_ago) >= self.runs_per_day:
            return "daily_limit"
        return None

    def parse_refusal(self, conn: sqlite3.Connection, ip_hash: str, now: datetime) -> str | None:
        """Why a CV may not be read now ('busy', 'parse_limit', 'parse_daily_limit'), or None."""
        if running_since(conn, (now - MAX_RUN_TIME).isoformat()) is not None:
            return "busy"  # a run needs the letter model's per-minute budget
        day_ago = (now - DAY).isoformat()
        if count_parses_since(conn, day_ago, ip_hash) >= self.parses_per_ip_per_day:
            return "parse_limit"
        if count_parses_since(conn, day_ago) >= self.parses_per_day:
            return "parse_daily_limit"
        return None
