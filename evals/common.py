"""Loading the frozen eval cases, shared by all suites."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from job_agent.cv import read_cv
from job_agent.db import connect, init_db
from job_agent.db.queries import upsert_vacancies
from job_agent.skills import Taxonomy

CASES = Path(__file__).parent / "cases"
RESULTS = Path(__file__).parent / "results"
LABELS = ("good", "maybe", "bad")


def load_vacancies() -> list[dict[str, Any]]:
    return json.loads((CASES / "vacancies.json").read_text(encoding="utf-8"))


def load_cvs(taxonomy: Taxonomy) -> dict[str, dict[str, Any]]:
    """The eval CVs by name, as read_cv returns them."""
    return {path.stem: read_cv(path, taxonomy) for path in sorted((CASES / "cvs").glob("*.yaml"))}


def load_match_labels() -> list[dict[str, str]]:
    return yaml.safe_load((CASES / "match_labels.yaml").read_text(encoding="utf-8"))


def database_with_cases(path: Path) -> sqlite3.Connection:
    """A fresh database holding only the frozen vacancies."""
    conn = connect(path)
    init_db(conn)
    upsert_vacancies(conn, load_vacancies())
    return conn
