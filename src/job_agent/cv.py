"""Evidence-based CV: loading, fact IDs and the skill index.

Every statement a cover letter may use is a fact with an ID:
  p1, p1.f1, p1.f2  - project 1, its facts
  e1, e1.f1         - job 1, its facts
  ed1               - education entry 1
  lang1             - spoken language 1
  sk                - other (unverified) skills
IDs come from the position in the file, so nobody has to write them by hand.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from job_agent.skills import Taxonomy, load_taxonomy

DEFAULT_CV_PATH = Path(__file__).parents[2] / "data" / "cv.yaml"

YearMonth = Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
Level = Literal["A1", "A2", "B1", "B2", "C1", "C2", "native"]


class CVError(Exception):
    """The CV file is missing or invalid."""


class Strict(BaseModel):
    # A typo in a field name is an error, not a silently ignored field.
    model_config = ConfigDict(extra="forbid")


class Education(Strict):
    degree: str
    institution: str
    start: YearMonth
    end: YearMonth | None = None  # None means still studying


class Job(Strict):
    role: str
    company: str
    start: YearMonth
    end: YearMonth | None = None  # None means current job
    tech: list[str] = []
    facts: list[str] = []


class Project(Strict):
    name: str
    description: str
    tech: list[str] = []
    facts: list[str] = []


class SpokenLanguage(Strict):
    language: str
    level: Level


class CV(Strict):
    name: str
    headline: str
    location: str
    availability: str | None = None
    education: list[Education] = []
    experience: list[Job] = []
    projects: list[Project] = []
    other_skills: list[str] = []
    languages: list[SpokenLanguage] = []


def load_cv(path: Path = DEFAULT_CV_PATH) -> CV:
    """Read and validate a CV file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CV.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise CVError(f"Invalid CV in {path}: {error}") from error


def _period(start: str, end: str | None) -> str:
    return f"{start} - {end or 'present'}"


def fact_table(cv: CV) -> dict[str, str]:
    """Every citable statement in the CV, keyed by its fact ID."""
    facts: dict[str, str] = {}
    for i, project in enumerate(cv.projects, start=1):
        facts[f"p{i}"] = (
            f"Project {project.name}: {project.description} (tech: {', '.join(project.tech)})"
        )
        for j, fact in enumerate(project.facts, start=1):
            facts[f"p{i}.f{j}"] = fact
    for i, job in enumerate(cv.experience, start=1):
        facts[f"e{i}"] = (
            f"{job.role} at {job.company}, {_period(job.start, job.end)} "
            f"(tech: {', '.join(job.tech)})"
        )
        for j, fact in enumerate(job.facts, start=1):
            facts[f"e{i}.f{j}"] = fact
    for i, education in enumerate(cv.education, start=1):
        facts[f"ed{i}"] = (
            f"{education.degree}, {education.institution}, "
            f"{_period(education.start, education.end)}"
        )
    for i, spoken in enumerate(cv.languages, start=1):
        facts[f"lang{i}"] = f"{spoken.language}: {spoken.level}"
    if cv.other_skills:
        facts["sk"] = f"Other skills, not used in a project or job: {', '.join(cv.other_skills)}"
    return facts


def skill_index(cv: CV, taxonomy: Taxonomy) -> dict[str, dict[str, Any]]:
    """Map each canonical skill to its category and the fact IDs that prove it.

    A skill is verified when at least one project or job used it.
    """
    evidence: dict[str, list[str]] = {}
    for i, project in enumerate(cv.projects, start=1):
        for raw in project.tech:
            evidence.setdefault(taxonomy.normalize(raw), []).append(f"p{i}")
    for i, job in enumerate(cv.experience, start=1):
        for raw in job.tech:
            evidence.setdefault(taxonomy.normalize(raw), []).append(f"e{i}")
    for raw in cv.other_skills:
        evidence.setdefault(taxonomy.normalize(raw), [])

    return {
        skill: {
            "category": taxonomy.category(skill),
            "evidence": sorted(set(ids)),
            "verified": bool(ids),
        }
        for skill, ids in sorted(evidence.items())
    }


def read_cv(path: Path = DEFAULT_CV_PATH, taxonomy: Taxonomy | None = None) -> dict[str, Any]:
    """Return the CV as plain data for the agent: profile, facts with IDs and skills.

    This is what the `read_cv` MCP tool returns.
    """
    cv = load_cv(path)
    taxonomy = taxonomy or load_taxonomy()
    return {
        "name": cv.name,
        "headline": cv.headline,
        "location": cv.location,
        "availability": cv.availability,
        "facts": fact_table(cv),
        "skills": skill_index(cv, taxonomy),
    }
