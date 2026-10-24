"""Skill vocabulary: canonical skill names, their aliases and categories.

CVs and vacancies are both normalized to the same canonical names,
so the overlap between them can be computed with plain set operations.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

DEFAULT_SKILLS_PATH = Path(__file__).parents[2] / "data" / "skills.yaml"

Category = Literal["programming", "ml", "data", "web", "devops", "tools"]


class TaxonomyError(Exception):
    """The skill vocabulary file is missing or invalid."""


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    category: Category
    aliases: list[str] = []
    implies: list[str] = []  # broader skills this one proves


class Taxonomy:
    """Looks up skills by name or alias, case-insensitively."""

    def __init__(self, skills: list[Skill]) -> None:
        self.skills = {skill.name: skill for skill in skills}
        self._by_alias: dict[str, str] = {}
        for skill in skills:
            for alias in [skill.name, *skill.aliases]:
                key = alias.lower()
                if key in self._by_alias:
                    raise TaxonomyError(
                        f"'{alias}' is used by both {self._by_alias[key]} and {skill.name}"
                    )
                self._by_alias[key] = skill.name
        for skill in skills:
            for implied in skill.implies:
                if implied not in self.skills:
                    raise TaxonomyError(f"{skill.name} implies unknown skill '{implied}'")

    def normalize(self, raw: str) -> str:
        """Return the canonical name for a skill, or the stripped input if it is unknown."""
        cleaned = raw.strip()
        return self._by_alias.get(cleaned.lower(), cleaned)

    def implied(self, name: str) -> list[str]:
        """All broader skills a skill proves, following chains (PyTorch -> Deep Learning ->
        Machine Learning). Unknown skills imply nothing."""
        found: list[str] = []
        todo = list(self.skills[name].implies) if name in self.skills else []
        while todo:
            implied = todo.pop(0)
            if implied not in found and implied != name:
                found.append(implied)
                todo.extend(self.skills[implied].implies)
        return found

    def category(self, name: str) -> str:
        """Return the category of a canonical skill, or 'other' if it is unknown."""
        skill = self.skills.get(name)
        return skill.category if skill else "other"


def load_taxonomy(path: Path = DEFAULT_SKILLS_PATH) -> Taxonomy:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        skills = [Skill.model_validate(item) for item in raw]
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as error:
        raise TaxonomyError(f"Invalid skill vocabulary in {path}: {error}") from error
    return Taxonomy(skills)
