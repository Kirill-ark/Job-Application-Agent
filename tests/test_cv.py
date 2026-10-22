from pathlib import Path

import pytest

from job_agent.cv import CVError, fact_table, load_cv, read_cv, skill_index
from job_agent.skills import Skill, Taxonomy, load_taxonomy

MINIMAL_CV = """
name: Test Person
headline: Student
location: Berlin
"""

FULL_CV = (
    MINIMAL_CV
    + """
education:
  - degree: B.Sc. CS
    institution: Test University
    start: 2024-10
experience:
  - role: Working Student
    company: Test GmbH
    start: 2025-01
    end: 2025-06
    tech: [Python]
    facts: [Wrote ETL scripts]
projects:
  - name: First
    description: First project
    tech: [Python, sklearn]
    facts: [Fact one, Fact two]
  - name: Second
    description: Second project
    facts: [Only fact]
other_skills: [Kubernetes]
languages:
  - language: German
    level: B2
"""
)


@pytest.fixture
def taxonomy() -> Taxonomy:
    return Taxonomy(
        [
            Skill(name="Python", category="programming"),
            Skill(name="scikit-learn", category="ml", aliases=["sklearn"]),
            Skill(name="Kubernetes", category="devops"),
        ]
    )


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cv.yaml"
    path.write_text(text)
    return path


def test_fact_ids_follow_position_in_file(tmp_path: Path) -> None:
    facts = fact_table(load_cv(write(tmp_path, FULL_CV)))
    assert facts["p1.f1"] == "Fact one"
    assert facts["p1.f2"] == "Fact two"
    assert facts["p2.f1"] == "Only fact"
    assert facts["e1.f1"] == "Wrote ETL scripts"
    assert facts["lang1"] == "German: B2"
    assert "Kubernetes" in facts["sk"]


def test_missing_end_date_means_present(tmp_path: Path) -> None:
    facts = fact_table(load_cv(write(tmp_path, FULL_CV)))
    assert facts["ed1"].endswith("2024-10 - present")
    assert facts["e1"].startswith("Working Student at Test GmbH, 2025-01 - 2025-06")


def test_skill_index_collects_evidence(tmp_path: Path, taxonomy: Taxonomy) -> None:
    skills = skill_index(load_cv(write(tmp_path, FULL_CV)), taxonomy)
    assert skills["Python"] == {
        "category": "programming",
        "evidence": ["e1", "p1"],
        "verified": True,
    }


def test_skill_index_normalizes_aliases(tmp_path: Path, taxonomy: Taxonomy) -> None:
    skills = skill_index(load_cv(write(tmp_path, FULL_CV)), taxonomy)
    assert "sklearn" not in skills
    assert skills["scikit-learn"]["evidence"] == ["p1"]


def test_other_skills_are_unverified(tmp_path: Path, taxonomy: Taxonomy) -> None:
    skills = skill_index(load_cv(write(tmp_path, FULL_CV)), taxonomy)
    assert skills["Kubernetes"] == {"category": "devops", "evidence": [], "verified": False}


def test_minimal_cv_is_valid(tmp_path: Path, taxonomy: Taxonomy) -> None:
    result = read_cv(write(tmp_path, MINIMAL_CV), taxonomy)
    assert result["name"] == "Test Person"
    assert result["facts"] == {}
    assert result["skills"] == {}


def test_missing_required_field_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CVError, match="headline"):
        load_cv(write(tmp_path, "name: Test\nlocation: Berlin\n"))


def test_unknown_field_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CVError, match="skils"):
        load_cv(write(tmp_path, MINIMAL_CV + "skils: [Python]\n"))


def test_bad_date_is_an_error(tmp_path: Path) -> None:
    bad = MINIMAL_CV + "education:\n  - degree: X\n    institution: Y\n    start: October 2024\n"
    with pytest.raises(CVError):
        load_cv(write(tmp_path, bad))


def test_broken_yaml_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CVError):
        load_cv(write(tmp_path, "name: [unclosed\n"))


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CVError):
        load_cv(tmp_path / "nope.yaml")


def test_example_cv_loads_and_every_skill_is_in_the_vocabulary() -> None:
    result = read_cv()
    assert result["name"] == "Alex Example"
    assert "p1.f1" in result["facts"]
    unknown = [name for name, skill in result["skills"].items() if skill["category"] == "other"]
    assert unknown == []
    taxonomy = load_taxonomy()
    assert result["skills"]["scikit-learn"]["evidence"] == ["p2"]
    assert taxonomy.category("scikit-learn") == "ml"
