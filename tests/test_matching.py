import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from job_agent.cv import read_cv
from job_agent.db import connect, init_db
from job_agent.db.queries import upsert_vacancies
from job_agent.llm import LLMError
from job_agent.matching import (
    MatchError,
    combine,
    cv_hash,
    find_skills,
    german_component,
    german_required,
    label,
    level_component,
    location_component,
    score_match,
    skills_component,
)
from job_agent.skills import Skill, Taxonomy

NOW = "2026-10-24T12:00:00+00:00"

CV_YAML = """
name: Test Person
headline: Student
location: Berlin
looking_for: [student, intern]
projects:
  - name: Classifier
    description: ML project
    tech: [Python, PyTorch, sklearn]
    facts: [Trained a model]
other_skills: [Kubernetes]
languages:
  - language: German
    level: B2
"""


@pytest.fixture
def taxonomy() -> Taxonomy:
    return Taxonomy(
        [
            Skill(name="Python", category="programming"),
            Skill(name="PyTorch", category="ml", aliases=["torch"]),
            Skill(name="scikit-learn", category="ml", aliases=["sklearn"]),
            Skill(name="Kubernetes", category="devops", aliases=["k8s"]),
            Skill(name="AWS", category="devops"),
            Skill(name="JavaScript", category="programming", aliases=["JS"]),
            Skill(name="C++", category="programming"),
        ]
    )


@pytest.fixture
def cv(tmp_path: Path, taxonomy: Taxonomy) -> dict[str, Any]:
    path = tmp_path / "cv.yaml"
    path.write_text(CV_YAML)
    return read_cv(path, taxonomy)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "test.db")
    init_db(conn)
    upsert_vacancies(conn, [make_vacancy()])
    yield conn
    conn.close()


def make_vacancy(**changes: Any) -> dict[str, Any]:
    vacancy = {
        "id": "ml-student",
        "title": "Working Student Machine Learning (PyTorch)",
        "company": "Acme GmbH",
        "location": "Berlin",
        "remote": False,
        "url": "https://example.com/ml-student",
        "description": "You will work with Python, scikit-learn and Kubernetes. Good English.",
        "tags": ["Data"],
        "job_types": ["working student"],
        "level": "student",
        "posted_at": NOW,
        "fetched_at": NOW,
    }
    return {**vacancy, **changes}


class FakeLLM:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        if self.fail:
            raise LLMError("rate limit")
        return "Strong fit thanks to PyTorch."


# Skills


def test_find_skills_normalizes_aliases_and_weights_the_title(taxonomy: Taxonomy) -> None:
    vacancy = make_vacancy(description="Experience with sklearn and k8s.")
    assert find_skills(vacancy, taxonomy) == {"PyTorch": 2, "scikit-learn": 1, "Kubernetes": 1}


def test_find_skills_respects_word_boundaries(taxonomy: Taxonomy) -> None:
    vacancy = make_vacancy(title="Engineer", description="We exchange JSON. C++ is a plus.")
    assert find_skills(vacancy, taxonomy) == {"C++": 1}


def test_skills_component(cv: dict[str, Any]) -> None:
    wanted = {"PyTorch": 2, "scikit-learn": 1, "Kubernetes": 1, "AWS": 1}
    value, matched, gaps = skills_component(wanted, cv["skills"])
    # Kubernetes is unverified (0.5), AWS is missing; smoothing adds 0.5 * 2 on top and 2 below
    assert value == pytest.approx((2 + 1 + 0.5 + 1) / (5 + 2))
    assert [m["skill"] for m in matched] == ["PyTorch", "scikit-learn"]
    assert matched[0]["evidence"] == ["p1"]
    assert gaps == ["AWS", "Kubernetes (listed in the CV, but not used in a project or job)"]


def test_vacancy_without_known_skills_is_probably_not_tech(cv: dict[str, Any]) -> None:
    value, matched, gaps = skills_component({}, cv["skills"])
    assert value == 0.25
    assert matched == []
    assert gaps == ["The vacancy names none of the tech skills we know"]


# Level, German, location


@pytest.mark.parametrize(
    ("level", "value"),
    [("student", 1.0), ("intern", 1.0), ("entry", 0.6), ("unknown", 0.5), ("experienced", 0.0)],
)
def test_level_component(level: str, value: float) -> None:
    assert level_component(level, ["student", "intern"]) == value


@pytest.mark.parametrize(
    ("description", "required"),
    [
        ("Fließende Deutschkenntnisse und gute Englischkenntnisse.", "C1"),
        ("Sehr gute Deutschkenntnisse in Wort und Schrift.", "C1"),
        ("You speak fluent German and English.", "C1"),
        ("German (C1) is required.", "C1"),
        ("Good command of German is a plus.", "B2"),
        ("Gute Deutschkenntnisse.", "B2"),
        ("We build ML systems with you and our team in Berlin.", None),
        ("Wir suchen dich für unser Team und du arbeitest mit Python.", "B2"),  # ad in German
    ],
)
def test_german_required(description: str, required: str | None) -> None:
    assert german_required(description) == required


def test_german_component() -> None:
    languages = [{"language": "German", "level": "B2"}]
    assert german_component("Fluent German required.", languages) == (
        0.5,
        "German C1 expected, the CV has B2",
    )
    assert german_component("We speak English.", languages) == (1.0, None)
    assert german_component("Fluent German required.", []) == (
        0.0,
        "German C1 expected, the CV has no German",
    )


def test_location_component() -> None:
    assert location_component(make_vacancy(), "Berlin") == 1.0
    assert location_component(make_vacancy(location="Munich"), "Berlin") == 0.0
    assert location_component(make_vacancy(location="Munich", remote=True), "Berlin") == 1.0
    assert location_component(make_vacancy(location=""), "Berlin") == 0.5
    assert location_component(make_vacancy(), "") == 0.5  # a CV without a city


def test_internship_elsewhere_is_possible_working_student_is_not() -> None:
    assert location_component(make_vacancy(location="Munich", level="intern"), "Berlin") == 0.5
    assert location_component(make_vacancy(location="Munich", level="student"), "Berlin") == 0.0


def test_combine_multiplies_factors() -> None:
    perfect = {"skills": 1.0, "level": 1.0, "german": 1.0, "location": 1.0}
    assert combine(perfect) == 100
    assert combine({**perfect, "level": 0.0}) == 30  # senior role: level floor 0.3
    assert combine({**perfect, "german": 0.0, "location": 0.0}) == 25  # 0.5 * 0.5
    assert combine({**perfect, "skills": 0.5, "level": 0.5}) == 32  # 0.5 * 0.65


def test_one_matched_skill_is_not_a_perfect_fit(cv: dict[str, Any]) -> None:
    value, _, _ = skills_component({"PyTorch": 1}, cv["skills"])
    assert value == pytest.approx(2 / 3)


def test_label_thresholds() -> None:
    assert [label(s) for s in (100, 65, 64, 30, 29, 0)] == [
        "good",
        "good",
        "maybe",
        "maybe",
        "bad",
        "bad",
    ]


# The tool


def test_score_match_end_to_end(
    conn: sqlite3.Connection, cv: dict[str, Any], taxonomy: Taxonomy
) -> None:
    result = score_match(conn, "ml-student", cv, taxonomy, complete=FakeLLM())
    # skills: (PyTorch 2 + Python 1 + scikit-learn 1 + Kubernetes 0.5 + prior 1) / (5 + 2)
    assert result["components"] == {"skills": 0.79, "level": 1.0, "german": 1.0, "location": 1.0}
    assert result["score"] == 79  # all factors are 1.0
    assert result["label"] == "good"
    assert result["reasons"] == "Strong fit thanks to PyTorch."


def test_senior_job_in_german_scores_low(
    conn: sqlite3.Connection, cv: dict[str, Any], taxonomy: Taxonomy
) -> None:
    upsert_vacancies(
        conn,
        [
            make_vacancy(
                id="senior",
                title="Senior Engineer",
                description="Wir suchen dich mit verhandlungssicherem Deutsch für AWS.",
                level="experienced",
                location="Munich",
            )
        ],
    )
    result = score_match(conn, "senior", cv, taxonomy)
    assert result["label"] == "bad"
    assert "Requires professional experience" in result["gaps"]
    assert "AWS" in result["gaps"]


def test_score_is_cached(conn: sqlite3.Connection, cv: dict[str, Any], taxonomy: Taxonomy) -> None:
    llm = FakeLLM()
    first = score_match(conn, "ml-student", cv, taxonomy, complete=llm)
    second = score_match(conn, "ml-student", cv, taxonomy, complete=llm)
    assert first == second
    assert llm.calls == 1


def test_other_cv_is_scored_again(
    conn: sqlite3.Connection, cv: dict[str, Any], taxonomy: Taxonomy
) -> None:
    llm = FakeLLM()
    score_match(conn, "ml-student", cv, taxonomy, complete=llm)
    other_cv = {**cv, "location": "Munich"}
    assert (
        score_match(conn, "ml-student", other_cv, taxonomy, complete=llm)["components"]["location"]
        == 0.0
    )
    assert llm.calls == 2


def test_llm_failure_falls_back_to_template(
    conn: sqlite3.Connection, cv: dict[str, Any], taxonomy: Taxonomy
) -> None:
    result = score_match(conn, "ml-student", cv, taxonomy, complete=FakeLLM(fail=True))
    assert result["score"] == 79
    assert result["reasons"].startswith("Good match (79/100). Matches: PyTorch, Python")


def test_without_llm_uses_template(
    conn: sqlite3.Connection, cv: dict[str, Any], taxonomy: Taxonomy
) -> None:
    result = score_match(conn, "ml-student", cv, taxonomy)
    assert "Gaps: Kubernetes" in result["reasons"]


def test_unknown_vacancy(conn: sqlite3.Connection, cv: dict[str, Any], taxonomy: Taxonomy) -> None:
    with pytest.raises(MatchError, match="search_jobs"):
        score_match(conn, "nope", cv, taxonomy)


def test_cv_hash_ignores_key_order(cv: dict[str, Any]) -> None:
    reordered = dict(reversed(list(cv.items())))
    assert cv_hash(reordered) == cv_hash(cv)
