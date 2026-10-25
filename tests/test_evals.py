"""The eval runner itself: the cases load and the matching suite runs offline."""

import json
from pathlib import Path

import pytest

from evals import letters, matching, report
from evals.common import LABELS, load_cvs, load_match_labels, load_vacancies
from job_agent.skills import load_taxonomy


def test_cases_are_consistent() -> None:
    vacancies = {v["id"] for v in load_vacancies()}
    cvs = load_cvs(load_taxonomy())
    for case in load_match_labels():
        assert case["vacancy"] in vacancies
        assert case["cv"] in cvs
        assert case["label"] in LABELS
        assert case["split"] in ("tune", "holdout")
        assert case["why"]


def test_matching_suite_runs_and_adds_up() -> None:
    result = matching.run()
    assert result["pairs"] == len(load_match_labels())
    assert sum(sum(row.values()) for row in result["confusion"].values()) == result["pairs"]
    assert len(result["misses"]) == round(result["pairs"] * (1 - result["accuracy"]["all"]))


def test_report_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    report.write({"matching": matching.run()})
    text = (tmp_path / "latest.md").read_text()
    assert "## Match scoring" in text
    assert "| **good** |" in text
    assert (tmp_path / "latest.json").exists()


# The letters suite, with a fake writer and judge


def fake_writer(system: str, user: str) -> str:
    """The same short letter for every case; Alex and Jonas both have a fact p1."""
    return json.dumps(
        {
            "paragraphs": [
                [{"text": "Dear team,", "kind": "other", "cites": []}],
                [
                    {"text": "I built a project.", "kind": "claim", "cites": ["p1"]},
                    {"text": "I led a team of engineers.", "kind": "claim", "cites": ["p1"]},
                ],
            ]
        }
    )


def fake_judge(system: str, user: str) -> str:
    # Sentence 2 is fine, sentence 3 overstates; sentence 1 is a greeting.
    return json.dumps(
        {
            "verdicts": [
                {"sentence": 1, "verdict": "supported", "reason": "greeting"},
                {"sentence": 2, "verdict": "supported", "reason": "fact p1"},
                {"sentence": 3, "verdict": "unsupported", "reason": "no leadership in the CV"},
            ]
        }
    )


def test_letters_suite_counts_verdicts_and_rule_agreement() -> None:
    result = letters.run(writer=fake_writer, judge_llm=fake_judge)
    cases = result["letters"]
    assert result["sentences"] == 3 * cases
    assert result["verdicts"]["unsupported"] == cases
    assert result["hallucination_rate"] == pytest.approx(1 / 3)
    assert result["letters_without_hallucination"] == 0
    # An overstatement without a skill or number: the rules cannot see it, the judge can.
    assert len(result["rules"]["missed"]) == cases


def test_judge_that_keeps_failing_is_reported() -> None:
    def broken_judge(system: str, user: str) -> str:
        return "Judging now..."

    result = letters.run(writer=fake_writer, judge_llm=broken_judge)
    assert result["letters"] == 0
    assert result["judge_failed"]  # every case is listed, not silently dropped


def test_letters_report_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    report.write({"letters": letters.run(writer=fake_writer, judge_llm=fake_judge)})
    text = (tmp_path / "latest.md").read_text()
    assert "Hallucination rate: 33% of sentences" in text
    assert "I led a team of engineers." in text
