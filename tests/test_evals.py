"""The eval runner itself: the cases load and the matching suite runs offline."""

from pathlib import Path

import pytest

from evals import matching, report
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
