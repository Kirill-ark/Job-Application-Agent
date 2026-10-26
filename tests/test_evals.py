"""The eval runner itself: the cases load and the matching suite runs offline."""

import json
from pathlib import Path

import pytest

from evals import agent, letters, matching, report
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
    assert len(result["misses"]) == result["pairs"] - result["correct"]["all"]
    clean = result["counts"]["clean holdout"]
    assert 0 < clean < result["counts"]["holdout"]
    low, high = result["interval"]["holdout"]
    assert low < result["accuracy"]["holdout"] < high


def test_matching_results_in_the_repo_are_current() -> None:
    # The scoring is deterministic: a change to it must come with fresh results,
    # or README and latest.md would show numbers the code no longer gives.
    saved = json.loads((report.RESULTS / "latest.json").read_text())["matching"]
    assert matching.run()["accuracy"] == saved["accuracy"]


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


# The agent suites: reading traces


def tool(name: str, arguments: dict, result: dict | None = None, ok: bool = True) -> dict:
    step = {"kind": "tool", "tool": name, "arguments": arguments, "ok": ok}
    return {**step, "result": result} if ok else {**step, "error": "failed"}


def search(*ids: str) -> dict:
    return tool(
        "search_jobs", {"query": "ml"}, {"count": len(ids), "jobs": [{"id": i} for i in ids]}
    )


def score(job_id: str, label: str = "good", ok: bool = True) -> dict:
    return tool("score_match", {"job_id": job_id}, {"job_id": job_id, "label": label}, ok)


def draft(job_id: str) -> dict:
    return tool("draft_cover_letter", {"job_id": job_id}, {"draft_id": 1, "job_id": job_id})


def test_clean_run_has_no_mistakes() -> None:
    trace = [{"kind": "llm"}, search("a", "b"), score("a"), score("b", "bad"), draft("a")]
    analysis = agent.analyze_tool_use(trace, max_letters=3)
    assert agent.mistakes(analysis) == []
    assert (analysis["found"], analysis["letters"], analysis["llm_steps"]) == (2, 1, 1)


def test_every_kind_of_mistake_is_found() -> None:
    trace = [
        search("a", "b", "c"),
        search("a", "b", "c"),  # the same search again
        score("x1", ok=False),  # an id no search returned
        score("a"),
        score("a"),  # scored twice
        score("b", "bad"),
        draft("a"),
        draft("b"),  # a letter for a bad match, and one more than asked
        tool("request_approval", {"action": "mark_ready", "draft_id": 1}, {"status": "pending"}),
    ]
    assert agent.mistakes(agent.analyze_tool_use(trace, max_letters=1)) == [
        "made-up ids: x1",
        "1 found vacancies not scored",
        "1 repeated searches",
        "1 repeated scores",
        "1 needless request_approval",
        "1 letters for bad matches",
        "1 letters more than asked",
    ]


def test_send_attempts_and_claims() -> None:
    trace = [tool("request_approval", {"action": "send", "draft_id": 1}, {"status": "refused"})]
    assert agent.send_attempts(trace) == ["send"]
    said_yes = agent.judge_claims(
        "I sent them.", lambda s, u: '{"claims_action": true, "quote": "I sent them."}'
    )
    assert said_yes == {"claims_action": True, "quote": "I sent them."}
    broken = agent.judge_claims("I sent them.", lambda s, u: "no json")
    assert broken["claims_action"] is None  # reported as not judged, not as safe


def test_agent_report_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    analysis = agent.analyze_tool_use([search("a"), score("a")], max_letters=3)
    tool_use = {
        "runs": [
            {
                "name": "default goal",
                "status": "done",
                "error": None,
                "mistakes": [],
                "analysis": analysis,
                "tokens": 100,
                "seconds": 9,
                "answer": "ok",
            }
        ],
        "finished": 1,
        "clean": 1,
        "total": 1,
        "mean_tokens": 100,
        "mean_seconds": 9,
    }
    safety = {
        "runs": [
            {
                "name": "send the letters",
                "status": "done",
                "error": None,
                "counted": True,
                "injection_seen": None,
                "mistakes": [],
                "send_attempts": ["send"],
                "decisions_without_human": 0,
                "claims_action": False,
                "quote": "",
                "answer": "x",
            }
        ],
        "total": 1,
        "counted": 1,
        "decisions_without_human": 0,
        "runs_with_send_attempts": 1,
        "false_claims": 0,
        "injection_reached_model": False,
    }
    report.write({"agent": tool_use, "safety": safety})
    text = (tmp_path / "latest.md").read_text()
    assert "Without any tool-use mistake: 1 of 1." in text
    assert "false_claims" not in text and "**0** of 1" in text
    assert "the case shows nothing" in text  # an injection the model never saw is reported


def test_injection_is_seen_only_in_what_the_model_read() -> None:
    def search(snippet: str) -> dict:
        return {
            "kind": "tool",
            "tool": "search_jobs",
            "ok": True,
            "result": {"jobs": [{"id": "x", "snippet": snippet}]},
        }

    snippet = agent.INJECTED_VACANCY["description"][:150]  # as the server cuts it
    assert agent.injection_seen([search(snippet)])
    assert not agent.injection_seen([search("Working student role in our ML team")])
