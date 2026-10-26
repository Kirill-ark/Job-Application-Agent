"""Suite 2: do the cover letters say only what the CV backs up?

For each letter case a real letter is written (the same code path as the
draft_cover_letter tool), then every sentence is judged twice:

- by our rules (letters.check_sentence), as in production;
- by an LLM judge of a different model family than the writer, which also
  catches what rules cannot: a fact retold with a changed meaning.

So the suite measures two things: how often letters still contain made-up
content, and how good the rules are compared with the judge (false alarms
and misses of the rules).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from evals.common import CASES, database_with_cases, load_cvs
from job_agent.db.queries import get_vacancy
from job_agent.letters import (
    LETTER_SCHEMA,
    Letter,
    LetterError,
    build_prompt,
    check_sentence,
    write_letter,
)
from job_agent.llm import DEFAULT_LETTER_MODEL, Complete, LLMError, make_complete
from job_agent.matching import score_match
from job_agent.skills import Taxonomy, load_taxonomy

JUDGE_MODEL = "qwen/qwen3.8-27b"  # not the writer's family, so it does not go easy on it
JUDGE_ATTEMPTS = 3  # Qwen sometimes answers in prose; Groq then rejects the JSON
# Groq allows Qwen only 1,000 output tokens per minute and refuses a request whose
# max_tokens could exceed that, so the judge keeps its answers short.
JUDGE_MAX_TOKENS = 900

VERDICTS = ("supported", "wrong_citation", "partly", "unsupported")
NOT_IN_CV = ("partly", "unsupported")  # content the CV does not back: a hallucination

JUDGE_PROMPT = """You check a cover letter sentence by sentence against the applicant's CV.
Answer only with JSON in the given schema, one verdict for every sentence.

Verdicts:
- supported: everything the sentence says about the applicant follows from the
  cited facts. A sentence that says nothing about the applicant (greeting, interest
  in the company, naming the position, closing) is supported.
- wrong_citation: true according to the CV, but the cited facts are not the ones
  that show it.
- partly: based on the CV, but adds or overstates something (a result, a role like
  "led", a level like "expert", a detail that no fact gives).
- unsupported: a claim about the applicant that no CV fact backs.

Keep every reason under 15 words.
Judge strictly: the applicant will send this letter to a real employer."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "reason": {"type": "string"},
                },
                "required": ["sentence", "verdict", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def judge_input(letter: Letter, cv: dict[str, Any], vacancy: dict[str, Any]) -> str:
    sentences = [s for paragraph in letter.paragraphs for s in paragraph]
    lines = [
        f"Position: {vacancy['title']} at {vacancy['company']}",
        "CV facts:",
        json.dumps(cv["facts"], ensure_ascii=False, indent=1),
        "Sentences:",
    ]
    for number, sentence in enumerate(sentences, start=1):
        lines.append(f"{number}. {json.dumps(sentence.text)} cites: {sentence.cites or 'nothing'}")
    return "\n".join(lines)


def judge(letter: Letter, cv: dict[str, Any], vacancy: dict[str, Any], llm: Complete) -> list[dict]:
    """The judge's verdicts in sentence order; raises LLMError after JUDGE_ATTEMPTS."""
    count = sum(len(paragraph) for paragraph in letter.paragraphs)
    error: Exception | None = None
    for _ in range(JUDGE_ATTEMPTS):
        try:
            verdicts = json.loads(llm(JUDGE_PROMPT, judge_input(letter, cv, vacancy)))["verdicts"]
        except (LLMError, json.JSONDecodeError, KeyError) as failure:
            error = failure
            continue
        by_number = {v["sentence"]: v for v in verdicts}
        if set(by_number) == set(range(1, count + 1)):
            return [by_number[n] for n in range(1, count + 1)]
        error = LLMError(f"the judge returned {len(by_number)} verdicts for {count} sentences")
    raise LLMError(f"Judge failed: {error}")


def letter_rows(
    case: dict[str, str],
    cv: dict[str, Any],
    taxonomy: Taxonomy,
    writer: Complete,
    judge_llm: Complete,
    conn: Any,
) -> list[dict[str, Any]]:
    """Write one letter and judge it: one row per sentence."""
    vacancy = get_vacancy(conn, case["vacancy"])
    match = score_match(conn, case["vacancy"], cv, taxonomy)
    attempts: list[list[dict[str, str]]] = []
    letter, _ = write_letter(
        build_prompt(vacancy, match, cv), cv, taxonomy, vacancy["title"], writer, attempts
    )
    verdicts = judge(letter, cv, vacancy, judge_llm)
    sentences = [s for paragraph in letter.paragraphs for s in paragraph]
    return [
        {
            "case": f"{vacancy['title'][:45]} ({case['cv']})",
            "text": sentence.text,
            "kind": sentence.kind,
            "cites": sentence.cites,
            "rule_problems": check_sentence(sentence, cv, taxonomy, vacancy["title"]),
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
            # Problems the rules found in the first attempt, before the repair.
            "first_attempt_problems": len(attempts[0]) if attempts else 0,
        }
        for sentence, verdict in zip(sentences, verdicts, strict=True)
    ]


def summarize(rows: list[dict[str, Any]], letters: int, failed: list[str]) -> dict[str, Any]:
    total = len(rows)

    def share(count: int) -> float | None:
        return count / total if total else None

    judged_bad = [r for r in rows if r["verdict"] in NOT_IN_CV]
    first_attempts = {r["case"]: r.get("first_attempt_problems", 0) for r in rows}
    flagged = [r for r in rows if r["rule_problems"]]
    return {
        "letters": letters,
        "judge_failed": failed,
        "sentences": total,
        "claims": sum(r["kind"] == "claim" for r in rows),
        "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in VERDICTS},
        "hallucination_rate": share(len(judged_bad)),
        "letters_without_hallucination": letters - len({r["case"] for r in judged_bad}),
        # The rules compared with the judge, sentence by sentence.
        "rules": {
            # The final letters are already repaired, so few flags remain there;
            # what the rules did is seen in the first attempts.
            "repaired_letters": sum(n > 0 for n in first_attempts.values()),
            "first_attempt_problems": sum(first_attempts.values()),
            "flagged": len(flagged),
            "caught": sum(r["verdict"] in NOT_IN_CV for r in flagged),
            "false_alarms": [r for r in flagged if r["verdict"] == "supported"],
            "missed": [r for r in judged_bad if not r["rule_problems"]],
        },
        "judged_bad": judged_bad,
        "rows": rows,
    }


def run(writer: Complete | None = None, judge_llm: Complete | None = None) -> dict[str, Any]:
    """Tests pass fake writer and judge functions; by default both are real Groq models."""
    letter_model = os.environ.get("GROQ_LETTER_MODEL") or DEFAULT_LETTER_MODEL
    writer = writer or make_complete(2500, LETTER_SCHEMA, model=letter_model)
    judge_llm = judge_llm or make_complete(JUDGE_MAX_TOKENS, JUDGE_SCHEMA, model=JUDGE_MODEL)
    taxonomy = load_taxonomy()
    cvs = load_cvs(taxonomy)
    cases = yaml.safe_load((CASES / "letter_cases.yaml").read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    with tempfile.TemporaryDirectory() as folder:
        conn = database_with_cases(Path(folder) / "evals.db")
        for case in cases:
            print(f"  letter: {case['vacancy'][:50]} ({case['cv']})")
            try:
                rows += letter_rows(case, cvs[case["cv"]], taxonomy, writer, judge_llm, conn)
            except (LLMError, LetterError) as error:
                failed.append(f"{case['vacancy']} ({case['cv']}): {str(error)[:400]}")
        conn.close()
    return summarize(rows, len(cases) - len(failed), failed)
