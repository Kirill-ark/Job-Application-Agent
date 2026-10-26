"""How errors from the LLM provider become our errors."""

import httpx
import openai

from job_agent.llm import InvalidToolCall, LLMError, _failed


def bad_request(body: dict) -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.example"))
    return openai.BadRequestError("Error code: 400 - " + str(body), response=response, body=body)


def test_the_models_partial_answer_stays_out_of_the_message() -> None:
    # Groq sends a failed JSON answer back, which can hold a visitor's CV.
    error = bad_request(
        {"message": "Failed to generate JSON", "failed_generation": "Maria Muster, Munich"}
    )
    message = str(_failed(error))
    assert "Failed to generate JSON" in message and "status 400" in message
    assert "Maria" not in message


def test_a_tool_call_against_the_schema_can_be_corrected() -> None:
    error = bad_request({"code": "tool_use_failed", "message": "missing properties: 'query'"})
    failed = _failed(error)
    assert isinstance(failed, InvalidToolCall)
    assert str(failed) == "missing properties: 'query'"
    assert not isinstance(_failed(bad_request({"message": "other"})), InvalidToolCall)
    assert isinstance(_failed(bad_request({"message": "other"})), LLMError)
