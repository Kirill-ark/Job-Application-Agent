"""The one place that talks to the LLM.

Groq has an OpenAI-compatible API, so we use the `openai` SDK with Groq's
address. Moving to another OpenAI-compatible provider (OpenRouter, Gemini,
a local Ollama) means changing GROQ_BASE_URL and the model name.
"""

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import openai
from dotenv import load_dotenv

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq limits tokens per minute per model, not per key, so two models double the budget.
# The agent model must make several tool calls in one step; on Groq gpt-oss makes only one.
DEFAULT_MODEL = "qwen/qwen3.8-27b"
# Letters must follow a nested JSON schema; gpt-oss-20b broke it, gpt-oss-120b does not.
DEFAULT_LETTER_MODEL = "openai/gpt-oss-120b"

# What the rest of the code needs from an LLM: (system prompt, user prompt) -> text.
# Tests pass a plain function instead of a real client.
Complete = Callable[[str, str], str]


class LLMError(Exception):
    """The LLM is not configured, not reachable, or gave no answer."""


def _api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set")
    return api_key


def make_complete(
    max_tokens: int = 800, json_schema: dict[str, Any] | None = None, model: str | None = None
) -> Complete:
    """Build a `complete` function backed by Groq, configured from the environment.

    With `json_schema`, the API itself makes the model answer with JSON that
    matches the schema (structured outputs), so we never parse free text.
    """
    api_key = _api_key()
    model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    # On a 429 (rate limit) the SDK waits as long as Groq's retry-after header says, then retries.
    client = openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=30, max_retries=2)
    extra: dict[str, Any] = {}
    if json_schema is not None:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": json_schema, "strict": True},
        }

    def complete(system: str, user: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,  # includes the model's hidden reasoning
                temperature=0.2,
                reasoning_effort="low",
                **extra,
            )
        except openai.OpenAIError as error:
            raise LLMError(f"LLM request failed: {error}") from error
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise LLMError("LLM returned an empty answer")
        return text.strip()

    return complete


# Tool calling, for the agent loop.


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model wrote it


@dataclass
class ChatReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


# (messages, tools) -> the model's next message. Tests pass a scripted fake.
Chat = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[ChatReply]]


def make_chat(max_tokens: int = 2000, model: str | None = None) -> Chat:
    """Build an async `chat` function for tool calling, backed by Groq."""
    api_key = _api_key()
    model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    # More retries than for single calls: a whole agent run can hit the
    # tokens-per-minute limit several times, and each retry waits it out.
    client = openai.AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=60, max_retries=4)

    async def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatReply:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=0.2,
                reasoning_effort="low",
            )
        except openai.OpenAIError as error:
            raise LLMError(f"LLM request failed: {error}") from error
        message = response.choices[0].message
        usage = response.usage
        return ChatReply(
            content=message.content,
            tool_calls=[
                ToolCall(call.id, call.function.name, call.function.arguments)
                for call in message.tool_calls or []
            ],
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    return chat
