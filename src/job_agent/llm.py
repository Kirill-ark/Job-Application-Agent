"""The one place that talks to the LLM.

Groq has an OpenAI-compatible API, so we use the `openai` SDK with Groq's
address. Moving to another OpenAI-compatible provider (OpenRouter, Gemini,
a local Ollama) means changing GROQ_BASE_URL and the model name.
"""

import os
from collections.abc import Callable

import openai
from dotenv import load_dotenv

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# What the rest of the code needs from an LLM: (system prompt, user prompt) -> text.
# Tests pass a plain function instead of a real client.
Complete = Callable[[str, str], str]


class LLMError(Exception):
    """The LLM is not configured, not reachable, or gave no answer."""


def make_complete(max_tokens: int = 800) -> Complete:
    """Build a `complete` function backed by Groq, configured from the environment."""
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set")
    model = os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    client = openai.OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=30, max_retries=1)

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
            )
        except openai.OpenAIError as error:
            raise LLMError(f"LLM request failed: {error}") from error
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise LLMError("LLM returned an empty answer")
        return text.strip()

    return complete
