"""Shared test setup: tests never reach the network or a real LLM.

Fakes are passed everywhere, but a forgotten one must fail loudly instead of
spending the free Groq budget or depending on the job board being up.
"""

import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)  # make_chat/make_complete fail at once

    def refuse(self: socket.socket, address: Any) -> None:
        if self.family == socket.AF_UNIX:  # local only (asyncio uses these internally)
            return original(self, address)
        raise RuntimeError(f"A test tried to connect to {address}")

    original = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", refuse)
