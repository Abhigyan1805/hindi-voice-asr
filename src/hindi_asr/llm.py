"""Pluggable LLM client for answering a transcribed question.

Selected with ``HINDI_ASR_LLM`` (``anthropic`` | ``openai`` | ``none``), or
automatically from whichever API key is present (``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``). With no key the demo still works and only transcribes.
``OPENAI_BASE_URL`` lets the OpenAI client point at any compatible server
(e.g. a local vLLM/Ollama endpoint).
"""

from __future__ import annotations

import os
from typing import Protocol

SYSTEM = (
    "You answer spoken questions from Indian users. The question was transcribed by a speech "
    "recogniser and may contain recognition errors. Reply in the language of the question "
    "(Hindi in Devanagari, or Hinglish), briefly and helpfully."
)


class LLMClient(Protocol):
    name: str

    def answer(self, question: str) -> str: ...


class AnthropicClient:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("HINDI_ASR_LLM_MODEL", "claude-haiku-4-5-20251001")

    def answer(self, question: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=512, system=SYSTEM, messages=[{"role": "user", "content": question}]
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()


class OpenAIClient:
    name = "openai"

    def __init__(self, model: str | None = None):
        import openai

        self.client = openai.OpenAI()
        self.model = model or os.environ.get("HINDI_ASR_LLM_MODEL", "gpt-4o-mini")

    def answer(self, question: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        )
        return (resp.choices[0].message.content or "").strip()


def get_client() -> LLMClient | None:
    """Return the configured client, or ``None`` when no provider/key is set up."""
    choice = os.environ.get("HINDI_ASR_LLM", "").lower()
    if choice == "none":
        return None
    if choice == "anthropic" or (not choice and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicClient()
    if choice == "openai" or (not choice and os.environ.get("OPENAI_API_KEY")):
        return OpenAIClient()
    return None
