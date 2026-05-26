from __future__ import annotations

import os
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


DEFAULT_MODEL = "gpt-5.4-nano"


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is unavailable."""


class LLMRequestError(RuntimeError):
    """Raised when an LLM provider request fails."""


@dataclass
class LLMResponse:
    text: str
    raw: Any


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, provider: str = "auto") -> None:
        self.model = model
        self.provider = self._resolve_provider(provider)
        self._anthropic_client = None
        if self.provider == "anthropic":
            self._anthropic_client = get_anthropic_client()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        if self.provider == "openai":
            return self._complete_openai(prompt, system=system, max_tokens=max_tokens, json_mode=json_mode)
        return self._complete_anthropic(prompt, system=system, max_tokens=max_tokens)

    def _resolve_provider(self, provider: str) -> str:
        if provider != "auto":
            return provider
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "openai"

    def _complete_openai(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMRequestError("OPENAI_API_KEY is not set.")

        input_items = []
        if system:
            input_items.append({"role": "system", "content": system})
        input_items.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            payload["text"] = {"format": {"type": "json_object"}}

        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMRequestError(f"OpenAI request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMRequestError(f"OpenAI request failed: {exc}") from exc

        return LLMResponse(text=_extract_openai_text(raw), raw=raw)

    def _complete_anthropic(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
    ) -> LLMResponse:
        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(text=first_text(response), raw=response)


def _extract_openai_text(raw: dict[str, Any]) -> str:
    if isinstance(raw.get("output_text"), str):
        return raw["output_text"]
    chunks: list[str] = []
    for item in raw.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"}:
                chunks.append(content.get("text", ""))
    return "".join(chunks)


def get_anthropic_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise MissingDependencyError(
            "The 'anthropic' package is required for LLM calls. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return anthropic.Anthropic()


def first_text(response: Any) -> str:
    if isinstance(response, LLMResponse):
        return response.text
    if isinstance(response, dict):
        return _extract_openai_text(response)
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "")
    return ""


def text_response(text: str, raw: Any | None = None) -> Any:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], raw=raw)


def parse_json_response(raw: str) -> Any:
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :]
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    return json.loads(cleaned.strip())
