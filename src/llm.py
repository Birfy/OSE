from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_CODEX_BIN = "codex"


class CodexAgentError(RuntimeError):
    """Raised when a Codex agent invocation fails."""


@dataclass
class LLMResponse:
    text: str
    raw: Any


class CodexAgentClient:
    """
    Thin non-interactive Codex agent adapter.

    The rest of the project treats this as a text completion client, but every
    generation is performed by `codex exec`, so trajectory generation, proposal
    creation, scoring, and evaluation all use the Codex agent runtime.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        codex_bin: str | None = None,
        cwd: str | None = None,
        sandbox: str = "read-only",
        timeout_seconds: int = 300,
    ) -> None:
        self.model = model
        self.codex_bin = codex_bin or os.getenv("CODEX_BIN", DEFAULT_CODEX_BIN)
        self.cwd = cwd or os.getcwd()
        self.sandbox = sandbox
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        del max_tokens  # Codex CLI owns token budgeting.
        full_prompt = self._build_prompt(prompt, system=system, json_mode=json_mode)
        output_file = tempfile.NamedTemporaryFile(prefix="codex-agent-", suffix=".txt", delete=False)
        output_path = output_file.name
        output_file.close()

        cmd = [
            self._resolve_codex_bin(),
            "exec",
            "--disable",
            "tool_search",
            "--disable",
            "tool_suggest",
            "--disable",
            "multi_agent",
            "--ephemeral",
            "--model",
            self.model,
            "--sandbox",
            self.sandbox,
            "--skip-git-repo-check",
            "--cd",
            self.cwd,
            "--output-last-message",
            output_path,
            "-",
        ]

        try:
            proc = subprocess.run(
                cmd,
                input=full_prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            text = self._read_output(output_path)
        finally:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass

        if proc.returncode != 0:
            raise CodexAgentError(
                "Codex agent failed with exit code "
                f"{proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if not text.strip():
            text = proc.stdout.strip()
        return LLMResponse(text=text.strip(), raw={"stdout": proc.stdout, "stderr": proc.stderr})

    def run_agent(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
    ) -> list[dict[str, Any]]:
        full_prompt = self._build_prompt(prompt, system=system, json_mode=json_mode)
        cmd = [
            self._resolve_codex_bin(),
            "exec",
            "--disable",
            "tool_search",
            "--disable",
            "tool_suggest",
            "--disable",
            "multi_agent",
            "--ephemeral",
            "--model",
            self.model,
            "--sandbox",
            self.sandbox,
            "--skip-git-repo-check",
            "--json",
            "--cd",
            self.cwd,
            "-",
        ]
        proc = subprocess.run(
            cmd,
            input=full_prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            raise CodexAgentError(
                "Codex agent failed with exit code "
                f"{proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return self._parse_jsonl_events(proc.stdout)

    def _build_prompt(self, prompt: str, *, system: str | None, json_mode: bool) -> str:
        parts = [
            "You are running as a Codex agent subtask for the OffSkillEvo project.",
            "Do not edit files or run commands unless the prompt explicitly asks you to.",
            "Return only the requested final answer.",
        ]
        if json_mode:
            parts.append("Return valid JSON only. Do not wrap it in markdown fences.")
        if system:
            parts.extend(["", "# System Instructions", system])
        parts.extend(["", "# Task", prompt])
        return "\n".join(parts)

    def _resolve_codex_bin(self) -> str:
        resolved = shutil.which(self.codex_bin)
        if not resolved:
            raise CodexAgentError(
                f"Codex CLI not found: {self.codex_bin}. Install Codex or set CODEX_BIN."
            )
        return resolved

    def _read_output(self, output_path: str) -> str:
        try:
            with open(output_path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _parse_jsonl_events(self, text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"type": "raw", "text": line})
        return events


# Backward-compatible name used by the rest of the codebase.
LLMClient = CodexAgentClient


def first_text(response: Any) -> str:
    if isinstance(response, LLMResponse):
        return response.text
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
