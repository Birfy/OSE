from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .llm import DEFAULT_MODEL, LLMClient, first_text, text_response


@dataclass
class TrajectoryStep:
    step_id: int
    state: str
    skills_active: list[str]
    action: str
    action_type: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    observation: str = ""
    success: bool = True
    failure_reason: str | None = None


@dataclass
class Trajectory:
    task_id: str
    task_description: str
    domain: str
    skills_used: list[str]
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_success: bool = False
    error_log: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "domain": self.domain,
            "skills_used": self.skills_used,
            "final_success": self.final_success,
            "error_log": self.error_log,
            "steps": [asdict(step) for step in self.steps],
        }


class TrajectoryGenerator:
    def __init__(
        self,
        skill_library_path: str,
        model: str = DEFAULT_MODEL,
        max_steps: int = 20,
        dry_run: bool = False,
    ) -> None:
        self.skill_lib = Path(skill_library_path)
        self.model = model
        self.max_steps = max_steps
        self.dry_run = dry_run
        self.client = None if dry_run else LLMClient(model)

    def load_skill(self, skill_name: str) -> str:
        skill_path = self.skill_lib / skill_name / "SKILL.md"
        return skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""

    def build_system_prompt(self, task: dict[str, Any], skills: list[str]) -> str:
        skill_contents = []
        for skill in skills:
            content = self.load_skill(skill)
            if content:
                skill_contents.append(f"## Skill: {skill}\n{content}")
        skills_block = "\n\n".join(skill_contents) if skill_contents else "No skills loaded."
        return f"""You are a capable agent solving real-world tasks.

# Available Skills
{skills_block}

# Task
{task["description"]}

# Instructions
- Follow the skill procedures when applicable.
- Record concrete actions, not hidden reasoning.
- If a skill's trigger condition matches, use it explicitly.
- Report failures with specific error details.
"""

    def run_task(self, task: dict[str, Any], skills: list[str]) -> Trajectory:
        if self.dry_run:
            return self._run_task_dry(task, skills)

        trajectory = Trajectory(
            task_id=str(task["id"]),
            task_description=task["description"],
            domain=task.get("domain", "unknown"),
            skills_used=skills,
        )
        system_prompt = self.build_system_prompt(task, skills)

        for step_id in range(self.max_steps):
            llm_response = self.client.complete(
                task["description"],
                system=system_prompt,
                max_tokens=4096,
            )
            response = text_response(llm_response.text, llm_response.raw)
            step = self._parse_response_to_step(step_id, response, skills)
            trajectory.steps.append(step)
            if step.failure_reason:
                trajectory.error_log.append(step.failure_reason)

            trajectory.final_success = self._check_task_success(task, trajectory.steps)
            break

        return trajectory

    def _run_task_dry(self, task: dict[str, Any], skills: list[str]) -> Trajectory:
        step = TrajectoryStep(
            step_id=0,
            state="dry-run",
            skills_active=skills,
            action=f"Apply skills {skills or ['none']} to task: {task['description']}",
            action_type="text",
            observation="Dry-run mode does not call external APIs.",
            success=True,
        )
        return Trajectory(
            task_id=str(task["id"]),
            task_description=task["description"],
            domain=task.get("domain", "unknown"),
            skills_used=skills,
            steps=[step],
            final_success=True,
            error_log=[],
        )

    def _parse_response_to_step(
        self, step_id: int, response: Any, active_skills: list[str]
    ) -> TrajectoryStep:
        action_text = ""
        action_type = "text"
        tool_name = None
        tool_input = None

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                action_text = getattr(block, "text", "")
                action_type = "text"
            elif block_type == "tool_use":
                tool_name = getattr(block, "name", None)
                tool_input = getattr(block, "input", None)
                action_text = f"[{tool_name}] {json.dumps(tool_input, ensure_ascii=False)}"
                action_type = "tool_use"

        if not action_text:
            action_text = first_text(response)

        failure_reason = self._extract_failure_reason(action_text)
        return TrajectoryStep(
            step_id=step_id,
            state=f"Step {step_id}",
            skills_active=active_skills,
            action=action_text,
            action_type=action_type,
            tool_name=tool_name,
            tool_input=tool_input,
            success=failure_reason is None,
            failure_reason=failure_reason,
        )

    def _extract_failure_reason(self, text: str) -> str | None:
        lowered = text.lower()
        if "keyerror" in lowered:
            return "MISSING_KEY: data field missing"
        if "typeerror" in lowered:
            return "TYPE_MISMATCH: incompatible value type"
        if "filenotfound" in lowered or "file not found" in lowered:
            return "MISSING_FILE: invalid or absent file path"
        if "timeout" in lowered:
            return "TIMEOUT: execution exceeded time limit"
        fail_keywords = ("error", "failed", "exception", "traceback", "cannot")
        if any(keyword in lowered for keyword in fail_keywords):
            return f"UNKNOWN: {text[:200]}"
        return None

    def _get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "python_repl",
                "description": "Execute Python code in a short-lived sandbox process.",
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
            {
                "name": "read_file",
                "description": "Read a UTF-8 text file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        ]

    def _execute_tool_calls(self, response: Any) -> list[dict[str, Any]]:
        results = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_use_id = getattr(block, "id", "")
            name = getattr(block, "name", "")
            tool_input = getattr(block, "input", {}) or {}
            try:
                if name == "python_repl":
                    content = self._run_python(tool_input.get("code", ""))
                elif name == "read_file":
                    content = Path(tool_input["path"]).read_text(encoding="utf-8")
                else:
                    content = f"Unsupported tool: {name}"
            except Exception as exc:
                content = f"{exc.__class__.__name__}: {exc}"
            results.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": content})
        return results

    def _run_python(self, code: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                ["/home/admin/.local/bin/python3.11", "-c", code],
                cwd=tmpdir,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        output = (proc.stdout + proc.stderr).strip()
        return output[:4000] if output else f"Process exited with code {proc.returncode}"

    def _attach_observation(self, step: TrajectoryStep, tool_results: list[dict[str, Any]]) -> None:
        step.observation = "\n".join(str(result.get("content", "")) for result in tool_results)
        failure = self._extract_failure_reason(step.observation)
        if failure:
            step.success = False
            step.failure_reason = failure

    def _check_task_success(self, task: dict[str, Any], steps: list[TrajectoryStep]) -> bool:
        if task.get("expected_keywords"):
            text = " ".join(step.action + " " + step.observation for step in steps).lower()
            return all(str(keyword).lower() in text for keyword in task["expected_keywords"])
        return bool(steps) and steps[-1].success

    def run_batch(
        self, tasks: list[dict[str, Any]], skills: list[str], output_path: str
    ) -> list[Trajectory]:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        trajectories = []
        with output_file.open("w", encoding="utf-8") as f:
            for task in tasks:
                print(f"Running task: {task['id']}")
                trajectory = self.run_task(task, skills)
                trajectories.append(trajectory)
                f.write(json.dumps(trajectory.to_record(), ensure_ascii=False) + "\n")
        print(f"Saved {len(trajectories)} trajectories to {output_path}")
        return trajectories
