from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .llm import DEFAULT_MODEL, LLMClient, first_text


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
        self.client = None if dry_run else LLMClient(model, sandbox="workspace-write", timeout_seconds=900)

    def load_skill(self, skill_name: str, task: dict[str, Any] | None = None) -> str:
        skill_path = self._resolve_skill_path(skill_name, task)
        return skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""

    def _resolve_skill_path(self, skill_name: str, task: dict[str, Any] | None = None) -> Path:
        domain = (task or {}).get("domain")
        if domain:
            task_scoped = self.skill_lib / str(domain) / skill_name / "SKILL.md"
            if task_scoped.exists():
                return task_scoped
        return self.skill_lib / skill_name / "SKILL.md"

    def build_system_prompt(self, task: dict[str, Any], skills: list[str]) -> str:
        skill_contents = []
        for skill in skills:
            content = self.load_skill(skill, task)
            if content:
                skill_contents.append(f"## Skill: {skill}\n{content}")
        skills_block = "\n\n".join(skill_contents) if skill_contents else "No skills loaded."
        return f"""You are a capable agent solving real-world tasks.

# Available Skills
{skills_block}

# Skill Usage
- The skill documents above are already loaded into this prompt and are the authoritative skill context.
- Do not search for these skill files in ~/.codex or other global skill directories.
- When solving a SkillLearnBench task, use only the skills selected for that task.

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

        rollout_prompt = f"""{task["description"]}

Run the task as a Codex agent and produce a complete trajectory.
Use up to {self.max_steps} meaningful actions. Inspect files and run commands when useful.
Stop when the task is complete or when you cannot make further progress.
"""
        try:
            events = self.client.run_agent(rollout_prompt, system=system_prompt)
        except Exception as exc:
            reason = f"AGENT_ERROR: {exc}"
            trajectory.steps.append(
                TrajectoryStep(
                    step_id=0,
                    state="agent-error",
                    skills_active=skills,
                    action=reason,
                    action_type="error",
                    success=False,
                    failure_reason=reason,
                )
            )
            trajectory.error_log.append(reason)
            return trajectory

        trajectory.steps.extend(self._parse_codex_events(events, skills))
        for step in trajectory.steps:
            if step.failure_reason:
                trajectory.error_log.append(step.failure_reason)

        trajectory.final_success = self._check_task_success(task, trajectory.steps)

        return trajectory

    def _parse_codex_events(
        self, events: list[dict[str, Any]], active_skills: list[str]
    ) -> list[TrajectoryStep]:
        steps: list[TrajectoryStep] = []
        usage: dict[str, Any] | None = None
        for event in events:
            event_type = event.get("type", "")
            if event_type == "turn.completed":
                usage = event.get("usage", {})
                continue
            if event_type != "item.completed":
                continue

            item = event.get("item", {})
            item_type = item.get("type", "")
            if item_type == "agent_message":
                text = item.get("text", "")
                failure_reason = self._extract_failure_reason(text)
                steps.append(
                    TrajectoryStep(
                        step_id=len(steps),
                        state=f"Codex event {len(steps)}",
                        skills_active=active_skills,
                        action=text,
                        action_type="text",
                        tool_input={"event_id": item.get("id")},
                        success=failure_reason is None,
                        failure_reason=failure_reason,
                    )
                )
            elif item_type == "command_execution":
                command = item.get("command", "")
                output = item.get("aggregated_output", "")
                exit_code = item.get("exit_code")
                success = exit_code == 0
                failure_reason = None if success else f"COMMAND_FAILED: exit_code={exit_code}"
                steps.append(
                    TrajectoryStep(
                        step_id=len(steps),
                        state=f"Codex event {len(steps)}",
                        skills_active=active_skills,
                        action=command,
                        action_type="tool_use",
                        tool_name="shell",
                        tool_input={
                            "event_id": item.get("id"),
                            "exit_code": exit_code,
                            "status": item.get("status"),
                        },
                        observation=output,
                        success=success,
                        failure_reason=failure_reason,
                    )
                )
        if usage and steps:
            steps[-1].tool_input = steps[-1].tool_input or {}
            steps[-1].tool_input["usage"] = usage
        if not steps:
            steps.append(
                TrajectoryStep(
                    step_id=0,
                    state="no-events",
                    skills_active=active_skills,
                    action=json.dumps(events, ensure_ascii=False)[:4000],
                    action_type="raw",
                    success=False,
                    failure_reason="NO_EVENTS: Codex produced no completed trajectory items",
                )
            )
        return steps

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
        return bool(steps) and steps[-1].success and all(step.success for step in steps)

    def run_batch(
        self,
        tasks: list[dict[str, Any]],
        skills: list[str] | dict[str, list[str]],
        output_path: str,
    ) -> list[Trajectory]:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        trajectories = []
        with output_file.open("w", encoding="utf-8") as f:
            for task in tasks:
                print(f"Running task: {task['id']}")
                task_skills = self._skills_for_task(task, skills)
                trajectory = self.run_task(task, task_skills)
                trajectories.append(trajectory)
                f.write(json.dumps(trajectory.to_record(), ensure_ascii=False) + "\n")
        print(f"Saved {len(trajectories)} trajectories to {output_path}")
        return trajectories

    def _skills_for_task(
        self, task: dict[str, Any], skills: list[str] | dict[str, list[str]]
    ) -> list[str]:
        if isinstance(skills, list):
            return skills
        return skills.get(str(task.get("domain", "")), [])
