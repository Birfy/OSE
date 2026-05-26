from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.llm import DEFAULT_MODEL, LLMClient, parse_json_response


@dataclass
class EvalResult:
    task_id: str
    domain: str
    skill_quality_score: float
    trajectory_alignment: float
    task_success: bool
    notes: str = ""


class SkillLearnBenchEvaluator:
    def __init__(self, benchmark_path: str, model: str = DEFAULT_MODEL) -> None:
        self.bench_path = Path(benchmark_path)
        self.client = LLMClient(model)
        self.model = model

    def evaluate_skill_quality(self, skill_content: str, task: dict[str, Any]) -> float:
        domain = str(task.get("domain", "")).lower()
        checks = {
            "has_trigger": "trigger_conditions:" in skill_content,
            "has_procedure": "## Procedure" in skill_content,
            "has_examples": "## Examples" in skill_content,
            "has_failure": "## Failure" in skill_content,
            "covers_domain": bool(domain and domain in skill_content.lower()),
            "reasonable_length": 200 < len(skill_content) < 5000,
        }
        return sum(1 for passed in checks.values() if passed) / len(checks)

    def evaluate_trajectory_alignment(self, trajectory: dict[str, Any], skill_content: str) -> float:
        steps_summary = "\n".join(
            f"Step {step.get('step_id')}: {step.get('action', '')[:160]}"
            for step in trajectory.get("steps", [])
        )
        response = self.client.complete(
            f"""Rate how well the following trajectory follows the skill from 0.0 to 1.0.

Skill:
{skill_content[:1200]}

Trajectory:
{steps_summary}

Respond only with JSON: {{"score": <float>, "reason": "<brief>"}}""",
            max_tokens=256,
            json_mode=True,
        )
        try:
            data = parse_json_response(response.text)
            return max(0.0, min(1.0, float(data["score"])))
        except Exception:
            return 0.0

    def evaluate_task_outcome(self, task: dict[str, Any], trajectory: dict[str, Any]) -> bool:
        keypoints_path = self.bench_path / "eval_keypoints" / f"{task['id']}.json"
        if not keypoints_path.exists():
            return bool(trajectory.get("final_success", False))

        keypoints = json.loads(keypoints_path.read_text(encoding="utf-8"))
        required_actions = keypoints.get("required_actions", [])
        if not required_actions:
            return bool(trajectory.get("final_success", False))

        trajectory_text = " ".join(
            f"{step.get('action', '')} {step.get('observation', '')}"
            for step in trajectory.get("steps", [])
        ).lower()
        matched = sum(1 for item in required_actions if str(item).lower() in trajectory_text)
        return matched >= len(required_actions) * 0.7

    def run_full_evaluation(
        self,
        skill_library_path: str,
        trajectory_path: str,
        condition_name: str = "ours",
    ) -> dict[str, Any]:
        trajectories = {}
        with Path(trajectory_path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trajectory = json.loads(line)
                    trajectories[trajectory["task_id"]] = trajectory

        tasks_data = json.loads((self.bench_path / "tasks.json").read_text(encoding="utf-8"))
        tasks = tasks_data["tasks"] if isinstance(tasks_data, dict) and "tasks" in tasks_data else tasks_data
        skill_lib = Path(skill_library_path)
        results: list[EvalResult] = []

        for task in tasks:
            task_id = str(task["id"])
            trajectory = trajectories.get(task_id)
            if not trajectory:
                continue
            skill_content = ""
            for skill_name in trajectory.get("skills_used", []):
                skill_path = skill_lib / skill_name / "SKILL.md"
                if skill_path.exists():
                    skill_content += skill_path.read_text(encoding="utf-8") + "\n"
            results.append(
                EvalResult(
                    task_id=task_id,
                    domain=task.get("domain", "unknown"),
                    skill_quality_score=self.evaluate_skill_quality(skill_content, task),
                    trajectory_alignment=self.evaluate_trajectory_alignment(trajectory, skill_content),
                    task_success=self.evaluate_task_outcome(task, trajectory),
                )
            )

        n = len(results)
        if n == 0:
            return {
                "condition": condition_name,
                "n_tasks": 0,
                "skill_quality": 0.0,
                "trajectory_alignment": 0.0,
                "task_success_rate": 0.0,
                "by_domain": {},
                "results": [],
            }

        return {
            "condition": condition_name,
            "n_tasks": n,
            "skill_quality": sum(item.skill_quality_score for item in results) / n,
            "trajectory_alignment": sum(item.trajectory_alignment for item in results) / n,
            "task_success_rate": sum(item.task_success for item in results) / n,
            "by_domain": self._aggregate_by_domain(results),
            "results": [asdict(item) for item in results],
        }

    def _aggregate_by_domain(self, results: list[EvalResult]) -> dict[str, float]:
        domain_stats: dict[str, list[bool]] = defaultdict(list)
        for result in results:
            domain_stats[result.domain].append(result.task_success)
        return {
            domain: sum(values) / len(values)
            for domain, values in sorted(domain_stats.items())
        }
