from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm import DEFAULT_MODEL, LLMClient, parse_json_response


@dataclass
class ScoreResult:
    overall: float
    step_quality: float
    skill_alignment: float
    failure_avoidance: float
    reasoning: str


SCORER_PROMPT = """
You are evaluating whether an evolved agent skill improves action quality.

## Original Skill
{original_skill}

## Evolved Skill
{evolved_skill}

## Task Context
{task_description}

## Original Failed Step
Step action: {original_action}
Failure reason: {failure_reason}

## Re-generated Action With Evolved Skill
{new_action}

Score the re-generated action on three dimensions from 0.0 to 1.0:
1. step_quality: clear, specific, and executable
2. skill_alignment: follows the evolved skill procedure
3. failure_avoidance: avoids the original failure pattern

Respond only in JSON:
{{
  "step_quality": <float>,
  "skill_alignment": <float>,
  "failure_avoidance": <float>,
  "reasoning": "<one sentence>"
}}
"""


class LLMScorer:
    """Offline scorer: regenerate and score actions without executing the task."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.client = LLMClient(model)
        self.model = model

    def regenerate_action(
        self,
        task_description: str,
        evolved_skill_content: str,
        failed_step: dict[str, Any],
        context_steps: list[dict[str, Any]],
    ) -> str:
        context_str = "\n".join(
            f"Step {step.get('step_id')}: {step.get('action', '')}"
            for step in context_steps[-5:]
        )
        prompt = f"""You are an agent solving a task. Based on the skill below and the context,
generate the single action you would take at the current step.

# Skill
{evolved_skill_content}

# Task
{task_description}

# Context
{context_str}

# Current Step
The previous attempt at this step resulted in: {failed_step.get('failure_reason', 'unknown error')}

Respond with just the action, no explanation."""
        response = self.client.complete(prompt, max_tokens=512)
        return response.text.strip()

    def score_evolution(
        self,
        original_skill: str,
        evolved_skill: str,
        task_description: str,
        failed_step: dict[str, Any],
        new_action: str,
    ) -> ScoreResult:
        prompt = SCORER_PROMPT.format(
            original_skill=original_skill,
            evolved_skill=evolved_skill,
            task_description=task_description,
            original_action=failed_step.get("action", ""),
            failure_reason=failed_step.get("failure_reason", ""),
            new_action=new_action,
        )
        response = self.client.complete(prompt, max_tokens=512, json_mode=True)
        try:
            data = parse_json_response(response.text)
            step_quality = float(data["step_quality"])
            skill_alignment = float(data["skill_alignment"])
            failure_avoidance = float(data["failure_avoidance"])
            return ScoreResult(
                overall=step_quality * 0.3 + skill_alignment * 0.4 + failure_avoidance * 0.3,
                step_quality=step_quality,
                skill_alignment=skill_alignment,
                failure_avoidance=failure_avoidance,
                reasoning=data.get("reasoning", ""),
            )
        except Exception as exc:
            return ScoreResult(0.0, 0.0, 0.0, 0.0, f"Scoring failed: {exc}")

    def batch_score(
        self,
        evolved_skill_content: str,
        original_skill_content: str,
        skill_name: str,
        trajectory_group: dict[str, Any],
        top_k_failures: int = 5,
    ) -> float:
        failures = trajectory_group.get("failures", [])
        if not failures:
            return 0.5

        scores: list[float] = []
        count = 0
        for trajectory in failures:
            if count >= top_k_failures:
                break
            steps = trajectory.get("steps", [])
            failed_steps = [step for step in steps if not step.get("success", True)]
            context_steps = [step for step in steps if step.get("success", True)]
            for step in failed_steps[:2]:
                if count >= top_k_failures:
                    break
                new_action = self.regenerate_action(
                    trajectory.get("task_description", ""),
                    evolved_skill_content,
                    step,
                    context_steps,
                )
                score = self.score_evolution(
                    original_skill_content,
                    evolved_skill_content,
                    trajectory.get("task_description", ""),
                    step,
                    new_action,
                )
                scores.append(score.overall)
                count += 1
        return sum(scores) / len(scores) if scores else 0.0
