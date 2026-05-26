from __future__ import annotations

import re
from dataclasses import dataclass

from .evolver import SkillProposal


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    score: float


class ParetoGate:
    """Two-part static gate: score improvement plus SKILL.md validity."""

    def __init__(self, baseline_score: float = 0.5, min_score_delta: float = 0.05) -> None:
        self.baseline_score = baseline_score
        self.min_score_delta = min_score_delta

    def evaluate(self, proposal: SkillProposal, new_q_score: float) -> GateDecision:
        threshold = self.baseline_score + self.min_score_delta
        if new_q_score < threshold:
            return GateDecision(
                accepted=False,
                reason=f"Score {new_q_score:.3f} < threshold {threshold:.3f}",
                score=new_q_score,
            )

        validity = self._check_skill_validity(proposal.new_content)
        if not validity["valid"]:
            return GateDecision(
                accepted=False,
                reason=f"Invalid skill content: {validity['reason']}",
                score=new_q_score,
            )

        return GateDecision(
            accepted=True,
            reason=f"Accepted: score {new_q_score:.3f}; {validity['reason']}",
            score=new_q_score,
        )

    def _check_skill_validity(self, content: str) -> dict[str, object]:
        checks = {
            "has_name": bool(re.search(r"^name:\s*\S+", content, re.MULTILINE)),
            "has_trigger": "trigger_conditions:" in content,
            "has_procedure": "## Procedure" in content or "procedure:" in content.lower(),
            "not_empty": len(content.strip()) > 100,
            "has_failure_patterns": "failure" in content.lower(),
        }
        failed = [name for name, passed in checks.items() if not passed]
        return {
            "valid": not failed,
            "reason": "All checks passed" if not failed else f"Failed checks: {failed}",
        }

