from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.llm import DEFAULT_MODEL, LLMClient, parse_json_response


ProposalType = Literal["refine", "create", "trigger_update"]


@dataclass
class SkillProposal:
    proposal_type: ProposalType
    skill_name: str
    new_content: str
    rationale: str
    prior_confidence: float


EVOLVER_PROMPT = """
You are an expert at improving agent skill specifications based on failure analysis.

## Current Skill
{skill_content}

## Failure Analysis
{failure_summary}

## Task
Generate {k} diverse improvement proposals. Each proposal must be one of:
- "refine": improve the existing skill's procedure or instructions
- "create": create a new complementary skill to handle uncovered cases
- "trigger_update": refine trigger conditions to prevent misapplication

For each proposal, provide:
1. The complete new SKILL.md content
2. The rationale
3. Your confidence score from 0.0 to 1.0

Respond only as a JSON array:
[
  {{
    "type": "refine|create|trigger_update",
    "skill_name": "<name>",
    "new_content": "<complete SKILL.md>",
    "rationale": "<why>",
    "confidence": <float>
  }}
]
"""


class SkillEvolver:
    def __init__(self, skill_library_path: str, model: str = DEFAULT_MODEL) -> None:
        self.skill_lib = Path(skill_library_path)
        self.client = LLMClient(model)
        self.model = model
        self.rejected_buffer: list[dict[str, str]] = []

    def generate_proposals(
        self, skill_name: str, failure_summary: str, k: int = 6
    ) -> list[SkillProposal]:
        skill_path = self.skill_lib / skill_name / "SKILL.md"
        skill_content = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
        if self.rejected_buffer:
            rejected_lines = [
                f"- {item['type']} {item['skill_name']}: {item['rationale']}"
                for item in self.rejected_buffer[-5:]
            ]
            skill_content += "\n\n## Previously Rejected Directions\n" + "\n".join(rejected_lines)

        response = self.client.complete(
            EVOLVER_PROMPT.format(
                skill_content=skill_content,
                failure_summary=failure_summary,
                k=k,
            ),
            max_tokens=4096,
            json_mode=True,
        )

        try:
            proposals_data = parse_json_response(response.text)
        except Exception:
            return []

        proposals = []
        for item in proposals_data if isinstance(proposals_data, list) else []:
            proposal_type = item.get("type", "refine")
            if proposal_type not in {"refine", "create", "trigger_update"}:
                proposal_type = "refine"
            proposals.append(
                SkillProposal(
                    proposal_type=proposal_type,
                    skill_name=item.get("skill_name", skill_name),
                    new_content=item.get("new_content", ""),
                    rationale=item.get("rationale", ""),
                    prior_confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                )
            )
        return proposals

    def add_to_rejected_buffer(self, proposal: SkillProposal) -> None:
        self.rejected_buffer.append(
            {
                "skill_name": proposal.skill_name,
                "type": proposal.proposal_type,
                "rationale": proposal.rationale,
            }
        )
        self.rejected_buffer = self.rejected_buffer[-20:]
