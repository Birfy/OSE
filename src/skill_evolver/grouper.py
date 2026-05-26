from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class SkillGrouper:
    """SkillClaw-style G(s) grouping: G(s) = trajectories that used skill s."""

    def __init__(self, trajectory_path: str) -> None:
        self.trajectory_path = Path(trajectory_path)
        self.trajectories = self._load(self.trajectory_path)

    def _load(self, path: Path) -> list[dict[str, Any]]:
        trajectories = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))
        return trajectories

    def group_by_skill(self) -> dict[str, dict[str, Any]]:
        groups = defaultdict(
            lambda: {"successes": [], "failures": [], "error_patterns": Counter()}
        )
        for trajectory in self.trajectories:
            for skill_name in trajectory.get("skills_used", []):
                if trajectory.get("final_success", False):
                    groups[skill_name]["successes"].append(trajectory)
                    continue
                groups[skill_name]["failures"].append(trajectory)
                for error in trajectory.get("error_log", []):
                    error_type = str(error).split(":", 1)[0].strip() or "UNKNOWN"
                    groups[skill_name]["error_patterns"][error_type] += 1
        return dict(groups)

    def get_skill_signal_summary(self, skill_name: str) -> str:
        groups = self.group_by_skill()
        if skill_name not in groups:
            return "No data available for this skill."

        group = groups[skill_name]
        successes = group["successes"]
        failures = group["failures"]
        total = len(successes) + len(failures)
        success_rate = (len(successes) / total) if total else 0.0

        lines = [
            f"Skill: {skill_name}",
            f"Usage: {len(successes)} successes, {len(failures)} failures",
            f"Success rate: {success_rate * 100:.1f}%",
            "Top failure patterns:",
        ]
        for error_type, count in group["error_patterns"].most_common(3):
            lines.append(f"  - {error_type}: {count} occurrences")

        if failures:
            lines.append("")
            lines.append("Sample failure steps:")
            for trajectory in failures[:3]:
                for step in trajectory.get("steps", []):
                    if not step.get("success", True):
                        lines.append(f"  > {step.get('failure_reason', 'unknown')}")
        return "\n".join(lines)

