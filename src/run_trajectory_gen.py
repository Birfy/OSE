from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .llm import DEFAULT_MODEL
from .trajectory_generator import TrajectoryGenerator


def discover_skills(skill_dir: Path, no_skill: bool) -> list[str]:
    if no_skill or not skill_dir.exists():
        return []
    return sorted(path.parent.name for path in skill_dir.glob("*/SKILL.md"))


def load_tasks(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        return load_skilllearnbench_tasks(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported task file format: {path}")


def load_skilllearnbench_tasks(path: Path) -> list[dict[str, Any]]:
    """
    Load upstream SkillLearnBench task folders.

    Accepts either the repository root or the `tasks/` directory. Each instance
    is expected to contain `instruction.md`, matching the upstream layout:
    tasks/<task-name>/<task-name>-N/instruction.md.
    """
    tasks_root = path / "tasks" if (path / "tasks").is_dir() else path
    records: list[dict[str, Any]] = []
    for instruction in sorted(tasks_root.glob("*/*/instruction.md")):
        instance_dir = instruction.parent
        task_name = instance_dir.parent.name
        instance_id = instance_dir.name
        description = instruction.read_text(encoding="utf-8").strip()
        records.append(
            {
                "id": instance_id,
                "task_description": description,
                "description": (
                    f"{description}\n\n"
                    f"SkillLearnBench instance path: {instance_dir}\n"
                    "Generate the next agent action for this task; do not assume hidden files "
                    "outside the instance path."
                ),
                "domain": task_name,
                "benchmark": "SkillLearnBench",
                "task_path": str(instance_dir),
            }
        )
    if not records:
        raise ValueError(f"No SkillLearnBench instruction.md files found under {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate skill-augmented trajectories.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-skill", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    skills = discover_skills(Path(args.skills), args.no_skill)
    generator = TrajectoryGenerator(
        args.skills, model=args.model, max_steps=args.max_steps, dry_run=args.dry_run
    )
    generator.run_batch(tasks, skills, args.output)


if __name__ == "__main__":
    main()
