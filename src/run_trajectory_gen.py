from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import DEFAULT_MODEL
from .trajectory_generator import TrajectoryGenerator


def discover_skills(skill_dir: Path, no_skill: bool) -> list[str]:
    if no_skill or not skill_dir.exists():
        return []
    return sorted(path.parent.name for path in skill_dir.glob("*/SKILL.md"))


def load_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported task file format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate skill-augmented trajectories.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--no-skill", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    skills = discover_skills(Path(args.skills), args.no_skill)
    generator = TrajectoryGenerator(
        args.skills, model=args.model, max_steps=args.max_steps, dry_run=args.dry_run
    )
    generator.run_batch(tasks, skills, args.output)


if __name__ == "__main__":
    main()
