from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import DEFAULT_MODEL
from .evaluation.evaluator import SkillLearnBenchEvaluator


def parse_condition(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Condition must use format name:trajectory_jsonl:skill_library_path"
        )
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkillLearnBench-style evaluation.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--conditions", nargs="+", required=True, type=parse_condition)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    evaluator = SkillLearnBenchEvaluator(args.benchmark, args.model)
    summaries = []
    for condition_name, trajectory_path, skill_path in args.conditions:
        summaries.append(
            evaluator.run_full_evaluation(skill_path, trajectory_path, condition_name)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation summary written to {args.output}")


if __name__ == "__main__":
    main()
