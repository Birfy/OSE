from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import DEFAULT_MODEL
from .skill_evolver.pipeline import SkillEvolutionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline skill evolution.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--output", default=None, help="Optional evolved skill library path.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--log", default="results/evolution_log.json")
    args = parser.parse_args()

    pipeline = SkillEvolutionPipeline(
        skill_library_path=args.skills,
        trajectory_path=args.trajectories,
        n_rounds=args.rounds,
        model=args.model,
        output_skill_library_path=args.output,
        n_iterations=args.iterations,
    )
    result = pipeline.run()
    pipeline.write_log(args.log)
    Path(args.log).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evolution log written to {args.log}")


if __name__ == "__main__":
    main()
