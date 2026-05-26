from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm import DEFAULT_MODEL
from src.run_trajectory_gen import discover_skills, load_tasks
from src.trajectory_generator import TrajectoryGenerator


ROOT_PATH_RE = re.compile(r'(?P<quote>["\'])(/root/[^"\']+)(?P=quote)')


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run full Codex rollouts for SkillLearnBench instances and score outputs."
    )
    parser.add_argument("--benchmark", default="data/skilllearnbench_upstream")
    parser.add_argument("--skills", default="data/skilllearnbench_upstream/skills/human_authored")
    parser.add_argument("--output-dir", default="results/skilllearnbench_rollouts")
    parser.add_argument("--artifact-root", default="/tmp/offskillevo_skilllearnbench_artifacts")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task", action="append", default=None, help="Task id/domain filter; repeatable.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=sys.executable, help="Python executable with pytest/test deps.")
    args = parser.parse_args()

    benchmark = Path(args.benchmark)
    output_dir = Path(args.output_dir)
    trajectories_dir = output_dir / "trajectories"
    scores_path = output_dir / "scores.jsonl"
    summary_path = output_dir / "summary.json"
    artifact_root = Path(args.artifact_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(benchmark, artifact_root)
    if args.task:
        filters = set(args.task)
        tasks = [
            task for task in tasks
            if task["id"] in filters or task.get("domain") in filters
        ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    skills = discover_skills(Path(args.skills), no_skill=False)
    generator = TrajectoryGenerator(
        args.skills,
        model=args.model,
        max_steps=args.max_steps,
        dry_run=False,
    )

    score_records: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        task_id = task["id"]
        print(f"[{index}/{len(tasks)}] rollout {task_id}", flush=True)
        trajectory_path = trajectories_dir / f"{task_id}.jsonl"
        score_path = output_dir / "scores" / f"{task_id}.json"
        score_path.parent.mkdir(parents=True, exist_ok=True)

        if args.resume and trajectory_path.exists() and score_path.exists():
            print(f"  resume: skip {task_id}", flush=True)
            score_records.append(json.loads(score_path.read_text(encoding="utf-8")))
            continue

        task_skills = generator._skills_for_task(task, skills)
        trajectory = generator.run_task(task, task_skills)
        trajectory_path.write_text(
            json.dumps(trajectory.to_record(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        score = score_task(
            benchmark=benchmark,
            task=task,
            artifact_dir=Path(task["artifact_dir"]),
            python=args.python,
            output_dir=output_dir,
        )
        score["trajectory_path"] = str(trajectory_path)
        score["skills_used"] = task_skills
        score["trajectory_final_success"] = trajectory.final_success
        score["trajectory_steps"] = len(trajectory.steps)
        score["trajectory_error_count"] = len(trajectory.error_log)
        score_path.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")
        append_jsonl(scores_path, score)
        score_records.append(score)
        print(
            f"  score={score['passed']}/{score['total']} "
            f"success={score['success']} steps={len(trajectory.steps)}",
            flush=True,
        )

    summary = summarize(score_records)
    summary.update({
        "benchmark": str(benchmark),
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "n_tasks": len(tasks),
    })
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("failed", 0) == 0 else 1


def score_task(
    *,
    benchmark: Path,
    task: dict[str, Any],
    artifact_dir: Path,
    python: str,
    output_dir: Path,
) -> dict[str, Any]:
    test_path = benchmark / "tasks" / task["domain"] / task["id"] / "tests" / "test_outputs.py"
    if not test_path.exists():
        return {
            "task_id": task["id"],
            "domain": task["domain"],
            "success": False,
            "passed": 0,
            "failed": 0,
            "total": 0,
            "reason": f"missing verifier: {test_path}",
            "artifact_dir": str(artifact_dir),
        }

    patched_dir = output_dir / "patched_tests"
    patched_dir.mkdir(parents=True, exist_ok=True)
    patched_test = patched_dir / f"{task['id']}_test_outputs.py"
    patched_test.write_text(
        patch_test_paths(test_path.read_text(encoding="utf-8"), artifact_dir),
        encoding="utf-8",
    )

    junit_path = output_dir / "junit" / f"{task['id']}.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [python, "-m", "pytest", str(patched_test), "-q", f"--junitxml={junit_path}"],
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    total, failures = parse_pytest_summary(proc.stdout + "\n" + proc.stderr)
    passed = max(total - failures, 0)
    return {
        "task_id": task["id"],
        "domain": task["domain"],
        "success": proc.returncode == 0,
        "passed": passed,
        "failed": failures,
        "total": total,
        "returncode": proc.returncode,
        "artifact_dir": str(artifact_dir),
        "test_path": str(test_path),
        "patched_test_path": str(patched_test),
        "junit_path": str(junit_path),
        "stdout_tail": tail(proc.stdout),
        "stderr_tail": tail(proc.stderr),
    }


def patch_test_paths(source: str, artifact_dir: Path) -> str:
    def repl(match: re.Match[str]) -> str:
        quote = match.group("quote")
        root_path = match.group(2)
        return f"{quote}{artifact_dir / Path(root_path).name}{quote}"

    return ROOT_PATH_RE.sub(repl, source)


def parse_pytest_summary(output: str) -> tuple[int, int]:
    failed = 0
    passed = 0
    for count, word in re.findall(r"(\d+)\s+(passed|failed|error|errors|skipped)", output):
        n = int(count)
        if word == "passed":
            passed += n
        elif word in {"failed", "error", "errors"}:
            failed += n
    if passed or failed:
        return passed + failed, failed
    if "no tests ran" in output:
        return 0, 0
    return 0, 1


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    succeeded = sum(1 for record in records if record.get("success"))
    tests_total = sum(int(record.get("total", 0)) for record in records)
    tests_passed = sum(int(record.get("passed", 0)) for record in records)
    by_domain: dict[str, dict[str, int]] = {}
    for record in records:
        domain = record.get("domain", "unknown")
        stats = by_domain.setdefault(domain, {"tasks": 0, "successes": 0, "tests": 0, "passed": 0})
        stats["tasks"] += 1
        stats["successes"] += int(bool(record.get("success")))
        stats["tests"] += int(record.get("total", 0))
        stats["passed"] += int(record.get("passed", 0))
    return {
        "total": total,
        "succeeded": succeeded,
        "failed": total - succeeded,
        "task_success_rate": succeeded / total if total else 0.0,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "test_pass_rate": tests_passed / tests_total if tests_total else 0.0,
        "by_domain": by_domain,
    }


def tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


if __name__ == "__main__":
    raise SystemExit(main())
