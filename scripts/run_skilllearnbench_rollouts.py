from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm import DEFAULT_MODEL
from src.run_trajectory_gen import discover_skills, load_tasks
from src.trajectory_generator import TrajectoryGenerator


ROOT_PATH_RE = re.compile(r'(?P<quote>["\'])(/root/[^"\']+)(?P=quote)')
APP_PATH_RE = re.compile(r'(?P<quote>["\'])(/app/[^"\']+)(?P=quote)')


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
    parser.add_argument("--concurrency", type=int, default=1)
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

    score_records: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, task in enumerate(tasks, start=1):
        score_path = output_dir / "scores" / f"{task['id']}.json"
        trajectory_path = trajectories_dir / f"{task['id']}.jsonl"
        if args.resume and trajectory_path.exists() and score_path.exists():
            print(f"[{index}/{len(tasks)}] resume: skip {task['id']}", flush=True)
            score_records.append(json.loads(score_path.read_text(encoding="utf-8")))
        else:
            pending.append((index, task))

    max_workers = max(1, args.concurrency)
    if pending:
        print(f"Running {len(pending)} rollout(s) with concurrency={max_workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_one_task,
                index=index,
                total=len(tasks),
                task=task,
                skills=skills,
                skills_path=args.skills,
                model=args.model,
                max_steps=args.max_steps,
                benchmark=benchmark,
                trajectories_dir=trajectories_dir,
                output_dir=output_dir,
                python=args.python,
            ): (index, task)
            for index, task in pending
        }
        for future in concurrent.futures.as_completed(futures):
            index, task = futures[future]
            try:
                score = future.result()
            except Exception as exc:
                score = failure_record(task, exc)
            score_records.append(score)
            print(
                f"[{index}/{len(tasks)}] done {task['id']} "
                f"score={score['passed']}/{score['total']} "
                f"success={score['success']} steps={score.get('trajectory_steps', 0)}",
                flush=True,
            )

    summary = summarize(score_records)
    summary.update({
        "benchmark": str(benchmark),
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "n_tasks": len(tasks),
    })
    score_records.sort(key=lambda record: record.get("task_id", ""))
    scores_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in score_records),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("failed", 0) == 0 else 1


def run_one_task(
    *,
    index: int,
    total: int,
    task: dict[str, Any],
    skills: list[str] | dict[str, list[str]],
    skills_path: str,
    model: str,
    max_steps: int,
    benchmark: Path,
    trajectories_dir: Path,
    output_dir: Path,
    python: str,
) -> dict[str, Any]:
    print(f"[{index}/{total}] rollout {task['id']}", flush=True)
    generator = TrajectoryGenerator(
        skills_path,
        model=model,
        max_steps=max_steps,
        dry_run=False,
    )
    task_skills = generator._skills_for_task(task, skills)
    trajectory = generator.run_task(task, task_skills)

    trajectory_path = trajectories_dir / f"{task['id']}.jsonl"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(
        json.dumps(trajectory.to_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    score = score_task(
        benchmark=benchmark,
        task=task,
        artifact_dir=Path(task["artifact_dir"]),
        python=python,
        output_dir=output_dir,
    )
    score["trajectory_path"] = str(trajectory_path)
    score["skills_used"] = task_skills
    score["trajectory_final_success"] = trajectory.final_success
    score["trajectory_steps"] = len(trajectory.steps)
    score["trajectory_error_count"] = len(trajectory.error_log)

    score_path = output_dir / "scores" / f"{task['id']}.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")
    return score


def failure_record(task: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "task_id": task["id"],
        "domain": task.get("domain", "unknown"),
        "success": False,
        "passed": 0,
        "failed": 1,
        "total": 1,
        "reason": f"{exc.__class__.__name__}: {exc}",
        "artifact_dir": task.get("artifact_dir", ""),
    }


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

    patched_dir = output_dir / "patched_tests" / task["id"]
    patched_dir.mkdir(parents=True, exist_ok=True)
    copy_test_support_files(test_path.parent, patched_dir)
    patched_test = patched_dir / "test_outputs.py"
    task_dir = benchmark / "tasks" / task["domain"] / task["id"]
    patched_source = patch_test_paths(
        test_path.read_text(encoding="utf-8"),
        artifact_dir=artifact_dir,
        task_dir=task_dir,
    )
    patched_source = patch_verifier_environment(
        patched_source,
        verifier_dir=output_dir / "verifier_logs" / task["id"],
        task_id=task["id"],
    )
    patched_test.write_text(patched_source, encoding="utf-8")

    junit_path = output_dir / "junit" / f"{task['id']}.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [python, "-m", "pytest", str(patched_test), "-q", f"--junitxml={junit_path}"],
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    total, failures = parse_junit_counts(junit_path)
    if total is None or failures is None:
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


def patch_test_paths(source: str, *, artifact_dir: Path, task_dir: Path) -> str:
    def repl_root(match: re.Match[str]) -> str:
        quote = match.group("quote")
        root_path = match.group(2)
        artifact_path = artifact_dir / Path(root_path).name
        environment_path = task_dir / "environment" / Path(root_path).name
        rewritten = environment_path if not artifact_path.exists() and environment_path.exists() else artifact_path
        return f"{quote}{rewritten}{quote}"

    def repl_app(match: re.Match[str]) -> str:
        quote = match.group("quote")
        app_path = Path(match.group(2))
        parts = app_path.parts
        if len(parts) >= 3 and parts[2] == "data":
            rewritten = task_dir / "environment" / Path(*parts[2:])
        else:
            rewritten = artifact_dir / app_path.name
        return f"{quote}{rewritten}{quote}"

    source = ROOT_PATH_RE.sub(repl_root, source)
    return APP_PATH_RE.sub(repl_app, source)


def copy_test_support_files(source_dir: Path, patched_dir: Path) -> None:
    for source_path in source_dir.iterdir():
        if source_path.name == "__pycache__":
            continue
        if source_path.name == "test_outputs.py":
            continue
        target_path = patched_dir / source_path.name
        if source_path.is_dir():
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
        elif source_path.is_file():
            shutil.copy2(source_path, target_path)


def patch_verifier_environment(source: str, *, verifier_dir: Path, task_id: str) -> str:
    verifier_dir.mkdir(parents=True, exist_ok=True)
    quoted_dir = json.dumps(str(verifier_dir))
    source = re.sub(
        r'(?P<quote>["\'])/logs/verifier(?P=quote)',
        quoted_dir,
        source,
    )
    port = 20000 + (zlib.crc32(task_id.encode("utf-8")) % 20000)
    return re.sub(
        r"(?m)^(?P<indent>\s*)PORT\s*=\s*8765\b",
        rf"\g<indent>PORT = {port}",
        source,
    )


def parse_pytest_summary(output: str) -> tuple[int, int]:
    summary_lines = []
    for line in output.splitlines():
        stripped = line.strip().strip("=")
        if " in " not in stripped:
            continue
        if re.search(r"\b\d+\s+(passed|failed|error|errors)\b", stripped):
            summary_lines.append(stripped)
    if summary_lines:
        output = summary_lines[-1]
    else:
        output = ""

    failed = 0
    passed = 0
    for count, word in re.findall(r"(\d+)\s+(passed|failed|error|errors)\b", output):
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


def parse_junit_counts(junit_path: Path) -> tuple[int | None, int | None]:
    if not junit_path.exists():
        return None, None
    try:
        root = ET.parse(junit_path).getroot()
    except ET.ParseError:
        return None, None

    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        return None, None

    total = 0
    failures = 0
    for suite in suites:
        total += int(suite.attrib.get("tests", 0))
        failures += int(suite.attrib.get("failures", 0))
        failures += int(suite.attrib.get("errors", 0))
    return total, failures


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
