#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
SKILLS_DIR="skills"
BENCH_DIR="data/skilllearnbench"
RESULTS_DIR="results"

mkdir -p "$DATA_DIR/trajectories" "$RESULTS_DIR"

echo "=== Step 1: Generate baseline trajectories without skills ==="
python -m src.run_trajectory_gen \
  --tasks "$BENCH_DIR/tasks.json" \
  --skills "$SKILLS_DIR" \
  --output "$DATA_DIR/trajectories/no_skill.jsonl" \
  --no-skill

echo "=== Step 2: Generate trajectories with initial skills ==="
python -m src.run_trajectory_gen \
  --tasks "$BENCH_DIR/tasks.json" \
  --skills "$SKILLS_DIR" \
  --output "$DATA_DIR/trajectories/with_skill.jsonl"

echo "=== Step 3: Run offline skill evolution ==="
python -m src.run_evolution \
  --trajectories "$DATA_DIR/trajectories/with_skill.jsonl" \
  --skills "$SKILLS_DIR" \
  --output "$SKILLS_DIR/evolved" \
  --rounds 5 \
  --log "$RESULTS_DIR/evolution_log.json"

echo "=== Step 4: Generate trajectories with evolved skills ==="
python -m src.run_trajectory_gen \
  --tasks "$BENCH_DIR/tasks.json" \
  --skills "$SKILLS_DIR/evolved" \
  --output "$DATA_DIR/trajectories/evolved_skill.jsonl"

echo "=== Step 5: Evaluate all conditions ==="
python -m src.run_evaluation \
  --benchmark "$BENCH_DIR" \
  --conditions \
    "no_skill:$DATA_DIR/trajectories/no_skill.jsonl:$SKILLS_DIR" \
    "with_skill:$DATA_DIR/trajectories/with_skill.jsonl:$SKILLS_DIR" \
    "evolved:$DATA_DIR/trajectories/evolved_skill.jsonl:$SKILLS_DIR/evolved" \
  --output "$RESULTS_DIR/summary.json"

echo "=== Done ==="

