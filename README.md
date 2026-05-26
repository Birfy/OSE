# OffSkillEvo

Offline Skill Self-Evolution prototype based on `plan.md`.

The project has three stages:

1. Generate skill-augmented trajectories.
2. Evolve `SKILL.md` files offline from stored trajectories with G(s) grouping, proposal search, LLM scoring, and Pareto gating.
3. Evaluate skill quality, trajectory alignment, and task outcome.

## Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Commands

```bash
python -m src.run_trajectory_gen \
  --tasks data/skilllearnbench/tasks.json \
  --skills skills \
  --output data/trajectories/with_skill.jsonl

python -m src.run_evolution \
  --trajectories data/trajectories/with_skill.jsonl \
  --skills skills \
  --rounds 5 \
  --log results/evolution_log.json

python -m src.run_evaluation \
  --benchmark data/skilllearnbench \
  --conditions "ours:data/trajectories/with_skill.jsonl:skills" \
  --output results/summary.json
```

Use `--dry-run` on trajectory generation when you want deterministic local sample trajectories without API calls.

