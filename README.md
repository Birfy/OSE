# OffSkillEvo

Offline Skill Self-Evolution prototype based on `plan.md`.

All model-backed work is delegated to the Codex agent runtime through
`codex exec`. This includes trajectory generation, skill proposal generation,
offline action regeneration/scoring, and trajectory-alignment evaluation.

The project has three stages:

1. Generate skill-augmented trajectories.
2. Evolve `SKILL.md` files offline from stored trajectories with G(s) grouping, proposal search, LLM scoring, and Pareto gating.
3. Evaluate skill quality, trajectory alignment, and task outcome.

## Setup

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
codex login
```

The default Codex agent model is `gpt-5.4-nano`. Override it with each CLI's
`--model` flag when needed.

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

## SkillLearnBench Full Rollouts

After downloading SkillLearnBench into `data/skilllearnbench_upstream`, run all
100 instances with task-scoped human-authored skills and score each instance:

```bash
/home/admin/OSkill/.venv-skilllearnbench/bin/python scripts/run_skilllearnbench_rollouts.py \
  --benchmark data/skilllearnbench_upstream \
  --skills data/skilllearnbench_upstream/skills/human_authored \
  --output-dir results/skilllearnbench_rollouts \
  --artifact-root /tmp/offskillevo_skilllearnbench_artifacts \
  --python /home/admin/OSkill/.venv-skilllearnbench/bin/python \
  --concurrency 4 \
  --resume
```

Outputs:

- `results/skilllearnbench_rollouts/trajectories/*.jsonl`
- `results/skilllearnbench_rollouts/scores.jsonl`
- `results/skilllearnbench_rollouts/summary.json`
