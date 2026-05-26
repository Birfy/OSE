# Offline Skill Self-Evolution 开发文档

**项目名称**：OffSkillEvo  
**版本**：v0.1  
**日期**：2026-05  
**技术栈**：Python 3.11 · codex SDK · Anthropic API · SkillLearnBench

---

## 目录

1. [系统概述](#1-系统概述)
2. [模块一：轨迹生成（codex + Skills）](#2-模块一轨迹生成)
3. [模块二：离线 Skill 自演进（树搜索 + LLM 打分）](#3-模块二离线-skill-自演进)
4. [模块三：SkillLearnBench 实验评估](#4-模块三实验评估)
5. [项目结构](#5-项目结构)
6. [环境配置](#6-环境配置)
7. [运行示例](#7-运行示例)
8. [关键设计决策](#8-关键设计决策)

---

## 1. 系统概述

### 1.1 核心思想

三个阶段串联：

```
阶段 1                阶段 2                        阶段 3
codex           离线 Skill 自演进              SkillLearnBench
+ Skills              树搜索 + LLM 打分               评估
─────────             ──────────────────             ──────────────
生成轨迹 D     →      skill 演进 S* = evolve(D)  →   三层指标对比
```

**核心约束**：
- 整个演进过程**不重新跑任何推理**（不重新执行任务）
- 只根据演进出的新 skill，**让 LLM 对原轨迹的关键步重新生成 action 并打分**
- 全量轨迹不切分，使用 SkillClaw 的 G(s) 分组策略处理异构数据

### 1.2 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  Module 1: Trajectory Generator                                  │
│  codex CLI → skill-augmented rollout → trajectory JSONL   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ D (全量轨迹)
┌───────────────────────────▼─────────────────────────────────────┐
│  Module 2: Offline Skill Evolver                                 │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │ G(s) 分组   │→  │ SkillEvolver │→  │ PUCT Tree Search    │  │
│  │ (SkillClaw) │   │ 失败诊断+K   │   │ + bounded edits     │  │
│  └─────────────┘   │ proposals    │   └──────────┬──────────┘  │
│                    └──────────────┘              │             │
│  ┌─────────────────────────────────────────────▼──────────┐   │
│  │ LLM Scorer: 对关键步重新生成 action + 多维度打分        │   │
│  │ (不重新执行，只让 LLM 根据新 skill 生成 action)          │   │
│  └───────────────────────────┬──────────────────────────┘    │
│                              │ Q 值回传                        │
│  ┌───────────────────────────▼──────────────────────────┐     │
│  │ Pareto Gate: LLM 分数提升 + skill 内容合法性检查       │     │
│  └───────────────────────────┬──────────────────────────┘     │
│                              │ accepted skills                  │
│                         Skill Library (SKILL.md)                │
└─────────────────────────────────────────────────────────────────┘
                            │ S*
┌───────────────────────────▼─────────────────────────────────────┐
│  Module 3: SkillLearnBench Evaluation                            │
│  三层评估：skill 质量 · 轨迹对齐 · 任务结果                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块一：轨迹生成

### 2.1 功能目标

用 codex CLI 配合 Skill Library 对 SkillLearnBench 的 20 个任务跑 agent，收集完整的交互轨迹并结构化存储。

### 2.2 Skill 格式规范

每个 skill 存为独立目录，遵循 OpenClaw SKILL.md 规范：

```
skills/
├── data_analysis/
│   ├── SKILL.md          # 主文件，含元数据 + 指令
│   ├── examples/         # 可选：示例
│   └── helpers.py        # 可选：辅助脚本
├── web_search/
│   └── SKILL.md
└── code_execution/
    └── SKILL.md
```

**SKILL.md 模板**：

```markdown
---
name: data_analysis
version: 1.0.0
trigger_conditions:
  - "task involves analyzing structured data"
  - "user asks for statistics or aggregation"
applicable_domains: ["data science", "analytics", "research"]
tools_required: ["python", "pandas"]
cost_estimate: low
---

## Description
分析结构化数据，生成统计摘要和可视化。

## Procedure
1. 读取数据文件，检查格式和缺失值
2. 计算描述性统计（mean, std, quartiles）
3. 识别异常值和分布特征
4. 生成简洁的文字摘要

## Examples
<example>
Input: 用户上传了一个 CSV 文件，要求分析销售趋势
Action: pd.read_csv → describe() → groupby('month').sum()
Output: 月度销售趋势表 + 同比增长率
</example>

## Failure Patterns
- 数据格式不一致：先做类型转换
- 大文件内存溢出：使用 chunksize 分批读取
```

### 2.3 轨迹生成器实现

**文件：`src/trajectory_generator.py`**

```python
import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import anthropic

@dataclass
class TrajectoryStep:
    step_id: int
    state: str                    # 当前任务状态描述
    skills_active: list[str]      # 本步激活的 skill 名称
    action: str                   # agent 实际采取的 action
    action_type: str              # tool_use / text / code
    tool_name: Optional[str]
    tool_input: Optional[dict]
    observation: str              # 环境反馈
    success: bool                 # 本步是否成功
    failure_reason: Optional[str] # 失败原因（结构化）

@dataclass
class Trajectory:
    task_id: str
    task_description: str
    domain: str                   # SkillLearnBench 子领域
    skills_used: list[str]        # 整条轨迹用到的 skill
    steps: list[TrajectoryStep]
    final_success: bool
    error_log: list[str]          # 收集所有错误信息，用于后续失败诊断


class TrajectoryGenerator:
    def __init__(self, skill_library_path: str, model: str = "gpt5.5-nano"):
        self.skill_lib = Path(skill_library_path)
        self.model = model
        self.client = anthropic.Anthropic()

    def load_skill(self, skill_name: str) -> str:
        """读取 SKILL.md 内容"""
        skill_path = self.skill_lib / skill_name / "SKILL.md"
        if skill_path.exists():
            return skill_path.read_text()
        return ""

    def build_system_prompt(self, task: dict, skills: list[str]) -> str:
        """构建带 skill 注入的 system prompt"""
        skill_contents = []
        for s in skills:
            content = self.load_skill(s)
            if content:
                skill_contents.append(f"## Skill: {s}\n{content}")

        skills_block = "\n\n".join(skill_contents) if skill_contents else "No skills loaded."

        return f"""You are a capable agent solving real-world tasks.

# Available Skills
{skills_block}

# Task
{task['description']}

# Instructions
- Follow the skill procedures when applicable
- Record your reasoning at each step
- If a skill's trigger condition matches, invoke it explicitly
- Report failures with specific error details
"""

    def run_task(self, task: dict, skills: list[str]) -> Trajectory:
        """
        对单个任务跑 agent，收集轨迹。
        使用 claude-code SDK 的 streaming tool use。
        """
        trajectory = Trajectory(
            task_id=task["id"],
            task_description=task["description"],
            domain=task.get("domain", "unknown"),
            skills_used=skills,
            steps=[],
            final_success=False,
            error_log=[]
        )

        messages = []
        system_prompt = self.build_system_prompt(task, skills)
        messages.append({"role": "user", "content": task["description"]})

        step_id = 0
        max_steps = 20

        while step_id < max_steps:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=self._get_tools()
            )

            # 解析 response，记录 step
            step = self._parse_response_to_step(step_id, response, skills)
            trajectory.steps.append(step)

            if not step.success and step.failure_reason:
                trajectory.error_log.append(step.failure_reason)

            # 检查终止条件
            if response.stop_reason == "end_turn":
                trajectory.final_success = self._check_task_success(
                    task, trajectory.steps
                )
                break

            # 执行工具调用，获取 observation
            if response.stop_reason == "tool_use":
                tool_result = self._execute_tool(response)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_result})
            else:
                break

            step_id += 1

        return trajectory

    def _parse_response_to_step(
        self, step_id: int, response, active_skills: list[str]
    ) -> TrajectoryStep:
        """从 API response 中提取结构化 step"""
        action_text = ""
        action_type = "text"
        tool_name = None
        tool_input = None

        for block in response.content:
            if block.type == "text":
                action_text = block.text
                action_type = "text"
            elif block.type == "tool_use":
                action_text = f"[{block.name}] {json.dumps(block.input)}"
                action_type = "tool_use"
                tool_name = block.name
                tool_input = block.input

        # 简单的失败检测（可扩展为结构化错误码映射）
        failure_reason = None
        success = True
        fail_keywords = ["error", "failed", "exception", "traceback", "cannot"]
        if any(kw in action_text.lower() for kw in fail_keywords):
            success = False
            failure_reason = self._extract_failure_reason(action_text)

        return TrajectoryStep(
            step_id=step_id,
            state=f"Step {step_id}",
            skills_active=active_skills,
            action=action_text,
            action_type=action_type,
            tool_name=tool_name,
            tool_input=tool_input,
            observation="",           # 由后续 tool_result 填充
            success=success,
            failure_reason=failure_reason
        )

    def _extract_failure_reason(self, text: str) -> str:
        """提取结构化失败原因，用于 SkillEvolver 的失败诊断"""
        # 可扩展为领域特定的错误码映射表
        if "keyerror" in text.lower():
            return "MISSING_KEY: 数据字段缺失"
        elif "typeerror" in text.lower():
            return "TYPE_MISMATCH: 数据类型不匹配"
        elif "filenotfound" in text.lower():
            return "MISSING_FILE: 文件路径错误"
        elif "timeout" in text.lower():
            return "TIMEOUT: 执行超时"
        else:
            # 截取前 200 字符作为 raw reason
            return f"UNKNOWN: {text[:200]}"

    def _get_tools(self) -> list:
        """返回 agent 可用工具列表"""
        return [
            {
                "name": "python_repl",
                "description": "Execute Python code",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "read_file",
                "description": "Read file content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "web_search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]

    def _execute_tool(self, response) -> list:
        """模拟工具执行，返回 tool_result"""
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 实际项目中替换为真实工具执行
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"[Tool {block.name} executed successfully]"
                })
        return results

    def _check_task_success(self, task: dict, steps: list) -> bool:
        """检查任务是否最终成功"""
        # 实际应使用 SkillLearnBench 的 deterministic verifier
        return any(s.success for s in steps[-3:])

    def run_batch(
        self,
        tasks: list[dict],
        skills: list[str],
        output_path: str
    ) -> list[Trajectory]:
        """批量运行所有任务，保存为 JSONL"""
        trajectories = []
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for task in tasks:
                print(f"Running task: {task['id']}")
                traj = self.run_task(task, skills)
                trajectories.append(traj)

                # 序列化保存
                record = {
                    "task_id": traj.task_id,
                    "domain": traj.domain,
                    "skills_used": traj.skills_used,
                    "final_success": traj.final_success,
                    "error_log": traj.error_log,
                    "steps": [
                        {
                            "step_id": s.step_id,
                            "skills_active": s.skills_active,
                            "action": s.action,
                            "action_type": s.action_type,
                            "success": s.success,
                            "failure_reason": s.failure_reason
                        }
                        for s in traj.steps
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {len(trajectories)} trajectories to {output_path}")
        return trajectories
```

### 2.4 运行轨迹生成

```bash
# 对 SkillLearnBench 所有任务生成轨迹
python -m src.run_trajectory_gen \
  --tasks data/skilllearnbench/tasks.json \
  --skills skills/ \
  --output data/trajectories/initial.jsonl \
  --model claude-sonnet-4-6
```

---

## 3. 模块二：离线 Skill 自演进

### 3.1 功能目标

输入离线轨迹，通过以下流程演进 Skill Library：

1. **G(s) 分组**（SkillClaw）：按引用 skill 聚合轨迹
2. **结构化失败诊断**：错误码 → 修改靶点
3. **SkillEvolver**：生成 K 个 proposals，涵盖内容修改 / 新建 / Trigger 更新三类
4. **PUCT 树搜索**：探索 skill 操作空间，有界编辑（SkillOpt）
5. **LLM Scorer**：对关键步重新生成 action，多维打分，**不重新执行任务**
6. **Pareto 门控**：分数提升 + 内容合法性检查

### 3.2 G(s) 分组器

**文件：`src/skill_evolver/grouper.py`**

```python
import json
from collections import defaultdict
from pathlib import Path


class SkillGrouper:
    """
    SkillClaw 的 G(s) 分组策略：
    按被引用的 skill 聚合轨迹，解决异构数据问题。
    G(s) = {τ ∈ D | τ 调用过 skill s}
    """

    def __init__(self, trajectory_path: str):
        self.trajectories = self._load(trajectory_path)

    def _load(self, path: str) -> list[dict]:
        trajs = []
        with open(path) as f:
            for line in f:
                trajs.append(json.loads(line.strip()))
        return trajs

    def group_by_skill(self) -> dict[str, dict]:
        """
        返回：
        {
          "data_analysis": {
            "successes": [...],   # 用了这个 skill 且成功的轨迹
            "failures": [...],    # 用了这个 skill 但失败的轨迹
            "error_patterns": Counter({"TYPE_MISMATCH": 3, ...})
          },
          ...
        }
        """
        from collections import Counter
        groups = defaultdict(lambda: {
            "successes": [], "failures": [], "error_patterns": Counter()
        })

        for traj in self.trajectories:
            for skill_name in traj.get("skills_used", []):
                if traj["final_success"]:
                    groups[skill_name]["successes"].append(traj)
                else:
                    groups[skill_name]["failures"].append(traj)
                    for err in traj.get("error_log", []):
                        # 取错误类型前缀，如 "TYPE_MISMATCH"
                        err_type = err.split(":")[0].strip()
                        groups[skill_name]["error_patterns"][err_type] += 1

        return dict(groups)

    def get_skill_signal_summary(self, skill_name: str) -> str:
        """为 SkillEvolver 准备的自然语言失败摘要"""
        groups = self.group_by_skill()
        if skill_name not in groups:
            return "No data available for this skill."

        g = groups[skill_name]
        n_success = len(g["successes"])
        n_fail = len(g["failures"])
        top_errors = g["error_patterns"].most_common(3)

        summary = f"""Skill: {skill_name}
Usage: {n_success} successes, {n_fail} failures
Success rate: {n_success/(n_success+n_fail)*100:.1f}%
Top failure patterns:
"""
        for err_type, count in top_errors:
            summary += f"  - {err_type}: {count} occurrences\n"

        # 附上具体失败案例（最多 3 条）
        if g["failures"]:
            summary += "\nSample failure steps:\n"
            for traj in g["failures"][:3]:
                failed_steps = [
                    s for s in traj["steps"] if not s["success"]
                ]
                for step in failed_steps[:2]:
                    summary += f"  > {step['failure_reason']}\n"

        return summary
```

### 3.3 LLM Scorer（核心：不重新执行）

这是整个系统最关键的设计。**不重新跑任务**，而是：
1. 取原轨迹中的关键失败步骤
2. 注入候选新 skill，让 LLM **重新生成该步的 action**
3. 对比新旧 action，从多个维度打分

**文件：`src/skill_evolver/scorer.py`**

```python
import anthropic
import json
from dataclasses import dataclass


@dataclass
class ScoreResult:
    overall: float              # 0-1 总分
    step_quality: float         # 生成的 action 是否清晰合理
    skill_alignment: float      # action 是否符合新 skill 的指导
    failure_avoidance: float    # 新 action 是否规避了原来的失败模式
    reasoning: str              # LLM 的打分依据


SCORER_PROMPT = """
You are evaluating whether an evolved agent skill improves action quality.

## Original Skill
{original_skill}

## Evolved Skill (candidate)
{evolved_skill}

## Task Context
{task_description}

## Original Trajectory Step (failed)
Step action: {original_action}
Failure reason: {failure_reason}

## Re-generated Action with Evolved Skill
{new_action}

## Evaluation
Score the re-generated action on three dimensions (0.0 to 1.0):

1. **step_quality**: Is the new action clear, specific, and executable?
2. **skill_alignment**: Does the new action follow the evolved skill's procedure?
3. **failure_avoidance**: Does the new action avoid the original failure pattern?

Respond ONLY in JSON:
{
  "step_quality": <float>,
  "skill_alignment": <float>,
  "failure_avoidance": <float>,
  "reasoning": "<one sentence>"
}
"""


class LLMScorer:
    """
    不重新执行任务；只让 LLM 根据新 skill 重新生成 action，再对新 action 打分。
    这保证了离线性：不依赖任何实时环境交互。
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def regenerate_action(
        self,
        task_description: str,
        evolved_skill_content: str,
        failed_step: dict,
        context_steps: list[dict]
    ) -> str:
        """
        核心：注入新 skill，让 LLM 重新生成该步的 action。
        不执行任何工具，纯生成。
        """
        context_str = "\n".join([
            f"Step {s['step_id']}: {s['action']}"
            for s in context_steps[-5:]    # 最近 5 步作为上下文
        ])

        prompt = f"""You are an agent solving a task. Based on the skill below and the context, 
generate the action you would take at the current step.

# Skill
{evolved_skill_content}

# Task
{task_description}

# Context (previous steps)
{context_str}

# Current Step
The previous attempt at this step resulted in: {failed_step.get('failure_reason', 'unknown error')}

Generate a single action to handle this step correctly.
Respond with just the action, no explanation."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    def score_evolution(
        self,
        original_skill: str,
        evolved_skill: str,
        task_description: str,
        failed_step: dict,
        new_action: str
    ) -> ScoreResult:
        """对 evolved skill 生成的新 action 打分"""

        prompt = SCORER_PROMPT.format(
            original_skill=original_skill,
            evolved_skill=evolved_skill,
            task_description=task_description,
            original_action=failed_step.get("action", ""),
            failure_reason=failed_step.get("failure_reason", ""),
            new_action=new_action
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            raw = response.content[0].text.strip()
            # 清理可能的 markdown code fence
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            return ScoreResult(
                overall=(
                    data["step_quality"] * 0.3 +
                    data["skill_alignment"] * 0.4 +
                    data["failure_avoidance"] * 0.3
                ),
                step_quality=data["step_quality"],
                skill_alignment=data["skill_alignment"],
                failure_avoidance=data["failure_avoidance"],
                reasoning=data.get("reasoning", "")
            )
        except (json.JSONDecodeError, KeyError) as e:
            # 打分失败时返回保守的低分，不让错误传播
            return ScoreResult(
                overall=0.0,
                step_quality=0.0,
                skill_alignment=0.0,
                failure_avoidance=0.0,
                reasoning=f"Scoring failed: {e}"
            )

    def batch_score(
        self,
        evolved_skill_content: str,
        original_skill_content: str,
        skill_name: str,
        trajectory_group: dict,
        top_k_failures: int = 5
    ) -> float:
        """
        对一个 skill 的所有失败案例批量打分，返回平均 Q 值。
        只对 top_k 个失败步骤打分，控制 API 调用量。
        """
        failures = trajectory_group.get("failures", [])
        if not failures:
            return 0.5   # 无失败案例，中性分

        scores = []
        count = 0

        for traj in failures:
            if count >= top_k_failures:
                break

            failed_steps = [s for s in traj["steps"] if not s["success"]]
            context_steps = [s for s in traj["steps"] if s["success"]]

            for step in failed_steps[:2]:   # 每条轨迹最多取 2 个失败步
                if count >= top_k_failures:
                    break

                # 1. 用新 skill 重新生成 action
                new_action = self.regenerate_action(
                    task_description=traj.get("task_description", ""),
                    evolved_skill_content=evolved_skill_content,
                    failed_step=step,
                    context_steps=context_steps
                )

                # 2. 打分
                score = self.score_evolution(
                    original_skill=original_skill_content,
                    evolved_skill=evolved_skill_content,
                    task_description=traj.get("task_description", ""),
                    failed_step=step,
                    new_action=new_action
                )

                scores.append(score.overall)
                count += 1

        return sum(scores) / len(scores) if scores else 0.0
```

### 3.4 SkillEvolver（Proposal 生成）

**文件：`src/skill_evolver/evolver.py`**

```python
import anthropic
from pathlib import Path
from typing import Literal
from dataclasses import dataclass


ProposalType = Literal["refine", "create", "trigger_update"]


@dataclass
class SkillProposal:
    proposal_type: ProposalType
    skill_name: str               # 目标 skill
    new_content: str              # 建议的新 SKILL.md 内容
    rationale: str                # 为什么这样改
    prior_confidence: float       # SkillEvolver 对这个 proposal 的置信度 (0-1)


EVOLVER_PROMPT = """
You are an expert at improving agent skill specifications based on failure analysis.

## Current Skill
{skill_content}

## Failure Analysis
{failure_summary}

## Task
Generate {k} diverse improvement proposals. Each proposal must be one of:
- "refine": Improve the existing skill's procedure or instructions
- "create": Create a new complementary skill to handle uncovered cases
- "trigger_update": Refine the trigger conditions to prevent misapplication

For each proposal, provide:
1. The complete new SKILL.md content
2. The rationale
3. Your confidence score (0.0-1.0)

Respond in JSON array:
[
  {{
    "type": "refine|create|trigger_update",
    "skill_name": "<name>",
    "new_content": "<complete SKILL.md>",
    "rationale": "<why>",
    "confidence": <float>
  }},
  ...
]
"""


class SkillEvolver:
    def __init__(self, skill_library_path: str, model: str = "claude-sonnet-4-6"):
        self.skill_lib = Path(skill_library_path)
        self.client = anthropic.Anthropic()
        self.model = model

        # rejected-edit buffer（SkillOpt 机制③）
        # 记录被 Pareto 门控拒绝的 proposal 方向，防止重复
        self.rejected_buffer: list[dict] = []

    def generate_proposals(
        self,
        skill_name: str,
        failure_summary: str,
        k: int = 6
    ) -> list[SkillProposal]:
        """生成 K 个 proposals，涵盖三种类型"""

        skill_path = self.skill_lib / skill_name / "SKILL.md"
        skill_content = skill_path.read_text() if skill_path.exists() else ""

        # 注入 rejected buffer 以避免重复方向
        rejected_hint = ""
        if self.rejected_buffer:
            recent_rejected = self.rejected_buffer[-5:]
            rejected_hint = "\n\n## Previously Rejected Directions (avoid these)\n"
            for r in recent_rejected:
                rejected_hint += f"- {r['rationale']}\n"

        prompt = EVOLVER_PROMPT.format(
            skill_content=skill_content + rejected_hint,
            failure_summary=failure_summary,
            k=k
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        try:
            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            proposals_data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        proposals = []
        for p in proposals_data:
            proposals.append(SkillProposal(
                proposal_type=p.get("type", "refine"),
                skill_name=p.get("skill_name", skill_name),
                new_content=p.get("new_content", ""),
                rationale=p.get("rationale", ""),
                prior_confidence=float(p.get("confidence", 0.5))
            ))

        return proposals

    def add_to_rejected_buffer(self, proposal: SkillProposal) -> None:
        """记录被拒绝的 proposal（SkillOpt 机制③）"""
        self.rejected_buffer.append({
            "skill_name": proposal.skill_name,
            "type": proposal.proposal_type,
            "rationale": proposal.rationale
        })
        # 保持 buffer 大小，防止 context 过长
        if len(self.rejected_buffer) > 20:
            self.rejected_buffer = self.rejected_buffer[-20:]
```

### 3.5 PUCT 树搜索

**文件：`src/skill_evolver/puct_search.py`**

```python
import math
from dataclasses import dataclass, field
from typing import Optional
from .evolver import SkillEvolver, SkillProposal
from .scorer import LLMScorer


@dataclass
class TreeNode:
    skill_name: str
    skill_content: str            # 该节点的 skill 内容
    parent: Optional["TreeNode"]
    proposal: Optional[SkillProposal]  # 产生该节点的 proposal

    visit_count: int = 0
    total_q: float = 0.0
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_q / self.visit_count

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        """PUCT 选择公式：Q + c * P * sqrt(N) / (1 + n)"""
        prior = self.proposal.prior_confidence if self.proposal else 0.5
        exploration = c_puct * prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


class PUCTSearch:
    """
    在 skill 操作空间中做 PUCT 树搜索。
    保守策略：小 c_puct（0.5），浅深度（3层），
    稀疏数据下防止噪声 Q 值被过度利用。
    """

    def __init__(
        self,
        evolver: SkillEvolver,
        scorer: LLMScorer,
        c_puct: float = 0.5,       # 保守策略
        max_depth: int = 3,
        n_iterations: int = 20,
        proposals_per_node: int = 6
    ):
        self.evolver = evolver
        self.scorer = scorer
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.n_iterations = n_iterations
        self.k = proposals_per_node

    def search(
        self,
        skill_name: str,
        current_skill_content: str,
        failure_summary: str,
        trajectory_group: dict
    ) -> list[SkillProposal]:
        """
        执行 PUCT 搜索，返回最优 top-k proposals。
        """
        root = TreeNode(
            skill_name=skill_name,
            skill_content=current_skill_content,
            parent=None,
            proposal=None
        )

        for _ in range(self.n_iterations):
            # 1. Select
            node = self._select(root)
            if node is None:
                continue

            # 2. Expand（只在浅深度内展开）
            depth = self._get_depth(node)
            if depth < self.max_depth:
                self._expand(node, failure_summary)

            # 3. Simulate（LLM 打分，不重新执行）
            if node.children:
                for child in node.children:
                    q = self.scorer.batch_score(
                        evolved_skill_content=child.skill_content,
                        original_skill_content=current_skill_content,
                        skill_name=skill_name,
                        trajectory_group=trajectory_group,
                        top_k_failures=3
                    )
                    # 4. Backpropagate
                    self._backpropagate(child, q)

        # 返回 top-k 叶节点，按 Q 值排序
        all_leaves = self._collect_leaves(root)
        all_leaves.sort(key=lambda n: n.q_value, reverse=True)
        return [n.proposal for n in all_leaves[:3] if n.proposal]

    def _select(self, node: TreeNode) -> Optional[TreeNode]:
        """沿最优 PUCT 路径选择节点"""
        while node.children:
            best_child = max(
                node.children,
                key=lambda c: c.puct_score(self.c_puct, node.visit_count)
            )
            node = best_child
        return node

    def _expand(self, node: TreeNode, failure_summary: str) -> None:
        """用 SkillEvolver 展开子节点"""
        proposals = self.evolver.generate_proposals(
            skill_name=node.skill_name,
            failure_summary=failure_summary,
            k=self.k
        )
        for p in proposals:
            child = TreeNode(
                skill_name=p.skill_name,
                skill_content=p.new_content,
                parent=node,
                proposal=p
            )
            node.children.append(child)

    def _backpropagate(self, node: TreeNode, q: float) -> None:
        """回传 Q 值"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_q += q
            current = current.parent

    def _get_depth(self, node: TreeNode) -> int:
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth

    def _collect_leaves(self, node: TreeNode) -> list[TreeNode]:
        if not node.children:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves
```

### 3.6 Pareto 门控

**文件：`src/skill_evolver/pareto_gate.py`**

```python
import re
from dataclasses import dataclass
from .evolver import SkillProposal
from .scorer import ScoreResult


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    score: float


class ParetoGate:
    """
    两层门控，替代数据切分：
    - 层1: LLM 分数必须高于当前基线（来自 LLMScorer）
    - 层2: skill 内容合法性检查（格式、trigger 条件完整性）
    不依赖任何 held-out 执行，完全静态。
    """

    def __init__(self, baseline_score: float = 0.5, min_score_delta: float = 0.05):
        self.baseline_score = baseline_score
        self.min_score_delta = min_score_delta  # 至少提升 5% 才接受

    def evaluate(
        self,
        proposal: SkillProposal,
        new_q_score: float
    ) -> GateDecision:
        """综合判断是否接受 proposal"""

        # 层1: Q 值必须提升
        if new_q_score < self.baseline_score + self.min_score_delta:
            return GateDecision(
                accepted=False,
                reason=f"Score {new_q_score:.3f} < baseline {self.baseline_score:.3f} + delta {self.min_score_delta}",
                score=new_q_score
            )

        # 层2: skill 内容合法性检查
        validity = self._check_skill_validity(proposal.new_content)
        if not validity["valid"]:
            return GateDecision(
                accepted=False,
                reason=f"Invalid skill content: {validity['reason']}",
                score=new_q_score
            )

        return GateDecision(
            accepted=True,
            reason=f"Accepted: score {new_q_score:.3f}, {validity['reason']}",
            score=new_q_score
        )

    def _check_skill_validity(self, content: str) -> dict:
        """
        静态合法性检查：
        - 包含必要的 SKILL.md 字段
        - trigger_conditions 非空
        - procedure 步骤非空
        不依赖执行，完全基于文本分析。
        """
        checks = {
            "has_name": "name:" in content,
            "has_trigger": "trigger_conditions:" in content,
            "has_procedure": "## Procedure" in content or "procedure:" in content.lower(),
            "not_empty": len(content.strip()) > 100,
            "has_failure_patterns": "failure" in content.lower() or "Failure" in content
        }

        all_pass = all(checks.values())
        failed = [k for k, v in checks.items() if not v]

        return {
            "valid": all_pass,
            "reason": "All checks passed" if all_pass else f"Failed: {failed}"
        }
```

### 3.7 主演进循环

**文件：`src/skill_evolver/pipeline.py`**

```python
from pathlib import Path
import json
from .grouper import SkillGrouper
from .evolver import SkillEvolver
from .scorer import LLMScorer
from .puct_search import PUCTSearch
from .pareto_gate import ParetoGate


class SkillEvolutionPipeline:

    def __init__(
        self,
        skill_library_path: str,
        trajectory_path: str,
        n_rounds: int = 5,
        model: str = "claude-sonnet-4-6"
    ):
        self.skill_lib = Path(skill_library_path)
        self.grouper = SkillGrouper(trajectory_path)
        self.evolver = SkillEvolver(skill_library_path, model)
        self.scorer = LLMScorer(model)
        self.gate = ParetoGate()
        self.search = PUCTSearch(self.evolver, self.scorer)
        self.n_rounds = n_rounds
        self.evolution_log: list[dict] = []

    def run(self) -> dict:
        """
        主循环：
        1. G(s) 分组
        2. 失败诊断
        3. PUCT 搜索
        4. Pareto 门控
        5. 更新 skill 库
        单调不退化：只接受改进版本
        """
        groups = self.grouper.group_by_skill()

        for round_id in range(self.n_rounds):
            print(f"\n=== Evolution Round {round_id + 1}/{self.n_rounds} ===")

            for skill_name, group in groups.items():
                if not group["failures"]:
                    continue   # 无失败案例，跳过

                print(f"  Processing skill: {skill_name} "
                      f"({len(group['failures'])} failures)")

                # 1. 失败摘要
                failure_summary = self.grouper.get_skill_signal_summary(skill_name)

                # 2. 读取当前 skill 内容
                skill_path = self.skill_lib / skill_name / "SKILL.md"
                if not skill_path.exists():
                    continue
                current_content = skill_path.read_text()

                # 3. 计算当前基线分数
                baseline_score = self.scorer.batch_score(
                    evolved_skill_content=current_content,
                    original_skill_content=current_content,
                    skill_name=skill_name,
                    trajectory_group=group
                )
                self.gate.baseline_score = baseline_score

                # 4. PUCT 搜索出 top proposals
                top_proposals = self.search.search(
                    skill_name=skill_name,
                    current_skill_content=current_content,
                    failure_summary=failure_summary,
                    trajectory_group=group
                )

                # 5. Pareto 门控，找到第一个通过的 proposal
                accepted = False
                for proposal in top_proposals:
                    new_score = self.scorer.batch_score(
                        evolved_skill_content=proposal.new_content,
                        original_skill_content=current_content,
                        skill_name=skill_name,
                        trajectory_group=group
                    )
                    decision = self.gate.evaluate(proposal, new_score)

                    if decision.accepted:
                        # 写入 skill 库
                        self._commit_skill(skill_name, proposal, new_score, round_id)
                        accepted = True
                        print(f"    ✓ Accepted: {proposal.proposal_type} "
                              f"(score: {baseline_score:.3f} → {new_score:.3f})")
                        break
                    else:
                        # 记入 rejected buffer
                        self.evolver.add_to_rejected_buffer(proposal)
                        print(f"    ✗ Rejected: {decision.reason}")

                if not accepted:
                    print(f"    — No improvement found for {skill_name} in round {round_id+1}")

        return {"evolution_log": self.evolution_log, "rounds": self.n_rounds}

    def _commit_skill(
        self,
        skill_name: str,
        proposal,
        new_score: float,
        round_id: int
    ) -> None:
        """写入新 skill，保留版本历史"""
        skill_dir = self.skill_lib / skill_name
        skill_dir.mkdir(exist_ok=True)

        # 备份旧版本
        current_path = skill_dir / "SKILL.md"
        if current_path.exists():
            backup_path = skill_dir / f"SKILL.v{round_id}.md"
            backup_path.write_text(current_path.read_text())

        # 写入新版本
        current_path.write_text(proposal.new_content)

        # 记录演化日志
        self.evolution_log.append({
            "round": round_id,
            "skill": skill_name,
            "type": proposal.proposal_type,
            "score": new_score,
            "rationale": proposal.rationale
        })
```

---

## 4. 模块三：实验评估

### 4.1 SkillLearnBench 评估协议

SkillLearnBench 包含 20 个真实任务，覆盖 15 个子领域，三层评估：skill 质量、执行轨迹对齐、任务结果。

三层评估对应三个指标：

| 层级 | 指标 | 来源 |
|------|------|------|
| Skill Quality | SKILL.md 完整性 + 触发条件覆盖率 | 静态分析 |
| Trajectory Alignment | action 与 skill 步骤的语义匹配度 | LLM 判断 |
| Task Outcome | 任务最终成功率 | SkillLearnBench verifier |

### 4.2 实验设置

```
Baseline:   无 skill（直接 codex）
Baseline+:  人工策划 skill（SkillLearnBench 提供）
Ours:       离线自演进后的 skill（本系统输出）
Ablation1:  去掉 PUCT，改为随机采样 proposal
Ablation2:  去掉 rejected-edit buffer
Ablation3:  去掉 G(s) 分组，直接用全量轨迹
```

### 4.3 评估器实现

**文件：`src/evaluation/evaluator.py`**

```python
import json
import anthropic
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EvalResult:
    task_id: str
    domain: str
    skill_quality_score: float     # 0-1
    trajectory_alignment: float    # 0-1
    task_success: bool
    notes: str


class SkillLearnBenchEvaluator:

    def __init__(self, benchmark_path: str, model: str = "claude-sonnet-4-6"):
        self.bench_path = Path(benchmark_path)
        self.client = anthropic.Anthropic()
        self.model = model

    def evaluate_skill_quality(self, skill_content: str, task: dict) -> float:
        """层1：静态检查 skill 文档质量"""
        checks = {
            "has_trigger": "trigger_conditions:" in skill_content,
            "has_procedure": "## Procedure" in skill_content,
            "has_examples": "## Examples" in skill_content,
            "has_failure": "## Failure" in skill_content,
            "covers_domain": task.get("domain", "") in skill_content.lower(),
            "reasonable_length": 200 < len(skill_content) < 3000
        }
        return sum(checks.values()) / len(checks)

    def evaluate_trajectory_alignment(
        self, trajectory: dict, skill_content: str
    ) -> float:
        """层2：LLM 判断轨迹步骤是否遵循 skill"""
        steps_summary = "\n".join([
            f"Step {s['step_id']}: {s['action'][:100]}"
            for s in trajectory["steps"]
        ])

        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"""Rate how well the following trajectory follows the skill (0.0-1.0).

Skill:
{skill_content[:500]}

Trajectory:
{steps_summary}

Respond ONLY with a JSON: {{"score": <float>, "reason": "<brief>"}}"""
            }]
        )

        try:
            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            return float(data["score"])
        except Exception:
            return 0.0

    def evaluate_task_outcome(self, task: dict, trajectory: dict) -> bool:
        """层3：使用 SkillLearnBench 的 deterministic verifier"""
        # 加载 benchmark 的 eval_keypoints
        keypoints_path = self.bench_path / "eval_keypoints" / f"{task['id']}.json"
        if not keypoints_path.exists():
            return trajectory.get("final_success", False)

        with open(keypoints_path) as f:
            keypoints = json.load(f)

        # 检查关键步骤是否出现在轨迹中
        trajectory_text = " ".join([
            s["action"] for s in trajectory["steps"]
        ]).lower()

        matched = sum(
            1 for kp in keypoints.get("required_actions", [])
            if kp.lower() in trajectory_text
        )
        return matched >= len(keypoints.get("required_actions", [])) * 0.7

    def run_full_evaluation(
        self,
        skill_library_path: str,
        trajectory_path: str,
        condition_name: str = "ours"
    ) -> dict:
        """完整的三层评估"""
        results = []

        # 加载轨迹
        trajectories = {}
        with open(trajectory_path) as f:
            for line in f:
                traj = json.loads(line.strip())
                trajectories[traj["task_id"]] = traj

        # 加载 benchmark 任务
        with open(self.bench_path / "tasks.json") as f:
            tasks = json.load(f)

        skill_lib = Path(skill_library_path)

        for task in tasks:
            task_id = task["id"]
            if task_id not in trajectories:
                continue

            traj = trajectories[task_id]
            domain = task.get("domain", "unknown")

            # 找到对应的 skill
            skill_content = ""
            for skill_name in traj.get("skills_used", []):
                skill_path = skill_lib / skill_name / "SKILL.md"
                if skill_path.exists():
                    skill_content += skill_path.read_text() + "\n"

            result = EvalResult(
                task_id=task_id,
                domain=domain,
                skill_quality_score=self.evaluate_skill_quality(skill_content, task),
                trajectory_alignment=self.evaluate_trajectory_alignment(traj, skill_content),
                task_success=self.evaluate_task_outcome(task, traj),
                notes=""
            )
            results.append(result)

        # 汇总
        n = len(results)
        summary = {
            "condition": condition_name,
            "n_tasks": n,
            "skill_quality": sum(r.skill_quality_score for r in results) / n,
            "trajectory_alignment": sum(r.trajectory_alignment for r in results) / n,
            "task_success_rate": sum(r.task_success for r in results) / n,
            "by_domain": self._aggregate_by_domain(results)
        }
        return summary

    def _aggregate_by_domain(self, results: list) -> dict:
        from collections import defaultdict
        domain_stats = defaultdict(list)
        for r in results:
            domain_stats[r.domain].append(r.task_success)
        return {
            domain: sum(vals) / len(vals)
            for domain, vals in domain_stats.items()
        }
```

### 4.4 实验运行脚本

**文件：`scripts/run_experiment.sh`**

```bash
#!/bin/bash
set -e

DATA_DIR="data"
SKILLS_DIR="skills"
BENCH_DIR="data/skilllearnbench"

echo "=== Step 1: Generate initial trajectories (baseline, no skill) ==="
python -m src.run_trajectory_gen \
  --tasks $BENCH_DIR/tasks.json \
  --skills $SKILLS_DIR \
  --output $DATA_DIR/trajectories/no_skill.jsonl \
  --no_skill

echo "=== Step 2: Generate trajectories with initial skills ==="
python -m src.run_trajectory_gen \
  --tasks $BENCH_DIR/tasks.json \
  --skills $SKILLS_DIR \
  --output $DATA_DIR/trajectories/with_skill.jsonl

echo "=== Step 3: Run offline skill evolution ==="
python -m src.run_evolution \
  --trajectories $DATA_DIR/trajectories/with_skill.jsonl \
  --skills $SKILLS_DIR \
  --output $SKILLS_DIR/evolved \
  --rounds 5

echo "=== Step 4: Generate trajectories with evolved skills ==="
python -m src.run_trajectory_gen \
  --tasks $BENCH_DIR/tasks.json \
  --skills $SKILLS_DIR/evolved \
  --output $DATA_DIR/trajectories/evolved_skill.jsonl

echo "=== Step 5: Evaluate all conditions ==="
python -m src.run_evaluation \
  --benchmark $BENCH_DIR \
  --conditions \
    "no_skill:$DATA_DIR/trajectories/no_skill.jsonl:$SKILLS_DIR" \
    "with_skill:$DATA_DIR/trajectories/with_skill.jsonl:$SKILLS_DIR" \
    "evolved:$DATA_DIR/trajectories/evolved_skill.jsonl:$SKILLS_DIR/evolved" \
  --output results/summary.json

echo "=== Done ==="
```

---

## 5. 项目结构

```
offskillevo/
├── README.md
├── requirements.txt
├── scripts/
│   └── run_experiment.sh
├── src/
│   ├── __init__.py
│   ├── run_trajectory_gen.py      # 模块一入口
│   ├── run_evolution.py           # 模块二入口
│   ├── run_evaluation.py          # 模块三入口
│   │
│   ├── trajectory_generator.py   # codex + skill rollout
│   │
│   ├── skill_evolver/
│   │   ├── __init__.py
│   │   ├── grouper.py             # G(s) 分组 (SkillClaw)
│   │   ├── evolver.py             # proposal 生成 + rejected buffer
│   │   ├── scorer.py              # LLM 打分 (不重新执行)
│   │   ├── puct_search.py         # PUCT 树搜索
│   │   ├── pareto_gate.py         # Pareto 门控
│   │   └── pipeline.py            # 主演进循环
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py           # SkillLearnBench 三层评估
│
├── skills/                        # 初始 skill 库
│   ├── data_analysis/
│   │   └── SKILL.md
│   ├── web_search/
│   │   └── SKILL.md
│   └── code_execution/
│       └── SKILL.md
│
└── data/
    ├── skilllearnbench/           # git clone cxcscmu/SkillLearnBench
    └── trajectories/              # 生成的轨迹文件
```

---

## 6. 环境配置

```bash
# 克隆项目
git clone <repo>
cd offskillevo

# 安装依赖
pip install anthropic>=0.40.0 pathlib dataclasses

# 克隆 SkillLearnBench
git clone https://github.com/cxcscmu/SkillLearnBench data/skilllearnbench

# 设置 API Key
export ANTHROPIC_API_KEY=your_key_here
```

**requirements.txt**：

```
anthropic>=0.40.0
python-dotenv>=1.0.0
tqdm>=4.66.0
```

---

## 7. 运行示例

```python
# 快速测试演进流程
from src.skill_evolver.pipeline import SkillEvolutionPipeline

pipeline = SkillEvolutionPipeline(
    skill_library_path="skills/",
    trajectory_path="data/trajectories/with_skill.jsonl",
    n_rounds=3,
    model="claude-sonnet-4-6"
)

results = pipeline.run()
print(f"Evolution complete. Log: {results['evolution_log']}")
```

---

## 8. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 不切分轨迹 | 全量 + G(s) 分组 | 异构稀疏数据，切分后样本不足 |
| 不重新执行 | LLM 打分替代 env rollout | 离线场景，执行代价高 |
| 保守 PUCT | c_puct=0.5，深度≤3 | 稀疏 Q 值噪声大，防过度探索 |
| Rejected buffer | 记录被拒方向 | SkillOpt 机制③，防止重复走错误方向 |
| 两层门控 | LLM 分数 + 合法性检查 | 替代 train/val 切分，不依赖数据量 |
| SKILL.md 格式 | OpenClaw 规范 | 兼容 SkillNet、AutoSkill4OpenClaw |
| 版本备份 | SKILL.v{round}.md | 支持 rollback，单调不退化 |


