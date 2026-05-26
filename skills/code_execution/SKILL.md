---
name: code_execution
version: 1.0.0
trigger_conditions:
  - "task requires running code, tests, scripts, or data transformations"
  - "user asks to debug, validate, or reproduce a software behavior"
applicable_domains: ["software engineering", "automation", "debugging"]
tools_required: ["python", "shell"]
cost_estimate: low
---

## Description
Use code execution to validate assumptions, reproduce failures, and produce deterministic artifacts.

## Procedure
1. Inspect project structure and existing commands before adding new tooling.
2. Reproduce the smallest relevant behavior with local commands or tests.
3. Prefer existing scripts, package managers, and test harnesses over ad hoc execution.
4. Capture command outputs that affect the conclusion.
5. Keep generated files scoped to the requested task and cleanly separated from source files.

## Examples
<example>
Input: A failing Python function and expected output.
Action: Run the focused test, inspect traceback, patch the function, and rerun the same test.
Output: Code change plus verification command and result.
</example>

## Failure Patterns
- Missing dependency: report the install command and continue with static checks if possible.
- Flaky test: rerun focused commands and note nondeterminism.
- Unsafe command: ask for approval before destructive or external side effects.

