---
name: data_analysis
version: 1.0.0
trigger_conditions:
  - "task involves analyzing structured data"
  - "user asks for statistics, aggregation, trends, or anomalies"
applicable_domains: ["data science", "analytics", "research"]
tools_required: ["python", "pandas"]
cost_estimate: low
---

## Description
Analyze structured data, compute reliable summaries, and report findings with enough context to audit the result.

## Procedure
1. Inspect file availability, schema, row count, column names, and missing values before computing metrics.
2. Convert obvious numeric, datetime, and categorical fields explicitly instead of assuming types.
3. Compute descriptive statistics and grouped aggregations that directly answer the task.
4. Check for outliers, duplicates, and inconsistent units when results look surprising.
5. Summarize the answer with the exact fields, filters, and aggregation logic used.

## Examples
<example>
Input: A CSV of monthly sales and a request for regional trends.
Action: Read the CSV, validate columns, parse month, group by region and month, compute totals and growth.
Output: Ranked trend summary with caveats for missing or invalid rows.
</example>

## Failure Patterns
- Missing columns: list available columns and map likely aliases before failing.
- Type mismatch: coerce with explicit error handling and report dropped rows.
- Large files: sample schema first, then process with chunking if needed.

