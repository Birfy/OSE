---
name: web_search
version: 1.0.0
trigger_conditions:
  - "task requires current or external information"
  - "user asks to verify facts, sources, prices, schedules, laws, or recent events"
applicable_domains: ["research", "current events", "fact checking"]
tools_required: ["web_search"]
cost_estimate: medium
---

## Description
Search and synthesize external information with source-aware reasoning and date sensitivity.

## Procedure
1. Identify which facts are unstable or require source attribution.
2. Search for primary or authoritative sources before secondary summaries.
3. Compare publication dates, event dates, and jurisdiction or locale constraints.
4. Cross-check important claims across independent sources when stakes are high.
5. Answer with concise synthesis and cite the sources used.

## Examples
<example>
Input: User asks for the latest product pricing.
Action: Search the vendor page first, then corroborate with release notes or support pages.
Output: Current price, date checked, and caveats about region or plan.
</example>

## Failure Patterns
- Stale result: prefer sources with explicit recent update dates.
- Unclear entity: disambiguate names and locations before synthesizing.
- Conflicting sources: report the disagreement and favor primary sources.

