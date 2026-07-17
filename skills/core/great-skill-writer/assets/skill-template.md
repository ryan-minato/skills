---
name: [skill-name — must exactly equal the directory name]
description: >
  [Short capability lead-in, third person: front-load the leading word — the
  action verb or domain concept that should trigger this skill.] Use when
  [the payload — give triggers most of the space: one trigger per branch,
  drawn from every signal type that applies (direct and indirect wording,
  scenarios, observed agent behavior, the shape of the material at hand),
  plus the adjacent tasks it must not fire on].
# license: Apache-2.0
# compatibility: [Only for real environment requirements, e.g. "Requires uv". Max 500 chars.]
# metadata:
#   author: [your-org]
#   version: "1.0"
---

# [Skill Title]

[One line: what this skill makes predictable.]

## [Workflow name]

1. [First action, imperative.]
2. [Next action whose execution benefits from an explicit boundary.]
   Done when: [checkable criterion; exhaustive where thoroughness matters].

<!-- Add Done when only when a meaningful, checkable boundary improves how
the agent executes or judges a step. -->

<!-- Conditional reference pointer — one per branch-specific file:
Read [references/topic.md](references/topic.md) when [precise condition].
-->

<!-- Script link — at first mention, before any invocation example:
Run [`scripts/tool.py`](scripts/tool.py) to [purpose]:
`uv run scripts/tool.py --flag VALUE`
-->
