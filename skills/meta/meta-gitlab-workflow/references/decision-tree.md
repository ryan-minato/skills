# Consensus Design Tree

Read before eliciting project workflow, milestone, or human-authored work-item
requirements. The tree prevents downstream questions and actions from being
built on guessed answers.

## Model

Maintain four collections in the conversation or a scratch file outside the
target project until consensus:

- **Facts:** evidenced repository or GitLab state, with its source.
- **Decisions:** user-owned choices and their current answer.
- **Dependencies:** directed edges from prerequisite facts/decisions to the
  decisions they unlock.
- **Frontier:** every unanswered decision whose prerequisites are resolved.

Do not write an interim tree into the project. Before consensus it is a working
model, not an agreed source of truth.

## Round loop

1. Explore every frontier item that is a discoverable fact. When clean-context
   subagents exist, dispatch independent read-only searches without blocking
   unrelated frontier questions.
2. Remove resolved facts from the question set and recompute the frontier.
3. Ask the entire remaining frontier in one round. Number questions, state why
   each decision matters, and provide one recommended answer reasoned from the
   known facts. The user retains the decision.
4. Wait. Do not ask a question whose prerequisites include another unanswered
   question from the same round.
5. Apply the answers, record trade-offs, reshape pruned branches, unlock new
   decisions, and repeat.

When answers conflict, add a trade-off decision instead of silently choosing a
winner. When the user changes an earlier answer, invalidate and recompute every
descendant.

## First frontier

A decision the target's workflow, authority, or specification contract
(`.agents/knowledge/project-workflow.md`,
`.agents/knowledge/agent-authority.md`, and
`.agents/knowledge/spec-workflow.md` by default; the entrypoint's pointers
are authoritative on location) already settles — planning method,
cadence objects, hierarchy, priority or status axes, agent autonomy, where
specifications live and who approves them — is a fact, not a frontier
item: map it per [semantic-mapping.md](semantic-mapping.md) or
[spec-expression.md](spec-expression.md) and never re-ask it.

After facts are gathered, the first user-owned frontier normally includes:

1. Planning system: Kanban, Scrum, another maintained board workflow, or no
   board. Recommend the smallest system the team will actually groom.
2. Team/review ownership and whether outside contributions are accepted.
3. Human, agent, or mixed creation of work items and merge requests.
4. Supervised versus unattended agent operation and its approval boundary.
5. Confidentiality boundary and where sensitive reports are allowed.
6. Whether the available runner capacity may be used for project CI.

Only ask an item when project evidence cannot answer it.

## Milestone consensus gate

Treat a milestone as a human-owned future state, not merely a date bucket.
Resolve vision, beneficiaries, observable completion, non-goals, included work,
dependencies, ownership, relationship to releases/iterations, and whether it
has a designed start or end date. If dates are absent, derive a suggested
window from scope, dependency depth, team capacity, and historical throughput;
offer it as a decision, never a commitment.

Do not draft or create the milestone until every branch is resolved or the
user explicitly says the information is sufficient, then present the complete
milestone for confirmation. Creation requires a separate approved write.

## Human-authored task, issue, and incident gate

Use the same loop, scaled to the object. Resolve outcome/facts, context,
acceptance, scope, relationships, labels, milestone, assignee choice, and
confidentiality. Suggest decomposition when one task contains independently
valuable or separately verifiable outcomes. Do not publish before the user
confirms the complete draft.
