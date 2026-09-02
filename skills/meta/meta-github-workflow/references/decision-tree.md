# Design Tree and Consensus Gates

Read before eliciting project workflow, goal, or human-authored issue
requirements.

## The tree

Maintain four collections: **Facts** (evidenced), **Decisions** (user-owned
choices with their current answer), **Dependencies** (edges from
prerequisites to the decisions they unlock), and the **Frontier** (every
unanswered decision whose prerequisites are resolved).

1. Explore every frontier item that is a discoverable fact. Dispatch
   clean-context subagents for independent read-only research when
   available; never ask the user for anything the repository or GitHub can
   prove.
2. Remove resolved facts and recompute the frontier.
3. Ask the entire remaining frontier in one numbered round; state why each
   decision matters and attach one recommendation reasoned from the facts.
   The user decides.
4. Apply answers, recompute, and repeat. A question whose answer depends on
   an unanswered question in the same round belongs to the next round.

When answers conflict, add a trade-off decision instead of silently
choosing. Do not write an interim tree into the repository: before
consensus it is a working model, not an agreed source of truth.

## First frontier

A decision the target's workflow or authority contract
(`.agents/knowledge/project-workflow.md` and
`.agents/knowledge/agent-authority.md` by default; the entrypoint's pointers
are authoritative on location) already settles — planning method,
cadence objects, hierarchy, priority or status axes, agent autonomy — is a
fact, not a frontier item: map it per
[semantic-mapping.md](semantic-mapping.md) and never re-ask it.

After facts are gathered, the first user-owned frontier normally includes:

1. Enforcement posture — given the evidenced visibility, plan, and org
   policy, what may actually block a merge. On a private Free repository
   nothing can; the user chooses advisory-only checks, going public,
   upgrading, or a local pre-push gate.
2. Automation boundary on people-facing objects: may a workflow label,
   comment on, close, or assign a contributor's issue or PR — and which.
3. Planning method: tracking issues + milestones (default), no cadence
   object, or an explicit GitHub Projects opt-in. Scrum without Projects
   has no sprint object — present that consequence, not a broken Scrum.
   In an organization this is a separate question from the taxonomy: types
   and fields are already the default there and do not need a board.
4. Review shape and whether outside contributions are accepted.
5. Agent autonomy: supervised or unattended, and its approval boundary.
6. Third-party action policy (default: first-party only, SHA-pinned).
7. Secret and deploy authority: which workflows may hold a secret or OIDC
   identity, and who approves environment deploys.
8. Billed minutes — ask only when the repository is private.
9. Runner substrate — ask only for GHES or when self-hosted runners exist.

Resolve organization-versus-personal ownership before any taxonomy
question: issue types and issue fields exist only in organizations, and
that fact changes the type and priority axes. Where they exist they are
the default home for both axes, so the remaining taxonomy questions are
about area, status, and community labels — not about inventing a type or
priority label set.

## Goal consensus gate: tracking issue plus milestone

A long-horizon goal is a human-owned future state. Resolve vision,
beneficiaries, observable completion, non-goals, included work,
dependencies, ownership, and any designed dates through the round loop.
Then deposit the agreement where GitHub can hold it: a **tracking issue**
carries the vision, non-goals, acceptance, and the sub-issue list — the
milestone stores only title, description, and due date, so it serves as
the date bucket and links to the tracking issue. If dates are absent,
derive a suggested window from scope, dependency depth, and historical
throughput; offer it as a decision, never a commitment. Milestones are
repository-scoped and flat: a cross-repository goal needs one tracking
issue plus per-repository milestones. Close milestones, never delete them —
deletion silently detaches every issue.

Do not draft or create the tracking issue or milestone until every branch
is resolved or the user explicitly says the information is sufficient, then
present the complete draft for confirmation. Creation requires a separate
approved write.

## Human-authored issue gate

Use the same loop, scaled to the object. Resolve outcome or facts, context,
acceptance, scope, relationships, type, priority, labels, milestone,
assignee choice, and confidentiality — type and priority meaning the native
type and field in an organization, and their label stand-ins elsewhere. Suggest decomposition into sub-issues when one issue
contains independently valuable outcomes. Do not publish before the user
confirms the complete draft.
