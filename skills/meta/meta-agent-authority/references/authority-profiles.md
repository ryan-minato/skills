# Authority Profiles H0–H3

Read when selecting or changing the authority level. The profiles are
presets over one variable — where the human accountability gate sits — not
maturity grades. Each level names its price; a level whose price the
project has not paid is offered only with the gap stated.

At every level the agent may: take tasks, analyze, plan, implement, run
tests and CI, self-review, create and update draft changes, fix what it
finds, and prepare acceptance evidence. At every level the no-self-
escalation rule and the escalation conditions apply unchanged.

## H0 — Human-admitted Change (default)

The agent stops at a complete draft change plus an acceptance-evidence
report. The human accepts, rejects, or requests fixes; the human turns the
draft ready, requests formal review, and makes the integration decision.

Price: none beyond a working test/CI signal. This is the default for every
project until a human explicitly decides otherwise.

## H1 — Autonomous Author

A complete specification exists before implementation: goal, scope,
non-goals and constraints, acceptance criteria — supplied by the human up
front or, under a specification contract, written by the agent and
approved by the human at the approval gate on the draft. The agent then
owns the whole change-request preparation — plan, implement, self-review,
test, evaluate the acceptance criteria, turn the draft ready, request
review, and respond to review. The human reviews the change set as one
engineering unit and makes the integration decision.

The gate moves from review admission to integration. Price: the
specification is genuinely complete — the later the human gate, the more
complete the specification must be, and the approval gate passing is what
pays that price when the agent wrote it. The agent never approves its own
specification. Vague goals at H1 produce polished changes nobody asked for.

## H2 — Delegated Integrator

The agent may hold reviewer, approval, or merge authority — only under an
explicit human delegation naming a bounded scope, an explicit acceptance
policy, and verification requirements.

Independence default: the author agent is never the sole independent
reviewer of its own change. Its self-review, diff inspection, CI and test
evaluation, and acceptance checklist are author-side validation. The
independent reviewer is a human or a separately authorized reviewer agent.
The user may switch independence off, but only as an explicit recorded
decision — never by omission.

Outside the delegated scope, the level is H0. Price: the delegation
document itself — scope, acceptance policy, verification requirements,
independence decision — all recorded.

## H3 — Autonomous Iteration

The per-change human integration gate is lifted: within the authorized
scope the agent loops through tracked work, change, review, verification,
integration, and observation.

The accountability gate does not disappear; it moves to a **bounded
engineering boundary** the human still owns: completion of an objective
boundary, a delivery, a deployment, or another explicitly bounded outcome.
A recurring calendar check is supervision, not a boundary, and does not
qualify on its own. Where a workflow contract exists, the boundary must
name a semantic that contract actually enables.

Price, all of it: a bounded objective; acceptance criteria a machine can
reasonably evaluate; tests and CI strong enough to be the review floor;
failure containment proportional to the change-propagation mode; a working
rollback or revert path; observability that would surface a bad change; and
the confirmed escalation conditions. Refuse to record H3 while any item is
missing, unless the user explicitly accepts the named gap.
