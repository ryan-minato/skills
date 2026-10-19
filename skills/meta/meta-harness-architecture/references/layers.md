# Layer Calibration

Per-layer guidance for a full harness design or a recalibration. The scale
is the one in SKILL.md: omitted (nothing built), light (a sentence or an
entry), medium (a dedicated file or configuration, enforced by convention),
thick (a check, hook, CI job, or isolation that machines enforce).

## The five variables

Each layer's rating comes from the variables that push it, not from the
project's type:

| Variable | Pushes | Direction |
|---|---|---|
| Error cost | hard gates, environment and sandbox | the more a wrong change breaks, the thicker the machine checks and the isolation |
| Maintenance horizon | entropy management, knowledge and routing | the longer the code lives, the more copies drift; a short-lived project never reaches the point where cleanup pays |
| Environment isolation | hard gates, authorization boundary (inverse) | the stronger the sandbox, the looser the gates and the wider the allow list can be, because a wrong call is absorbed |
| Agent autonomy | authorization boundary, specification and completion criteria, hard gates | the longer agents run unattended, the more they need pre-authorization, checkable done criteria, and machine feedback |
| Team size | hard gates, knowledge and routing | the more people, the less a convention holds; oral agreements become files and checks |

Task type is one typical combination of these, not an input. Two presets
follow; start from the closer one and deviate per layer.

## The two presets

| Layer | Exploratory (demo, PoC, research run, early prototype) | Engineering (production evolution, shared trunk, maintenance) |
|---|---|---|
| Environment and sandbox | light to medium: secret isolation and a basic sandbox, no network allow list | thick: container or VM, security rules hard read-only, session-log destination decided |
| Knowledge and routing | light, expanding: external API and SDK pointers, domain background, a quick-start script; few style rules | medium to thick, routing and constraining: layout, layer boundaries, procedures, sync rules |
| Hard gates | light: secret scan and a smoke run; strict format, coverage, and architecture lint off | thick: type check, lint, tests, hooks, CI gates all on |
| Authorization boundary | light: wide allow list, humans only at acceptance; one branch per parallel attempt | medium to thick: written policy, pre-authorized irreversible actions, a fixed pending-decisions shape |
| Specification and completion criteria | light: intent and final effect aligned, implementation left open; branches or worktrees for parallel attempts | thick: data structures, reused modules, interface contracts, change scope, a judgeable scenario per requirement |
| Entropy management | omitted to light: one sync sentence in the entrypoint | thick: self-checks in CI, one source of truth with two-way sync rules, periodic cleanup |

Deviate per layer by asking, for each one, whether its error cost,
horizon, and autonomy match the project's overall profile. A research
project's data pipeline (high error cost, long horizon) takes the
engineering column for hard gates and entropy management and the
exploratory column elsewhere; a production repository's `experiments/`
directory takes the reverse; a solo, fully paired production project
(small team, low autonomy) takes knowledge and authorization one step
thinner than the engineering column and keeps the gates.

## Environment and sandbox

Decide isolation first; it sets how much autonomy the other layers can
grant. Fully isolated (container or VM holding everything the agent
touches): minimal approval, secrets outside reach, a broken environment
rebuilt rather than repaired. Sandboxed host (reads beyond the workspace,
writes only inside it): agents escalate on their own judgment. Unisolated
host: per-command approval, because agents can damage the host or reach
confidential data. Prefer a reproducible isolated environment when the
stack allows one. Capability tools — driving the running software, reading
logs and metrics, querying telemetry — belong here; add the ones current
work needs, since each unused tool is description noise in every session.
Whether an agent role needs the strict tier, and the sandbox audit, are in
`security-boundary.md` (path relative to this skill's `references/`).

## Knowledge and routing

The entrypoint, its routing table, knowledge files, architecture
documents, project skills, and information tools: everything that expands
what agents know, each behind a pointer with a precise condition rather
than preloaded. Implementation constraints live here too — language,
paradigm, module architecture, dependency direction — and they shrink the
agent's freedom to explore, so calibrate by how settled the requirements
are, how complex the structure is, and how long the code must live: nearly
none for one-off work, documented boundaries and enforced dependency
direction for long-lived or complex structure. When the user has made
design decisions, record them as fixed. The practice is in `entrypoint.md`,
`knowledge.md`, and `project-skill.md`.

## Hard gates

Formatters, linters, type checks, tests, secret scans, and custom checks,
in hooks and CI. They narrow output variance without shrinking exploration,
but every future change pays to satisfy them: lean on the language's static
checking first; decide test granularity deliberately rather than defaulting
to maximal coverage; and make every custom check fail with what failed, why
it matters, and the likely fix, naming the file and printing no traceback.
Repository safety — keeping secrets and private data out of the tree with
scanning hooks and commit rules — is this layer, calibrated by leak risk
and the collaboration boundary; security of the code itself belongs to
tests. Output noise is in `management-code.md`.

## Authorization boundary

What an agent may do inside what it can do: allow and deny lists,
pre-authorized irreversible actions, the pending-decisions rule, and the
workflow tools (ticket system, review flow, messaging) the team has
delegated, each with a constraint that says how the team expects it used.
Add a workflow tool only when the delegation and the access are real;
otherwise document the human handoff. The practice is in
`authorization.md`.

## Specification and completion criteria

Target constraints give agents a durable understanding of what the project
is for. Every project needs at least a written goal; beyond that, match the
form to what the project exposes — a public interface for code consumed by
code, API behavior for a service, user stories or visual references for
user-facing software — and add no contract form for a surface the project
lacks. When the project works from written specifications, those are this
layer's form: the harness points to them and never restates a requirement.
A specification answers four things: how completion is judged by a check
the agent itself can run, what is not done, what is delivered, and which
irreversible actions are pre-authorized.

## Entropy management

Sync rules, drift reporting, harness self-checks, and periodic cleanup. The
mechanisms are in `maintenance.md`, `periodic-reclamation.md`, and
`harness-checks.md`.

## Hardening and thinning

Thickness moves in both directions.

Hardening: when a prototype graduates to maintained work, run one explicit
pass — extract the stable layout into the entrypoint and architecture
document; add static types, linters, and tests, and lock the security
control plane; delete the exploration-era scripts and context and install
the sync mechanism. Thinning: an over-thick harness never fails; it makes
agents slower and more conservative. The signal is a constraint that no
longer catches anything — a lint rule for a bug fixed long ago, a skill for
a migration that finished, a routing row for a module that was removed.
Periodic reclamation lists these beside stale content.
