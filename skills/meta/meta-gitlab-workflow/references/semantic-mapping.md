# Contract Semantics on GitLab

This is the translation table between the management model the contract
builders design in and GitLab's objects. It is read twice: by the workflow
and specification builders at deposit time, to write their files in GitLab
terms, and by this builder when no workflow file exists and planning
objects must be decided here, or when a deposited line still names a
semantic instead of an object. The
mapping is one-directional: the contract decides *whether* a semantic
exists; GitLab decides only *how* it is expressed — and semantics win over
names, because GitLab has objects whose names invite wrong mappings. A
weaker but explicit fallback is always acceptable; a semantically wrong
object never is. Every row is capability-gated: probe tier, version, and
configuration on the live instance rather than assuming the feature.

## The mapping

| Contract semantic | GitLab representation | Fallback, and what it loses |
|---|---|---|
| Tracked Work | Issue | — |
| Work Hierarchy | Epic → Issue → Task, only as deep as the contract's decomposition rule allows. A Task is work that independently earns an assignee, state, or dependency; an implementation TODO stays a checklist item, never a Task | Without Epics (tier): a parent issue with linked issues — loses roll-up |
| Dependency | Blocking issue links (`blocks` / `is blocked by`) | Related links plus body text — loses machine-readable blocking; never encode mere planned order as a dependency |
| Change Request / Draft Change | Merge request / draft MR. A draft is work in progress, never a backlog slot | — |
| Objective Boundary | Milestone — not only for releases. The OKR object literally named `Objective` is **not** this semantic's home by default; map to it only when the project actually runs OKRs there | A labeled tracking issue — loses the date bucket and burndown |
| Timebox | Iteration (with a cadence) | A documented `timebox::<window>` label convention — loses date arithmetic. **Never a milestone**: milestone answers "what", timebox answers "when", and reusing one for the other destroys both meanings |
| Planning Surface | Board, work-items list, group board, or roadmap — all views over the same items, deletable without loss | Filtered issue lists — lose columns, keep every fact |
| Type | Scoped `type::*` labels (work-item types are the platform's containers, and their names never rewrite the contract's `Type` attribute) | Plain `type/*` labels where scoped labels are unavailable — lose mutual exclusion; say so |
| Priority | Scoped `priority::*` labels. **Never Weight**: Weight is effort estimation, orthogonal to priority | Plain `priority/*` labels — same loss |
| Severity | Native severity on Incident work items; scoped `severity::*` labels for other work | Plain `severity/*` labels — lose mutual exclusion. **Never the Priority home and never Weight**: Priority is what the team does first, Severity is how bad the problem objectively is |
| Delivery | Release | Tag plus notes file — loses the release UI and assets |
| Source Marker | Tag | — |
| Acceptance | Approval rules plus required pipeline status, at the evidenced tier | Written convention with a named owner — record the tier honestly |
| Automation | CI/CD pipelines producing the checks and deliveries the contract automates | A documented manual procedure with a named owner — loses machine enforcement; record it as convention, not as a check that exists |
| Ownership | CODEOWNERS plus protected refs | Plain ownership section in project knowledge |
| Deliberation | An issue or design discussion thread | — |
| Decision Record | A committed repository document | Never an issue: a closed issue reads as "done", not "decided" |

Omitted semantics map to nothing: a contract that omits Priority gets no
`priority::*` labels; one that omits Timebox gets no iteration cadence and
no sprint-shaped milestones; one that omits a Planning Surface gets no
board. Building the object anyway reopens a settled decision.

Group hierarchy is not a planning tool: Groups and Subgroups are stable
namespace, ownership, governance, and configuration scope. Dynamic or
cross-project planning needs are answered by hierarchy, objective
boundaries, and planning surfaces — never by restructuring Groups.

## Authority mapping

The authority contract decides who acts; GitLab only enforces what it can:

- Marking an MR ready, requesting review, approving, merging, and releasing
  are contract-gated actions. At the default level they belong to humans;
  the durable project skill gates them on the deposited policy, not on a
  green pipeline.
- Significant milestones (objective boundaries) are created or confirmed by
  humans; an agent proposes, a human creates or approves.
- Protected branches and approval rules enforce the *floor*, not the
  policy: an agent whose role could merge still may not, and where a tier
  cannot enforce a rule, the policy stands as convention with its tier
  recorded.

## When there is no workflow file

Decide planning and taxonomy through the design tree as this skill always
has, write the result as `.agents/knowledge/gitlab-workflow.md` in GitLab terms (the
shape the workflow builder would have deposited: objects in use with what
is lost without them, objects deliberately not used with their triggers,
decomposition, triage, planning view), and record in it that the decisions
were made platform-side — a later `meta-workflow-design` run supersedes
them. Treat
agent authority as the conservative default: agents stop at draft MRs with
acceptance evidence, humans admit changes to review.
