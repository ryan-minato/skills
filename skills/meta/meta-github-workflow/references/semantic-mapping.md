# Contract Semantics on GitHub

This is the translation table between the management model the contract
builders design in and GitHub's objects. It is read twice: by the workflow
and specification builders at deposit time, to write their files in GitHub
terms, and by this builder when no workflow file exists and planning
objects must be decided here, or when a deposited line still names a
semantic instead of an object. This
mapping is one-directional: the contract decides *whether* a semantic
exists; GitHub decides only *how* it is expressed. A weaker but explicit
fallback is always acceptable; a semantically wrong object never is.

## The mapping

| Contract semantic | GitHub representation | Fallback, and what it loses |
|---|---|---|
| Tracked Work | Issue | — |
| Work Hierarchy | Sub-issues, only as deep as the contract's decomposition rule allows | None needed; never imitate deeper Jira-style tiers |
| Dependency | Real blocking relations (blocked-by) | Body cross-references — loses machine-readable blocking; never encode mere planned order as a dependency |
| Change Request / Draft Change | Pull request / draft PR. A draft is work in progress, never a backlog slot | — |
| Objective Boundary | Milestone — not only for releases; "CUDA 13 Support" is a valid milestone | A `goal` tracking issue alone — loses the date bucket and progress bar |
| Timebox | An iteration field, which exists only inside an opted-in Projects setup | A documented `timebox/<window>` label convention — loses native date arithmetic. **Never a milestone**: a milestone answers "what", a timebox answers "when", and reusing one for the other destroys both meanings |
| Planning Surface | GitHub Projects — always the opt-in view, never the record | Saved issue filters — lose custom fields, keep every fact |
| Type | Native issue type (organization) | Type labels (personal account) — lose single-select enforcement; say so |
| Priority | Organization `Priority` issue field | `priority/*` labels — same loss; never both homes at once |
| Severity | An organization `Severity` issue field where issue fields exist | `severity/*` labels — lose single-select. **Never the Priority home**: Priority is what the team does first, Severity is how bad the problem objectively is, and merging them loses both |
| Delivery | Release | Tag plus notes file — loses the release UI and assets |
| Source Marker | Tag | — |
| Acceptance | Required checks plus review rules, at the evidenced enforcement tier | Written convention with a named owner — record the tier honestly |
| Automation | Actions workflows producing the checks and deliveries the contract automates | A documented manual procedure with a named owner — loses machine enforcement; record it as convention, not as a check that exists |
| Ownership | CODEOWNERS plus the sync-ownership register | Plain ownership section in project knowledge |
| Deliberation | Discussion (or an RFC issue where Discussions is off) | — |
| Decision Record | A committed repository document | Never an issue: issues are lifecycle objects, and a closed issue reads as "done", not "decided" |

Omitted semantics map to nothing. A contract that omits Priority means no
priority field *and* no `priority/*` labels; a contract that omits Timebox
means no iteration field and no sprint-shaped milestones. Building the
platform object anyway reopens a settled decision.

## Authority mapping

The authority contract decides who acts; GitHub only enforces what it can:

- Marking ready (`gh pr ready`), requesting review, approving, merging, and
  releasing are contract-gated actions. At the default level they belong to
  humans; the durable project skill must gate them on the deposited policy,
  not on CI being green.
- Significant milestones (objective boundaries) are created or confirmed by
  humans; an agent proposes, a human creates or approves.
- Branch protection and rulesets enforce the *floor*, not the policy: an
  agent whose token could merge still may not, and where enforcement is
  impossible (private Free plan), the policy stands as convention with its
  tier recorded.

## When there is no workflow file

Decide planning and taxonomy through the design tree as this skill always
has, write the result as `.agents/knowledge/github-workflow.md` in GitHub terms (the
shape the workflow builder would have deposited: objects in use with what
is lost without them, objects deliberately not used with their triggers,
decomposition, triage, planning view), and record in it that the decisions
were made platform-side — a later `meta-workflow-design` run supersedes
them. Treat
agent authority as the conservative default: agents stop at draft PRs with
acceptance evidence, humans admit changes to review.
