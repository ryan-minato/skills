# Overlays and Change Propagation

Read when the facts suggest an overlay, or when changes propagate beyond the
repository itself.

## Overlays

An overlay is a stable bundle of extra semantics that a real recurring need
adds on top of any base profile. It exists only because that need crosses
profiles; do not invent overlays for one project's quirk, and do not apply
one without the selecting fact.

### Operations

Selected by: the team responds to live incidents. Adds: Incident as a
tracked-work kind, Severity (kept separate from Priority), mitigation versus
corrective work, postmortems feeding decision records, and recurring
maintenance, security, migration, and regression work.

### Community Contribution

Selected by: a meaningful volume of external reports and change requests.
Adds: triage as an explicit lifecycle stage, contributor-suitability
signals, needs-info / needs-reproducer states, area ownership for routing,
and attention management. A planning surface, if any, is never forced to
contain the whole historical intake.

### Change-driven

Selected by: most changes are self-explanatory and not worth a separate
tracked-work item. Allows a change request to stand alone as the record of
work in progress. The boundary is absolute in one direction: a draft change
records work already happening, never a store of future planned work.

### Template / Scaffold

Selected by: consumers copy or instantiate this repository. Propagation is
typically **Copy**. Adds: bootstrap correctness, validation of generated
projects, quality and freshness of defaults, minimal-defaults discipline,
and reasoning about future-instance behavior. Updating the template does not
update existing instances — plan for that explicitly.

### Infrastructure-as-Code

Selected by: declarations here mutate external real state. Propagation is
typically **State-mutating**. Adds: desired state versus observed state,
change preview before apply, environment-scoped approval, drift detection,
and remediation/rollback paths. Changes to real environments are never
handled as ordinary code changes.

### Specification / Standards

Selected by: the repository's artifact is normative for others. Adds:
Deliberation and proposal stages, explicit decision authority, Decision
Records, the normative artifact as the delivery, compatibility between
versions, and supersession. Editing normative text is a governance act, not
a wording change.

## Change Propagation modes

Record exactly one dominant mode in the contract (name a secondary mode only
when it is genuinely material). The mode exists because it expresses a real
management fact — it drives compatibility policy, validation depth, rollout
and rollback plans, blast radius, required human acceptance, and how safe
automation can be.

| Mode | A change here affects… | Typical shape |
|---|---|---|
| **Local** | mostly this repository | application, internal tool |
| **Copy** | future instances created from it | template, scaffold |
| **Dependency** | consumers via versions or references | library, SDK, reusable CI component |
| **Inherited / Enforced** | governed projects directly, through inheritance or enforcement | org governance, shared policy, central workflow |
| **State-mutating** | external real state the declarations describe | IaC, GitOps, cluster configuration |

The further right of Local the mode sits, the more a change costs its
consumers, and the stronger the acceptance, validation, and rollback story
the contract must demand.
