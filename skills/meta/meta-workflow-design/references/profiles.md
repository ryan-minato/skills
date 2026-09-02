# Base Workflow Profiles

Read when matching evidenced facts to a base profile, and when phrasing the
driver question with a recommendation. A profile is a composition strategy
over the management model — which semantics a project of this kind normally
earns — never a configuration template. The selecting facts matter more than
the label: recommend the profile whose facts the inspection actually found,
and let the user overrule.

Every profile lists what it does **not** force. Those lines are load-bearing:
they are what keeps a profile from becoming a template stamped onto every
project that matches its name.

## A — Maintainer-driven

Selected by: one core maintainer (perhaps a few collaborators), no fixed
development cadence, work rhythm set by the maintainer. Typical: personal
projects, small OSS, CLIs, libraries.

Minimum: Tracked Work, Change Request, Acceptance.
Open decisions: Objective Boundary, Planning Surface, Work Hierarchy,
Priority, Status — each only when a fact argues for it.

Objective boundaries follow the project's actual shape: a piecemeal project
needs none; a project with evident evolution themes ("CUDA 13 Support",
"Plugin System") may adopt them. **Not forced:** any planning surface, any
timebox, any status beyond open/closed — and neither the presence nor the
absence of objective boundaries.

## B — Release-driven

Selected by: the primary staged outcome is a consumable version — SDKs,
libraries, frameworks, CLIs, packages. Release history and versioning
discipline in the repository are the evidence.

Emphasizes: Objective Boundary, Delivery, Source Marker, compatibility
policy, release-readiness acceptance.

**Not forced:** an objective boundary being identical to a release
("Python 3.14 Compatibility" may span releases), a timebox, or a planning
surface.

## C — Product iteration

Selected by: a continuously running product — SaaS, website, long-lived
service, enterprise application — where the core loop is changing and
observing a live system.

Scale decides the structure, not the profile name:

- **Small** (a couple of people): Tracked Work, Change Request, deploy-as-
  delivery. No dedicated planning surface.
- **Medium**: add a Planning Surface only when parallel work makes a
  filtered list insufficient.
- **Large**: the planning surface may span repositories to plan one system.

**Not forced:** a board-like surface of any kind, a timebox, or estimation
ceremony. A two-person product on a filtered work list is a correct
instance of this profile.

## D — Research / Experiment

Selected by: the repository exists to answer questions — ML training, data
science, research code, experiment repos.

Emphasizes: the research question as tracked work, experiments with
provenance and reproducibility, recorded conclusions. Closing an experiment
does not mean it succeeded: negative and inconclusive results are valid,
complete outcomes and are recorded as such. The repository may carry
experiment history, but large artifacts and checkpoints are not forced into
version control.

**Not forced:** production-software ceremony — release discipline, status
ladders, or planning surfaces — unless the project also ships software.

## E — Shared Infrastructure / Governance

Selected by: the repository's value is what it provides to *other*
repositories, developers, or automation — shared CI, organization workflow,
policy, reusable automation, governance configuration.

Emphasizes: consumer scope, compatibility, blast radius, rollout and
rollback plans, enforcement tiers, and Change Propagation (typically
Inherited / Enforced).

**Not forced:** the consumers' own workflow structures; this profile governs
the shared repository, not the projects that consume it.

## Choosing

Recommend exactly one profile, name the selecting facts, and present the
runners-up only if the facts genuinely support two readings — then the
choice is the user's. A profile mismatch surfaces later as semantics nobody
uses or coordination nobody has; either is a trigger to redesign, not to
bolt on structure.
