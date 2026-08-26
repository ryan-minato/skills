# Security, Ownership, and Repository Governance

Read when designing CODEOWNERS, protected branches or tags, approvals, push or
merge rules, security scanning or policies, secrets, dependency updates, audit,
or unattended authority.

Resolve each selected capability live through `llms.txt`; search for its exact
topic and inspect tier, offering, history, prerequisites, API, and target
instance support. Never describe a convention-only control as platform
enforcement.

## Ownership and review

Rework `assets/codeowners-template` only from real paths and real GitLab
users/groups. Define a default owner only when one team accepts the fallback. Every new top-level
ownership boundary must update CODEOWNERS in the same MR. Validate syntax and
read GitLab's parsed result because invalid lines can be ignored.

Agree who may push and merge protected branches, create protected tags, approve
MRs, dismiss approvals, and deploy. Configure the smallest rules that enforce
the approved model. Require passing pipelines only after those pipelines are
working; key approval or status rules to stable job names and record the
coupling.

## Security feedback

Inventory dependency, secret, static, container, license, IaC, and dynamic
security capabilities relevant to the stack. Add only scanners with a triage
owner, response SLA, false-positive path, and available runner capacity. An
unowned alert stream is not a control.

Treat scanning and dependency updating as separate decisions. When automated
updates are selected, verify the current GitLab-compatible tool from its
first-party docs, pin/configure it for detected ecosystems, and record grouping,
schedule, auto-merge limits, and broken-update ownership.

## Secrets and deployment authority

Use least-privilege project/group access tokens or job-token capabilities,
protected and masked variables, environment scopes, and protected runners.
Record who rotates each credential and how jobs avoid printing it. Do not put
secret values, private addresses, personal data, or internal incident detail in
committed harness files, issues, MRs, comments, artifacts, caches, or logs.

## Apply and audit

For every remote setting, capture current state, exact proposed delta, tier and
permission preconditions, rollback, and readback command. Review and approve the
batch before applying. Store the intended setting—not a token—in reachable
project knowledge, including its owner and drift-check trigger.

Unattended agents receive only explicitly delegated operations with isolation,
validation, audit, stop, and rollback paths. Repository protection changes,
credential changes, security-policy changes, production deployments, and merge
authority remain human-approved by default.
