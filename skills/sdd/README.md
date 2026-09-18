# sdd

[中文](README.zh.md)

Spec-driven development in three layers that change independently: the
**methodology** (`spec-driven-development` — what the practice is, the
specify–clarify–approve–tasks–implement–verify loop, the approval package,
how specs meet issues and pull or merge requests), and one skill per
**framework** (`openspec-workflow`, `spec-kit-workflow` — that tool's
records, its approval package, what completes a change before the request
is ready, and the request automation it installs per platform). The
project's own rules — level, tool, approval mode, archive executor — are
settled and deposited by the disposable `meta` catalog's
`meta-spec-workflow`, which stays framework-agnostic and hands tool
adoption and automation to the framework skill here.

Install the methodology skill and the skill for the framework the project
runs; each works alone and hands off to its siblings by role.

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skills

| Skill | Description |
|---|---|
| [spec-driven-development](spec-driven-development/) | Work from written specifications, tool- and platform-agnostically: judge when spec-driven development pays and at which level, run the loop under the project's contract — publish the draft once the approval package is complete (the specification plus a bounded design when one is warranted) and stop, reconcile when the gate owner closes the discussion, write tasks only then, verify, archive inside the request before it is ready — apply the documented defaults when no contract exists, keep acceptance criteria in the spec only, convert a prototype or brownfield codebase one change at a time without backfilling specs, and hand the project's framework to its `sdd` skill and the project's rules to the `meta` catalog's builder. |
| [openspec-workflow](openspec-workflow/) | Run OpenSpec changes through pull or merge requests: the approval package (proposal, delta specs, design when warranted; tasks after approval), the spec-less marker, the strict validator's moments, the implementer's archive command inside the request, the `/spec show` and `/spec status` comment commands, the archived and progress status labels, and the automation that installs them on GitHub (a check that fails a ready request with an unarchived change, the comment and label workflows) or GitLab (regular and manual jobs); ships `spec_changes.py`. |
| [spec-kit-workflow](spec-kit-workflow/) | Run Spec-Kit features through pull or merge requests: the approval package (the specification and the plan; tasks after approval), completion before ready as every task ticked since the kit archives nothing, the `/spec` comment commands and progress labels over touched features, and the automation that installs them on GitHub or GitLab; ships `spec_kit_features.py`. |
