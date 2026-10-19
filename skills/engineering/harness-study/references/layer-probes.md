# Layer Probes

Where each harness layer lives in a reference project, and what counts as
evidence that a mechanism is live. Framework and platform names are
examples of where to look; a reference may use any of them or none.

## Environment and sandbox

Look in: `.devcontainer/`, `Dockerfile` and compose files used for
development, VM or codespace configuration, CI job containers, framework
sandbox settings, environment files that scope credentials, read-only
mounts of policy or hook directories.

Live when: the container or VM definition is what CI or the documented
setup command builds; credentials in the setup are scoped tokens rather
than mounted home directories; the entrypoint or setup document names it.

Record: what the sandbox isolates (filesystem, network, credentials, host
sockets) and what it exposes.

## Knowledge and routing

Look in: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursor/rules/`,
`.github/copilot-instructions.md`, `.claude/rules/`, subdirectory
entrypoints, `ARCHITECTURE.md`, knowledge or docs directories the
entrypoint points to, project skills under `.claude/skills/`,
`.agents/skills/`, `.github/skills/`, or another configured root.

Live when: the entrypoint's pointers resolve to files that exist; the
commands it names exist as recipes or scripts; a knowledge file is
reachable from a pointer; a project skill sits under a root the recorded
framework loads. A skill description that matches work the history shows
proves relevance, not execution: record the skill as reachable, and its
use as unverified unless an entrypoint pointer, a session log, or a
direct invocation shows it ran.

Record: entrypoint length and split (always-on rules versus routing
table), every rule stated more than once with both locations, how deeper
material loads (routing table, path rule, hook,
description match), the knowledge-versus-skill split, and the carriers
that are framework-dependent.

## Hard gates

Look in: `.pre-commit-config.yaml`, `.husky/`, `lefthook.yml`, git hook
paths set in configuration, the task runner (`justfile`, `Makefile`,
`package.json` scripts, `noxfile.py`, `Taskfile.yml`), CI workflows,
linter and type-checker configuration, custom check scripts, architecture
boundary tests.

Live when: the hook manager is installed by the documented setup; the CI
workflow runs on pull requests and is required (ruleset or branch
protection, when visible); the recipe the entrypoint names is the one CI
runs; the check's last run is recent.

Record: which checks run at which stage (editor, hook, CI), their output
mode (quiet flags, short tracebacks), and whether custom failures name the
fix.

## Authorization boundary

Look in: framework permission files (`.claude/settings.json` allow, deny,
and ask lists, equivalents in other frameworks), hooks that intercept
commands, an authorization or agent-authority document the entrypoint
names, plan or spec templates with a pre-authorized-actions field,
protection rules on the platform when visible.

Live when: the permission file is committed and the entrypoint or setup
names it; the intercepting hook is registered; the authority document is
pointed to from the entrypoint.

Record: allow and deny shapes, whether ask entries exist and why, how
irreversible actions (push, publish, comment, delete) are pre-authorized,
whether blocked work is batched for a human, and whether the policy
protects itself from agent edits.

## Specification and completion criteria

Look in: `openspec/`, `specs/`, `.specify/`, `PLAN.md` or `SPEC.md`
conventions, issue and pull request templates, a specification workflow
document, plan-mode instructions in the entrypoint.

Live when: the directory holds records that match merged changes; the
template fields are filled in recent pull requests; the validator runs in
CI.

Record: the carrier (framework, document, conversation), what a record
must contain (acceptance scenarios, non-goals, deliverable, pre-authorized
actions), and who approves.

## Entropy management

Look in: keep-in-sync sections of the entrypoint, maintenance or
harness-maintenance documents, harness validators and mirror checks in
the check scripts, labels or issue forms for drift, sync skills, periodic
cleanup instructions, hooks that remind after an edit.

Live when: the validator runs in CI or a hook; the sync rule names a
concrete trigger and a target file; drift issues exist in the history;
the mirror check fails on a deliberate test edit (do not run it against
the reference's tree; read what it checks).

Record: which facts have one source of truth, which are mirrored and how
the mirror is checked, forward and reverse triggers, and whether anything
reclaims excess thickness.

## Liveness evidence, in general

Strongest to weakest: a CI step or hook that fails (hard); a recipe or
script the entrypoint names and CI runs; an entrypoint pointer to an
existing file; a document that states the rule with nothing wiring it
(unverified). Last-touched dates from history separate a maintained
mechanism from a fossil. A mechanism that only the reference's framework
honours is live for that framework and absent for any other.
