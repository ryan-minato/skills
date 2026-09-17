# sdd — Catalog Context

Rules, notes, and references that apply only to skills in this catalog.
(Repo-wide standards live in `.agents/knowledge/skill-quality.md`.)

Durable, per-project skills for spec-driven development, in two classes:

- **The methodology skill** (`spec-driven-development`): the practice,
  the loop, the approval package, how specifications meet tracked work.
  It is platform- and framework-agnostic: no tool command, file layout,
  workflow, or platform object appears in it beyond naming the approach
  families and the framework skills that own them.
- **Framework skills** (`openspec-workflow`, `spec-kit-workflow`, one per
  framework): that tool's records and layout, its approval package, what
  completes a change before the request is ready, the comment commands
  and labels a project installs, and the automation per platform.
  Tool-specific content is expected here; platform-specific content is a
  branch inside the skill (`references/github.md`, `references/gitlab.md`,
  `assets/github/`, `assets/gitlab/`), never a separate skill.

Skills are installed one at a time, stay installed for the life of the
project, and carry no disposable marker.

## Requirements

- A framework skill verifies every tool command from the tool's own
  `--help` and current documentation before running or documenting it and
  quotes none in prose; a bundled script may invoke the tool and pins the
  flags it relies on with the verification date.
- A framework skill restates the two loop facts it needs (the approval
  package precedes the task list; archiving or completion precedes ready)
  instead of pointing at the methodology skill's files.
- Automation assets are fork-safe by construction: scripts run from the
  base ref, a request's head is fetched as data, and nothing from the head
  is checked out, installed, or executed except on a same-repository
  branch. Every workflow declares its permissions explicitly and pins
  actions by commit.
- The design (or plan) a framework skill puts in the approval package
  bounds the approach — constraints, preferences, rejected alternatives —
  and never lists implementation steps; it is committed, so it carries no
  secret or private data.
- Adding a framework is one new skill here plus a row in this catalog's
  README pair. The methodology skill and the `meta` builder carry no
  framework-specific behavior, but each names the framework skills it can
  hand off to, so adding one also updates those lists — the methodology's
  approach families and handoff, and the builder's questioning round and
  adoption step.

## Dependencies

- Default range only: skills here may depend on `core` skills. No grant
  between the catalog's own skills and no grant to another catalog — they
  are installed one at a time, so co-presence is never guaranteed. The
  methodology skill hands the project's framework to its framework skill,
  a framework skill hands practice questions to the methodology skill, and
  both hand project-rule setting to the `meta` catalog's
  `meta-spec-workflow`: each is an optional handoff named by role, routed
  through `ryan-minato-skills-installing`, with the fallback stated for
  when the user declines.
- No dependency on or recommendation of skills from other repositories; no
  exemptions.

## Naming

`spec-driven-development` is the practice's proper noun and stands whole.
Framework skills are `<framework>-workflow`, the framework spelled as its
project spells it in lowercase (`openspec-workflow`, `spec-kit-workflow`).

## Scope

This catalog owns the practice and its per-framework operation during a
project's life. Settling and depositing a project's rules — level, tool,
request shape, approval mode and package, archive executor, scope — and
filling the platform base's generic slots belongs to the `meta` catalog's
`meta-spec-workflow`; building a GitHub or GitLab project's complete
lifecycle harness to `meta-github-workflow` / `meta-gitlab-workflow`;
defining what something should achieve before any spec to the
`engineering` catalog's `goal-alignment`; a machine-learning research
task's spec to the `machine-learning` catalog's `research-workflow`.

## Disambiguation

Whether and at which level to adopt, what a good specification is, the
loop, where the design sits, how issues and requests link specs →
`spec-driven-development` · how a change of the project's tool is
recorded, approved, archived or completed, and shown on a request, and
how to install that tool's check, comment commands, and labels → the
framework skill for the tool · settling or changing the project's rules
and the generic template, form, and project-skill slots → the `meta`
catalog's `meta-spec-workflow` · a tool's own command that creates,
applies, or archives one change → the tool's own agent commands, not
this catalog.

## References

- OpenSpec: <https://github.com/Fission-AI/OpenSpec>
- Spec-Kit: <https://github.com/github/spec-kit>
