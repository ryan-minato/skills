# Planning, Labels, and Goals

Read when the harness uses labels, milestones, tracking issues, sub-issue
hierarchy, issue types, or triage states — that is, on every build that
plans work without GitHub Projects.

## The default planning system

The native day-to-day view is issue filters and the pull-request list, not
a board. The default cadence object is the **milestone per release**;
long-horizon goals are **tracking issues** (vision, non-goals, acceptance,
sub-issue list) with milestones as their date buckets — the consensus gate
in [decision-tree.md](decision-tree.md) produces them. Hierarchy is
**sub-issues** (up to 100 children, 8 levels); tasklist blocks are retired
and must never be emitted, though plain GFM task lists still render as
text. A Scrum request without a Projects opt-in has no sprint object —
surface that consequence and let the user choose knowingly.

## Labels: extend the defaults, never rebuild them

GitHub ships `bug`, `documentation`, `duplicate`, `enhancement`,
`good first issue`, `help wanted`, `invalid`, `question`, `wontfix` in
every repository, and the platform keys on them: `good first issue` feeds
the repository's `/contribute` page, and `bug`/`enhancement` are the
community's shared vocabulary and the natural `release.yml` categories.
Keep them. Add only what the harness needs:

- `priority/high|medium|low` — or `P0`–`P5` if the user prefers numeric;
  offer both, recommend semantic.
- `status/needs-triage` and `status/blocked`.
- `area/<real-boundary>` derived from the actual directory or ownership
  structure — never invented areas; each area label pairs with a CODEOWNERS
  pattern and a labeler path entry.

Labels have no scopes and no mutual exclusivity anywhere on the platform.
In an organization, put the type axis on native **issue types** (defaults:
Task, Bug, Feature — exactly one per issue, real single-select semantics)
and consider **org issue fields** for priority; then labels carry only
priority/status/area/community. On a personal account, `bug` and
`enhancement` double as the type axis plus an added `task` label, and
one-of-N is a written convention with a named owner — optionally checked
after the fact by the triage workflow, never enforceable in advance.

Every label carries a description and a 6-digit hex color without `#`.
Labels are per-repository with no organization inheritance: commit the
taxonomy as `labels.json` in the target and ship
`scripts/sync_labels.py` beside it so sibling repos
and drift both have a mechanical answer. The deposited taxonomy check keeps
`labels.json`, `release.yml`, the issue forms, and the labeler config
agreeing.

## Triage lifecycle

`status/needs-triage` is applied on arrival (by the form's `labels:` key or
the triage workflow) and removed when type and priority are both present.
Record who grooms, how often, and what each label state obligates. An
issue that is really a question becomes a discussion — conversion is part
of triage, not a rejection.

Done when: every label in the taxonomy has a meaning, an applier (form,
workflow, or human), and a consumer (filter, release notes, labeler, or
policy); the tracking-issue convention is written into project knowledge;
and no planning rule depends on a board that does not exist.
