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

## Where the type and priority axes live

Labels have no scopes and no mutual exclusivity anywhere on the platform,
so a label-only taxonomy can never enforce "exactly one type" or "exactly
one priority". Owner type decides whether that matters:

**Organization-owned — the default.** Put the type axis on native **issue
types** and the priority axis on the organization's **`Priority` issue
field**. Both ship pre-populated (types Task/Bug/Feature; fields Priority,
Effort, Start date, Target date), both are real single-select, and neither
costs a label. Labels then carry only area, status, and community meaning.
Do **not** add `priority/*` labels beside the field — one axis, one home.
Read [org-configuration.md](org-configuration.md) to audit and initialize
them.

**Personal account — the fallback.** Neither feature exists. `bug` and
`enhancement` double as the type axis alongside an added `task` label, and
priority becomes `priority/high|medium|low` (or `P0`–`P5` if the user
prefers numeric; recommend semantic). One-of-N is then a written convention
with a named owner — optionally checked after the fact by the triage
workflow, never enforceable in advance. Say that consequence out loud
rather than implying the labels behave like fields.

The same fallback applies to an organization on GitHub Enterprise Server
older than 3.23, where issue types exist but issue fields do not: native
type axis, label priority axis.

## Labels: extend the defaults, never rebuild them

GitHub ships `bug`, `documentation`, `duplicate`, `enhancement`,
`good first issue`, `help wanted`, `invalid`, `question`, `wontfix` in
every repository, and the platform keys on them: `good first issue` feeds
the repository's `/contribute` page, and `bug`/`enhancement` are the
community's shared vocabulary and the natural `release.yml` categories.
Keep them. Add only what the harness needs:

- `status/needs-triage` and `status/blocked`.
- `area/<real-boundary>` derived from the actual directory or ownership
  structure — never invented areas; each area label pairs with a CODEOWNERS
  pattern and a labeler path entry.
- `goal` for tracking issues, when the tracking-issue convention is adopted.
- On the personal-account branch only, the type and priority labels above.

Every label carries a description and a 6-digit hex color without `#`.
Labels are per-repository with no organization inheritance: commit the
taxonomy as `labels.json` in the target and ship
`scripts/sync_labels.py` beside it so sibling repos
and drift both have a mechanical answer. The deposited taxonomy check keeps
`labels.json`, `release.yml`, the issue forms, and the labeler config
agreeing.

## Triage lifecycle

`status/needs-triage` is applied on arrival (by the form's `labels:` key or
the triage workflow) and removed when type and priority are both present —
read from the native type and `Priority` field where they exist, from the
labels where they do not. A form can set `type:` but **cannot pre-fill a
field value**, so on the organization branch priority always starts empty
and triage is what fills it; design the lifecycle around that rather than
assuming intake supplies both axes.

Record who grooms, how often, and what each label state obligates. An
issue that is really a question becomes a discussion — conversion is part
of triage, not a rejection.

Done when: each axis has exactly one home for this owner type; every label
in the taxonomy has a meaning, an applier (form, workflow, or human), and a
consumer (filter, release notes, labeler, or policy); the tracking-issue
convention is written into project knowledge; and no planning rule depends
on a board that does not exist.
