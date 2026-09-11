## Why

The OpenSpec pilot (#80, #81) settled rules for this repository that the
public skills still lack or contradict — the approval record, the change
request body, the specification scope, failing baseline scenarios, the
in-request archive freeze, and the push path for archive automation — and
it exposed a responsibility fault line: `spec-driven-development` both
teaches the practice and sets project rules (near-verbatim duplicates of
`meta-spec-workflow`'s questions) while shipping a script two `meta`
builders depend on across the catalog boundary; the two platform builders
carry a whole specification-expression reference, archive assets, and
`{{SPEC_*}}` placeholders, so a project on any other paradigm still
receives spec-driven scaffolding; and `meta-spec-workflow` hands template
lines and checks to the platform builders instead of shaping them itself.
Now, before the next project is built from these builders.

## What Changes

- `meta-github-workflow`, `meta-gitlab-workflow` — **BREAKING**: become
  paradigm-neutral bases. Every specification-specific reference, asset,
  placeholder, form field, and passage is removed; the templates, forms,
  and project skill keep named structural extension slots (headings, step
  positions, field ids) listed in `references/durable-harness.md`, which a
  paradigm builder fills afterwards; the closing step hands off to the
  builder whose description claims the contract the entrypoint points to.
  The GitHub baseline gains "a required check is named in the ruleset only
  after it has run on the default branch"; the GitLab baseline gains "all
  threads resolved". Descriptions unchanged.
- `meta-spec-workflow` — **BREAKING**: gains a second phase. Phase one
  settles the contract as before, plus the approval mode (discussion-closed
  by default: the draft opens at the record, the agent stops, the gate
  owner discusses on the request and closes the discussion in
  conversation, the agent reconciles threads and record before
  implementing; blocking with a fixed comment as the alternative) and the
  specification scope (domains cover the product; the project's own
  harness, tooling, checks, workflows, and documents are spec-less
  changes). Phase two, after the platform base is delivered, fills the
  base's extension slots from per-platform references and assets: the
  request's specification block and checklist items, the intake field, the
  project skill's take-work, draft, reconcile, and finish steps, the
  archive workflow or job, the knowledge section, sync rows, and the
  maintainer action for the push path by owner type. It ships
  `scripts/archive_completed_changes.py`. Its description claims the second
  phase.
- `spec-driven-development` — **BREAKING**: keeps the methodology only —
  levels, when it pays, the approach families, the loop under the
  project's contract (publish the draft, stop, reconcile when the
  discussion closes, verify, converge or archive per the contract),
  specification quality, review scope, adopting existing code, the
  defaults it applies when no contract exists, and the rule that failing
  baseline scenarios for untouched behavior are filed as defects. Project
  rule-setting, the template lines, and the archive script leave it; it
  names the spec workflow builder of the `meta` catalog as the way to
  initialize or improve a project's rules.
- `meta-agent-authority`: the specification gate becomes a generic
  contract-defined gate that precedes review admission; under a
  discussion-closed contract the gate passes when the gate owner closes
  the discussion and the agent's reconciliation finds nothing open; the
  acceptance-evidence report points at the request's specification block
  and validation section.
- `meta-workflow-design`: the hand-off order runs the paradigm builder's
  second phase after the platform base; the deposited workflow file points
  at other contracts generically.
- Catalog files: `skills/meta/CONTEXT.md` names paradigm builders as a
  third kind beside contract and platform builders;
  `skills/engineering/CONTEXT.md`'s disambiguation and both catalogs'
  README pairs follow; `meta-harness-building`'s plan template lets a
  phased builder occupy one Build List row per phase.

## Skills touched

- `engineering/spec-driven-development` (modified): review and approval
  modes, publication then stop, archive per contract, reconciliation,
  request body, specification scope, baseline defects, contract-or-default
  reading, the handoff's fallback; the script requirement is removed.
- `meta/meta-spec-workflow` (modified): description, the questioning
  round, the deposited contract, tool references, the second phase, the
  four behaviors moved from the platform builders, reconciliation, the
  archive script.
- `meta/meta-github-workflow` (modified): four specification requirements
  removed; extension slots and the check-ordering baseline added.
- `meta/meta-gitlab-workflow` (modified): four specification requirements
  removed; extension slots and the thread-resolution baseline added.
- `meta/meta-agent-authority` (modified): the contract-defined gate; the
  report's pointers.
- `meta/meta-workflow-design` (modified): the hand-off order and the
  generic contract pointer.

## Installed behavior

- Platform builders: a project on any paradigm receives a base with no
  specification content and knows which builder shapes it next → `feat!`.
- `meta-spec-workflow`: configures a project's spec-driven rules end to
  end, including the platform artifacts, and defaults to a discussion gate
  → `feat!`.
- `spec-driven-development`: an agent following it applies the project's
  contract or the documented defaults and never sets project rules or
  edits harness files → `refactor!` for the boundary; the approval rule
  (no longer "a comment naming the commit") is a `fix`.
- `meta-agent-authority`, `meta-workflow-design`: corrected gate semantics
  and hand-off order → `fix`; the report pointers → `feat`.

## Impact

- README pairs: the `meta` rows for `meta-spec-workflow` and the two
  platform builders, the `engineering` row for `spec-driven-development`
  (both languages).
- Catalog `CONTEXT.md` of `meta` and `engineering`; the `meta` plugin
  description in `.claude-plugin/marketplace.json` (a human-owned field).
- The platform domains' `## Purpose` lines describe capabilities this
  change removes; they are corrected by hand in the same pull request
  under the narrow exception the companion repository change records.
- The mirror of `archive_completed_changes.py` in this repository, its
  origin path in `scripts/validate_harness.py`, and the register row —
  the companion repository change `sdd-pilot-lessons-harness`.
- Symlinks and `marketplace.json` skill lists are unaffected: no skill
  directory is added, renamed, or removed.

## Non-goals

- A draft-aware pull request body policy check shipped as a builder asset
  (a later change; phase two states the ready-state rules as a recipe).
- Granting this repository's archive workflow a push path, and the three
  harness sentences that still say "the Actions identity on the bypass
  list" (the `automated-archive` repository change).
- `Trigger: description` blocks for the domains that lack one
  (`meta-github-workflow`, `meta-gitlab-workflow`, `meta-agent-authority`,
  `meta-workflow-design`): a separate issue; this change modifies the
  platform builders without touching their descriptions.
- Changing `spec-driven-development`'s description.

## Tracked work

No issue: user-directed change planned in conversation as the follow-up
to the pilot (#76).
