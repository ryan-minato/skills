# Organization Types and Fields

Read when the repository is organization-owned and the harness needs to
audit or initialize org-level issue types and issue fields.

## The capability gate

Three conditions, each evidenced before promising anything:

- **Owner is an organization.** Personal accounts have neither issue types
  nor issue fields; that branch belongs to
  [planning-and-goals.md](planning-and-goals.md).
- **Host version.** Issue types reached GitHub Enterprise Server in 3.18;
  **issue fields only ship in GHES 3.23**. On an older GHES the priority
  axis falls back to labels while the type axis may still be native — the
  two features are gated separately, so probe both.
- **Write authority.** Reading types and fields needs `read:org`; creating
  or changing them needs organization ownership and `admin:org`. **A
  repository admin is not an organization admin** — this is the usual
  failure, and it surfaces as a 403 on the first write, after the design is
  already agreed. Confirm authority during investigation, not at apply time.

## Audit before creating

Every organization already owns a working taxonomy: the issue types
**Task**, **Bug**, and **Feature**, and the issue fields **Priority**
(Urgent/High/Medium/Low), **Effort** (High/Medium/Low), **Start date**, and
**Target date**. They are created automatically and are fully editable.

Treat them exactly as the default labels are treated — extend them, never
rebuild them. Read the live state first:

```bash
gh api orgs/{org}/issue-types
gh api orgs/{org}/issue-fields
```

An organization that has already renamed `Priority` or added its own types
has made decisions the harness must preserve, not overwrite.

## Assigning the axes

- **Type axis → issue types.** Exactly one per issue, enforced by the
  platform. This is real mutual exclusivity, which labels never provide.
- **Priority axis → the `Priority` issue field.** Do not create a parallel
  `priority/*` label set beside it; two sources of truth for one axis is
  the failure this branch exists to prevent.
- **Labels keep** area, status, and community meaning — the axes GitHub has
  no field for, plus the defaults the platform itself keys on.

A field's value lives on the issue and stays consistent across every
project the issue appears in, which is what makes it a taxonomy rather than
a per-board annotation.

## Visibility, pinning, and limits

Fields are either **Public** or organization-only — `all` and
`organization_members_only` in the API, which is the spelling the request
body needs. Only Public fields appear in public and internal projects; an
organization-only field renders as an empty cell there. Decide visibility per field against repository visibility,
and record the choice — it is invisible from the repository side.

Pinning a field to an issue type makes it appear automatically in that
type's sidebar and creation modal. Pin the priority field to every type the
triage lifecycle covers; leave estimate-style fields unpinned unless the
team actually grooms them.

Limits: 25 issue types and 25 issue fields per organization, 100 options per
single-select, 10 pinned fields per type. Issue fields also surface in
projects and **count toward the 50-field project limit**.

## Initializing

1. Read the live types and fields, and record which defaults are untouched.
2. Draft the target state as an `org-taxonomy.json` reworked from
   `assets/org-taxonomy.json`.
3. Dry-run `scripts/sync_org_taxonomy.py` and review the exact plan.
4. **State the blast radius before asking for approval**: organization
   settings apply to every repository in the organization, so this write
   reaches far beyond the repository being harnessed. Organization-level
   approval is a separate decision from repository-level approval.
5. Apply only with explicit authorization, then read both endpoints back.
6. Register the result in `platform-settings.md` with its readback commands.

## Setting values on issues

- Type: `gh issue create --type <name>` and `gh issue edit --type <name>`,
  which need **gh 2.94.0 or newer**; below that, use `gh api`.
- Field values have **no `gh` subcommand**. They go through
  `gh api repos/{owner}/{repo}/issues/{number}/issue-field-values`, and
  setting them needs push access to the repository, not `admin:org`.
- **Issue forms cannot pre-fill field values.** A form's `type:` and
  `labels:` keys apply on submission, but field values cannot be set by a
  form, a template, or a URL parameter. Every field value therefore arrives
  from triage automation or a human — design the triage lifecycle knowing
  the field starts empty.

## Destructive edges

- Deleting an issue type or issue field removes it from **every issue in the
  organization**, and the values do not come back. Disable a type
  (`is_enabled: false`) instead of deleting it; a disabled type stays
  visible on issues that already carry it and cannot be newly assigned.
- Updating a single-select field's `options` **replaces the whole set**.
  Carry each surviving option's existing `id` in the request or those
  options are destroyed and recreated, orphaning their values.

Done when: the live types and fields are read back and recorded, each axis
has exactly one home, every field's visibility and pinning is a stated
decision, and no proposed change to organization settings lacks its own
approval.
