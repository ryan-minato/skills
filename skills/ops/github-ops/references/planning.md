# Planning: milestones, labels, Projects

Loaded when the task manages milestone lifecycle, label lifecycle, or
Projects v2 boards. Putting one issue or PR *into* a milestone or label is
an issue/PR edit — [issues.md](issues.md) / [pull-requests.md](pull-requests.md)
own it.

Surface exposure: milestone titles and descriptions, project names, and
label names are visible to everyone who can see the repository. The
pre-publish gate applies to creating or editing titles, descriptions, and
names; closing, deleting, reordering, item moves, and reads carry no new
text and skip it.

Projects v2 hang off an **owner** (a user or an organization), not a
repository. `OWNER` below is that account (`@me` for your own), and a
project is addressed as `OWNER` + project `NUMBER` from `gh project list`.

## Token scope check (before any Projects v2 operation)

`gh project` needs the `project` token scope. Run `gh auth status` and
look for `project` in the token scopes; if it is missing, run
`gh auth refresh -s project` and complete the browser prompt.
Done when: `gh auth status` lists the `project` scope. Milestone and label
operations need no extra scope — skip this section for them.

## Milestones

Milestones are addressed by their `number` (from the list row). Due dates
are ISO timestamps (`2026-08-01T00:00:00Z`).

| Task | MCP capability | gh command |
|---|---|---|
| List | — | `gh api "repos/O/R/milestones?state=all" -q '.[] | {number, title, state, due_on}'` |
| Create | — | `gh api -X POST repos/O/R/milestones -f title="T" [-f description="D"] [-f due_on="YYYY-MM-DDT00:00:00Z"]` |
| Edit | — | `gh api -X PATCH repos/O/R/milestones/NUMBER -f title="T"` (same fields as create) |
| Close / reopen | — | `gh api -X PATCH repos/O/R/milestones/NUMBER -f state=closed` (or `open`) |
| Delete | — | `gh api -X DELETE repos/O/R/milestones/NUMBER` — see the gotcha; confirm with the user first |

Read [milestone-recipes.md](milestone-recipes.md) when the milestone
operation goes beyond create/edit/close — resolving a milestone by title,
listing a milestone's issues, or bulk-moving issues between milestones.

## Labels

| Task | MCP capability | gh command |
|---|---|---|
| List | list repository labels | `gh label list -R O/R --json name,color,description` |
| Create | create a label | `gh label create NAME -R O/R --color HEX --description "D"` |
| Rename / recolor / redescribe | update a label | `gh label edit NAME -R O/R [--name NEW] [--color HEX] [--description "D"]` |
| Delete | — | `gh label delete NAME -R O/R --yes` — labels are removed from every issue and PR carrying them; confirm with the user first |
| Copy a repo's labels into O/R | — | `gh label clone SOURCE_OWNER/SOURCE_REPO -R O/R` |

## Projects v2

All rows are gh-only (`—` throughout for MCP). `NUMBER` comes from
`gh project list --owner OWNER`.

| Task | gh command |
|---|---|
| List projects | `gh project list --owner OWNER` |
| View a project | `gh project view NUMBER --owner OWNER` |
| Create | `gh project create --owner OWNER --title "T"` |
| Edit title/description | `gh project edit NUMBER --owner OWNER --title "T"` |
| Close / delete | `gh project close NUMBER --owner OWNER` / `gh project delete NUMBER --owner OWNER` (delete: confirm with the user first) |
| Link to a repository | `gh project link NUMBER --owner OWNER --repo O/R` |
| Add an issue/PR as item | `gh project item-add NUMBER --owner OWNER --url URL` |
| List items | `gh project item-list NUMBER --owner OWNER --format json` |
| List fields (and Status options) | `gh project field-list NUMBER --owner OWNER --format json` |
| Create a field | `gh project field-create NUMBER --owner OWNER --name "N" --data-type SINGLE_SELECT --single-select-options "A,B"` (also `TEXT`, `NUMBER`, `DATE`) |
| Edit an item's field (move on the board) | resolve IDs with the script below, then `gh project item-edit --id ITEM_ID --project-id PROJECT_ID --field-id FIELD_ID --single-select-option-id OPTION_ID` (or `--text`/`--number`/`--date`/`--iteration-id`, or `--clear`) |
| Archive an item | `gh project item-archive NUMBER --owner OWNER --id ITEM_ID` |

`item-edit` takes GraphQL node IDs, not numbers. Resolve them in one call
with [scripts/project_fields.py](scripts/project_fields.py):

    python3 scripts/project_fields.py --owner OWNER --number N \
        --field "Status" --option "In Progress" --item-url ISSUE_URL

which prints `{project_id, field_id, option_id, item_id}` ready to paste
into `item-edit`. Setting the Status field is how an item moves between
board columns.
Done when: the edited item shows the new value in
`gh project item-list ... --format json`.

Read [projects-graphql.md](projects-graphql.md) when `gh project`
subcommands cannot express the needed Projects operation (draft-issue
conversion, filtered item queries, iteration ids).

## Gotchas

- gh has no `milestone` command group — the REST rows above are the only
  gh path; there is no MCP capability for milestones either. The REST
  fallback script has no milestone or Projects subcommands: on that tier,
  these reads stop like writes.
- Deleting a milestone silently detaches every issue and PR assigned to
  it; closing preserves history. Prefer close; delete only on explicit
  user confirmation.
- Renaming a label updates it everywhere it is applied; deleting removes
  it from every issue/PR. Rename beats delete-and-recreate.
- The `project` scope error surfaces as "your token has not been granted
  the required scopes" — run the token scope check section, then retry.
- `--owner` takes a user login, an org name, or `@me`; forgetting it makes
  gh guess from the current repository, which often picks the wrong owner.
- One `item-edit` invocation updates exactly one field; loop for several
  fields.
- Iteration fields are set by `--iteration-id`, not by name — the ids are
  in the field-list JSON (`configuration.iterations`).
