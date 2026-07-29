# Planning: milestones, labels, iterations, boards, epics

Loaded when the task manages planning structures themselves. The GitLab
Duo MCP server has no tools for these structures — glab (with `glab api`
for gaps) is the only path, so the tables have one column. Putting one
issue *into* a milestone/iteration/epic — or moving it between
label-backed board lists, which is relabeling — belongs to
[issues.md](issues.md); an MR's milestone to
[merge-requests.md](merge-requests.md); a release's milestone to
[releases.md](releases.md).

Surface exposure: milestone, board, list, epic, and label names and
descriptions are visible to everyone who can see the project. The
pre-publish gate applies to milestone/epic/label create and edit and to
board or list create and rename; close, reopen, delete, reorder, and
reads carry no new text.

Each section header carries its tier floor. GitLab returns **404, not
403, for features above the instance's tier or license** — on a 404 for a
Premium row, report the tier requirement instead of retrying. Probe the
instance version with `glab api version` when a feature might be too new
for a self-managed host.

## Milestones (all tiers)

glab addresses milestones by numeric id — find it with the list row
first. `--group GROUP` switches any row to group level (`--project` and
`--group` are mutually exclusive).

| Task | Command |
|---|---|
| List | `glab milestone list -R G/P [--state active\|closed] -F json` |
| Find id by title | `glab milestone list -R G/P --title "TITLE" -F json` |
| View one | `glab milestone get ID -R G/P` |
| Create | `glab milestone create -R G/P --title "TITLE" [--description "$(cat DESC.md)"] [--start-date 2026-07-01] [--due-date 2026-09-30]` |
| Edit | `glab milestone edit ID -R G/P [--title "T"] [--due-date ...]` |
| Close / reactivate | `glab milestone edit ID -R G/P --state close` (or `--state activate`) |
| Delete | `glab milestone delete ID -R G/P` — see the gotcha; confirm with the user first |

Edits are partial: only the fields you pass change; omitted fields keep
their values. Report the milestone's `web_url` from the JSON response.
Done when: the URL is reported.

## Labels (all tiers)

glab has `label list` and `label create` only; rename, recolor, and
delete go through the REST endpoints. `LABEL_ID` is the numeric `id` in
the list output. Group labels use `groups/GROUP/labels` endpoints and are
inherited by every project below the group — edit them at group level
only with the user's explicit confirmation.

| Task | Command |
|---|---|
| List (with ids) | `glab label list -R G/P -F json` |
| Create | `glab label create -R G/P --name "NAME" --color "#5843AD" --description "D"` |
| Rename / recolor / redescribe | `glab api --method PUT projects/:fullpath/labels/LABEL_ID [-f new_name="N"] [-f color="#HEX"] [-f description="D"]` |
| Delete | `glab api --method DELETE projects/:fullpath/labels/LABEL_ID` — removed from every issue and MR carrying it; confirm with the user first |

Renaming a label updates it everywhere it is applied — rename beats
delete-and-recreate. Scoped labels (`scope::value`) are plain labels with
`::` in the name; their mutual exclusion is enforced by GitLab on
assignment (enforcement is Premium, the naming works everywhere).

## Iterations (Premium; read-only)

List with `glab iteration list -R G/P -F json` (`-g GROUP` for the
group's; state filter via `glab api "groups/GROUP/iterations?state=current"`).
Iterations cannot be created or edited through REST or glab — cadences
manage them. Read
[iterations-and-cadences.md](iterations-and-cadences.md) when asked to
create, edit, or schedule an iteration or cadence.

## Issue boards (Free; scoping and non-label lists Premium)

Board and list management lives entirely in [boards.md](boards.md) — read
it for any board task: the Free operations table (create/rename/delete
boards, label lists, reordering), Premium scoping and non-label lists,
group boards, and epic boards.

## Epics (Premium/Ultimate; group-level)

Epic lifecycle lives entirely in
[epics-work-items.md](epics-work-items.md) — read it for any epic task:
the deprecated-but-stable REST operations table (list, view, create,
edit, close, labels, parent/child, delete) and the experimental
work-items successor path for instances where the legacy endpoints return
404 or the user asks for work items, tasks, objectives, or key results.

## Gotchas

- Deleting a milestone silently detaches every issue, MR, and epic that
  referenced it — closing (`edit --state close`) is almost always what
  the user actually wants; confirm before deleting.
- Group-inherited labels cannot be edited through the project endpoints —
  a 404 on `projects/.../labels/ID` for a label you can see usually means
  it lives at group level.
- 404 ≠ wrong path: for iterations, epics, and board scoping it usually
  means the tier gate (Free instance) — say so instead of retrying.
- `glab milestone` and `glab work-items` are recent command groups; if a
  subcommand is missing, update glab (`glab check-update`) instead of
  hunting for alternate spellings.
- The REST fallback script has no milestone, board, iteration, or epic
  subcommands: on that tier, these reads stop like writes.
