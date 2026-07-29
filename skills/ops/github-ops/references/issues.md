# Issues

Loaded when the task operates on issues. Use the column chosen in "Choose
your path"; `O/R` comes from "Identify the repository". The pre-publish
gate applies to create, comment, and any edit that changes public text or
metadata (title, body, labels, assignees, milestone); pure reads and lists
skip it.

## Operations

Send multi-line bodies through a file, never an inline shell string: write
the body to a file first, then pass it with `--body-file FILE` (gh) or
fill the MCP body parameter from the file.

| Task | MCP capability | gh command |
|---|---|---|
| Create issue | create an issue (title, body, labels, assignees) | `gh issue create -R O/R --title "TITLE" --body-file BODY.md [--label L] [--assignee U] [--milestone "M"]` |
| Comment | comment on an issue | `gh issue comment N -R O/R --body-file COMMENT.md` |
| Close | update an issue to closed, with a state reason | `gh issue close N -R O/R --reason completed` (or `--reason "not planned"`) |
| Reopen | update an issue to open | `gh issue reopen N -R O/R` |
| Read issue + comments | read an issue, then its comments | `gh issue view N -R O/R --comments` |
| Read labels on an issue | read an issue's labels | `gh issue view N -R O/R --json labels` |
| List issues | list issues (state open/closed) | `gh issue list -R O/R --state open --json number,title,labels,updatedAt` |
| Search issues | search issues (query `repo:O/R is:open TEXT`) | `gh issue list -R O/R --search "TEXT label:bug"` |
| List available labels | list repository labels | `gh label list -R O/R --json name,description,color` |
| Edit labels/assignees/title | update an issue's metadata | `gh issue edit N -R O/R --add-label L --remove-label M --add-assignee U --title "T"` |
| Set / clear milestone | update an issue's milestone | `gh issue edit N -R O/R --milestone "TITLE"` / `--remove-milestone` |

After any create or comment, the response contains the new item's URL;
always report that URL to the user.
Done when: the URL is reported.

On the REST fallback tier, the read rows map to `rest_read.py issues`,
`issue --comments`, `labels`, and `search` — the invocation table is in
[rest-fallback.md](rest-fallback.md), already loaded on that path.

Read [issue-recipes.md](issue-recipes.md) when the needed issue operation
is not in the table above (pin/unpin, lock/unlock, transfer, sub-issues,
advanced search filters).

## Gotchas

- The close reason is `completed` or `not planned`: gh takes it quoted,
  with the space (`--reason "not planned"`); the MCP update takes its
  state-reason value in underscore form (`not_planned`).
- `gh issue create` ignores issue templates in non-interactive mode — the
  template must be applied by drafting the body against it (see
  [use-issue-forms.md](use-issue-forms.md), pointed at from the
  conventions table).
