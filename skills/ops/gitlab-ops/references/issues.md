# Issues

Loaded when the task operates on issues. Use the column chosen in "Choose
your path"; `G/P` comes from "Identify the host and project". The
pre-publish gate applies to create, comment, and any edit that changes
public text or metadata (title, description, labels, assignees,
milestone); pure reads and lists skip it.

## Operations

| Task | glab command | MCP capability (min GitLab) |
|---|---|---|
| Create issue | `glab issue create -R G/P -t "TITLE" -d "$(cat BODY.md)" [-l LABEL] [-a USER] [-m "MILESTONE"] -y` | create an issue (18.4) |
| Comment | `glab issue note N -R G/P -m "$(cat COMMENT.md)"` | comment on an issue/work item (18.7) |
| Close | `glab issue close N -R G/P` | — |
| Reopen | `glab issue reopen N -R G/P` | — |
| Read issue + comments | `glab issue view N -R G/P --comments` | read an issue (18.4), then its comments (18.7) |
| List / filter issues | `glab issue list -R G/P [--closed] [-l bug] [--search "TEXT"]` | — |
| List available labels | `glab label list -R G/P -F json` | search labels (18.9) |
| Edit labels/assignees/title | `glab issue update N -R G/P [-l NEW] [-u OLD] [-a +USER] [-t "T"]` | — |
| Set / clear milestone | `glab issue update N -R G/P -m "TITLE"` (`-m ""` clears) | — |

After any create or comment, the response contains the new item's URL;
always report that URL to the user.
Done when: the URL is reported.

On the REST fallback tier, the read rows map to `rest_read.py issues`,
`issue --comments`, and `labels` — the invocation table is in
[rest-fallback.md](rest-fallback.md), already loaded on that path (note:
notes and labels need a token on gitlab.com).

Read [issue-recipes.md](issue-recipes.md) when the needed issue operation
is not in the table above (confidential issues, due date and weight,
iteration or epic assignment, lock discussion, linked issues, move to
another project, advanced search filters).

## Gotchas

- GitLab has no close *reason*; closing is unqualified. State filter
  values elsewhere are `opened`/`closed`, but `glab issue list` uses the
  `--closed`/`--all` flags instead.
- On `glab issue update`, `-a USER` **replaces** all assignees; prefix
  with `+` to add (`-a +USER`) or `!`/`-` to remove.
- Unknown label names never error, but the two mechanisms disagree:
  `-l NAME` via glab/API silently **creates** a new project label, while
  a `/label ~NAME` quick action silently **ignores** it. The conventions
  section exists so neither surprises you.
