# Pull requests

Loaded when the task operates on pull requests. Use the column chosen in
"Choose your path"; `O/R` comes from "Identify the repository".

Surface exposure: creating a PR publishes every commit message and the
complete diff of `BASE...HEAD`, not just the description. For PR creation,
comments, and review text, the pre-publish gate runs file-based:

1. Write the exact outgoing content to a scratch directory: title, body,
   each comment or review body, `git log BASE..HEAD --format=full >
   commits.txt`, `git diff BASE...HEAD > diff.patch`, and any attachments.
2. Run the review procedure in [publish-review.md](publish-review.md) over
   that directory. Read that file every time — do not review from memory.
3. Publish only after the verdict is exactly `SAFE TO PUBLISH: YES`. On
   `NO`, fix every finding, rebuild the files from the fixed content, and
   review again.

Pure reads, draft/ready flips, and merges of already-reviewed PRs carry no
new content and skip the gate.

## Operations

Send multi-line bodies through a file, never an inline shell string
(`--body-file FILE` for gh; fill the MCP body parameter from the file).

| Task | MCP capability | gh command |
|---|---|---|
| Create PR | create a pull request (base, head, title, body; optional draft, reviewers) | `gh pr create -R O/R --base BASE --head BRANCH --title "TITLE" --body-file BODY.md [--draft]` |
| Comment | comment on an issue or PR (PRs share issue numbering) | `gh pr comment N -R O/R --body-file COMMENT.md` |
| Read PR | read PR details | `gh pr view N -R O/R --json number,title,state,isDraft,mergeable,reviewDecision,url` |
| Read diff | read a PR's diff | `gh pr diff N -R O/R` |
| Check results | read a PR's status rollup or check runs | `gh pr checks N -R O/R` |
| List PRs | list pull requests | `gh pr list -R O/R --state open --json number,title,headRefName,updatedAt` |
| Search PRs | search pull requests (query `repo:O/R TEXT`) | `gh pr list -R O/R --search "TEXT"` |
| Close / reopen | update a PR's state | `gh pr close N -R O/R` / `gh pr reopen N -R O/R` |
| Draft → ready | update a PR's draft flag | `gh pr ready N -R O/R` |
| Edit labels / milestone | update a PR's metadata | `gh pr edit N -R O/R --add-label L --remove-label M --milestone "TITLE"` |
| Merge | merge a PR with an explicit method | `gh pr merge N -R O/R --squash` |

The default merge method is squash, because linear history is the safer
default — deviate only when the project's convention says otherwise.
After create and comment operations, report the returned URL to the user.
Done when: the URL is reported.

On the REST fallback tier, the read rows map to `rest_read.py prs` and
`pr` (with `--diff`, `--files`, `--reviews`, `--comments`, `--checks`) —
see [rest-fallback.md](rest-fallback.md), already loaded on that path.

- Read [ci-runs.md](ci-runs.md) when a check is failing and you need the
  logs.
- Read [reviews-and-copilot.md](reviews-and-copilot.md) when reading or
  replying to review threads, submitting a review, or requesting a
  Copilot code review.
- Read [pr-recipes.md](pr-recipes.md) when the needed PR operation is not
  in the table above (update branch, reviewers, linked issues, revert,
  checkout).

## Gotchas

- `gh pr checks` exits with code 8 while checks are still pending — that is
  not an error; wait and re-run, or use `--watch`.
- A PR created from a fork needs `--head FORK_OWNER:BRANCH` (MCP: the same
  `owner:branch` form in the head parameter).
- Reading reviews and threads is research; replying to them is publishing
  and passes the gate.
- Never dump a full CI log into context — follow the failed-only + tail
  rule in [ci-runs.md](ci-runs.md).
