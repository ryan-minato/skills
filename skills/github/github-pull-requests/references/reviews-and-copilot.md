# Review threads, reviews, and Copilot code review

Use the column (MCP or gh) chosen in "Choose your path"; `O/R` and `N`
come from "Identify the repository". Every write operation in this file
publishes text — the Pre-publish gate in SKILL.md applies before each one.

## Read review summaries

- MCP: `pull_request_read` with method `get_reviews` (owner, repo,
  pullNumber).
- gh: `gh api repos/O/R/pulls/N/reviews`

Each review carries an author (`user.login`), a `state` (`APPROVED`,
`CHANGES_REQUESTED`, `COMMENTED`, `PENDING`), and a summary `body`.

## Read inline review comments

- MCP: `pull_request_read` with method `get_review_comments`.
- gh: `gh api repos/O/R/pulls/N/comments`

These are code-anchored comments (each has `path`, a line position, and a
diff hunk). Plain conversation comments are a different resource and live
under `repos/O/R/issues/N/comments` — do not look for them here.

## Identify Copilot's review

Copilot's reviews and inline comments are the ones whose author login is
`copilot-pull-request-reviewer[bot]` (user type `Bot`). To filter with gh,
append this to either `gh api` call above:

    --jq '.[] | select(.user.login=="copilot-pull-request-reviewer[bot]")'

## Reply in a review thread

Replies are publishing — run the Pre-publish gate first.

- MCP: `add_reply_to_pull_request_comment` (owner, repo, pullNumber, the
  `comment_id` of the inline comment being replied to, and `body`).
- gh: `gh api -X POST repos/O/R/pulls/N/comments/COMMENT_ID/replies -F body=@REPLY.md`
  (the `@` makes gh read the body from the file).

`COMMENT_ID` is the `id` field of the inline comment, taken from "Read
inline review comments" above.

## Submit your own review

Review bodies and inline comments are publishing — run the Pre-publish
gate first.

- MCP: `pull_request_review_write` method=`create` (event `COMMENT`,
  `APPROVE`, or `REQUEST_CHANGES`; `body`); add inline comments with
  `add_comment_to_pending_review` (path, line, side); then
  `pull_request_review_write` method=`submit`.
- gh: `gh pr review N -R O/R --comment --body-file REVIEW.md` — replace
  `--comment` with `--approve` or `--request-changes` when that is the
  verdict the user asked for.

## Request a Copilot code review

- MCP: `request_copilot_review` (owner, repo, pullNumber).
- gh (requires gh ≥ 2.88): `gh pr edit N -R O/R --add-reviewer @copilot`

Copilot code review must be enabled for the repository or organization.
If GitHub rejects the reviewer, tell the user instead of retrying — no
invocation variant will succeed until the feature is enabled.
