## Why

The comment-command workflow that `openspec-workflow` and `spec-kit-workflow` install on GitHub can fail to answer a comment it admitted, answer a real command with the help text, or name and link a different head than the one its content came from. Its job admits any comment that starts with `/spec`, but the reply step reads only a line matching `/spec` plus a space or the end, keeps a trailing carriage return on the command word, and takes the head from a `gh pr view` call made before the snapshot reads the request again. Issues #96 and #100 report these; both sit in the same two steps of the same asset.

## What Changes

- `openspec-workflow`, the produced comment-command workflow (`assets/github/workflow-spec-command.yml`):
  - **Admission.** The job runs only for a comment that begins with `/spec` followed by the end of the comment, a space, a tab, or a line break, so `/specs`, `/specification`, and `/speckit.plan` no longer start a privileged run. Collaborator-only admission and the bot exclusion stay.
  - **Every admitted command is answered.** A carriage return at the end of the command line is ignored, a tab separates words as a space does, and a `/spec` line that names no known command, or that the step cannot read, is answered with the command list. The run passes in those cases, as a bare `/spec` does today.
  - **The echoed command is the parsed one.** The reply's header shows the command as the step read it, `/spec help` for anything answered with the command list, never the raw line.
  - **One head.** The reply's head line and every link it builds (the `[view]` links and the truncation link of `show`) name the head commit the snapshot recorded, never a head read by a separate call. The pull request's state still comes from `gh pr view`. A snapshot that carries no head, or a state read that returns none, fails the step that reads it, before any reply, as a failed snapshot does today.
  - The permission comment "reading the request and its files is a pull-request read" is reworded so that it stays true when the snapshot stops listing the request's files (#101). It is a comment; nothing it grants changes.
  - `references/github.md`: `## Fork safety` says which comments start the job, that every admitted command gets a reply, and that the reply names the head the snapshot read; `## Verification after installing` adds an unknown command answered with the command list.
- `spec-kit-workflow`: the same, in its `assets/github/workflow-spec-command.yml` and `references/github.md`, with its feature vocabulary.

## Skills touched

- `sdd/openspec-workflow` (modified): a new Behavior requirement for admission and answering, and one for the head the reply names.
- `sdd/spec-kit-workflow` (modified): the same two requirements.

## Installed behavior

An agent that installs the automation on GitHub now produces a comment workflow that answers every command it admits, starts no run for words that only begin with `/spec`, and names and links the head its content was read from. Before, a `/specs` or CRLF `/spec` comment turned the run red with no reply, `/spec status` ending in a carriage return got the help text, and a push between two API reads could make the reply name and link another commit → `fix`. Projects that installed the automation earlier keep their copy until they reinstall.

## Impact

- No description, reference load trigger, asset list, script, handoff, symlink, `marketplace.json` entry, or catalog README row changes.
- The GitLab fragment (`assets/gitlab/ci-spec-jobs.yml`) is untouched: it parses no comments and reads the head with git at the pipeline's commit.
- This repository's own `.github/workflows/spec-command.yml` mirrors the asset (register row in `.agents/knowledge/harness-maintenance.md`), and `.agents/knowledge/github-checks.md` describes its admission. Both move in the companion repository change `spec-command-reply-harness` on the same branch.

## Non-goals

- The management scripts (`assets/spec_changes.py`, `assets/spec_kit_features.py`, the bundled `scripts/`, and this repository's `scripts/spec_changes.py`) and the snapshot schema. The snapshot already records the head it read.
- Carrying the pull request's state in the snapshot to cut the workflow to one API read.
- The installation requirement and its scenarios, including "Command answered when the script fails" and "Command invoked by the request's author", which stay as they are.
- What `/spec show` and `/spec status` print, and the command names.
- The snapshot's file-list cap and large requests (#101).

## Tracked work

Issues #96 and #100.
