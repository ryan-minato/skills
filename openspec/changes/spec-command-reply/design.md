## Context

See proposal.md for motivation. The companion repository change `spec-command-reply-harness` carries this repository's own workflow and knowledge.

**Current shape:**
- **Three copies of one workflow.** `skills/sdd/openspec-workflow/assets/github/workflow-spec-command.yml`, `skills/sdd/spec-kit-workflow/assets/github/workflow-spec-command.yml`, and this repository's `.github/workflows/spec-command.yml` differ only in the header comment, the checkout pin placeholder, the script name, and the change or feature vocabulary (read with `diff` on 2026-09-30).
- **Admission.** The job's `if:` is `startsWith(github.event.comment.body, '/spec')`, plus the pull request, bot, and collaborator conditions. It admits `/specs`, `/specification`, and Spec-Kit's own `/speckit.*` commands pasted into a comment.
- **Parsing.** The reply step, which GitHub runs as `bash -e {0}` because it sets no `shell:`, takes the first line matching `^/spec( |$)` through `grep … || true`, splits it with `set -- $line`, and runs an unguarded `shift`.
  - A body with no matching line (`/specs`, `/spec` + CR LF) fails on that `shift` before any reply.
  - A tab after `/spec` fails the same way.
  - A command line ending in CR keeps the CR on the command word, so `/spec status` + CR is answered with the help text.
- **Head.** The resolving step writes `head` from `gh pr view --json headRefOid,state` before `snapshot` reads `repos/{repo}/pulls/{n}` again. The reply's header (`${HEAD:0:7}`) and the `--url-prefix …/blob/$HEAD` that builds every `[view]` and truncation link use the first read; the content comes from the second.
- **The snapshot already records its head.** `head.sha` is a required, type-checked key of `spec-snapshot/1` in both management assets (`SNAPSHOT_KEYS`), so no script needs to change.
- **GitLab is unaffected.** `assets/gitlab/ci-spec-jobs.yml` parses no comments and reads the head with git at the pipeline's commit.

**Binding constraints:**
- `sdd/CONTEXT.md`: automation stays fork-safe by construction, declares its permissions, and pins actions by commit.
- Each skill's `references/github.md` `## Fork safety`: comment text reaches the shell only through `env`, with globbing off, and only as arguments to the script.
- `.agents/knowledge/skill-quality.md` `## Management code`: R5, a read of a file another step wrote is checked and fails naming it; R6, no `|| true` wider than the exit it is written for. The designed deferral stays: the reply is posted even when `show` or `status` fails, and the run then fails for every exit but 2.
- `.agents/knowledge/harness-maintenance.md` register row for the sdd workflow assets: this repository's workflow keeps the same steps and the same script path as the openspec asset, so the companion change moves with this one.
- #101 will bump the snapshot schema to `spec-snapshot/2`. Whatever it changes, it keeps `head.sha`, because the reply's head is read from there after this change.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| OSW — Behavior: The produced comment-command workflow answers every command it admits | `skills/sdd/openspec-workflow/assets/github/workflow-spec-command.yml`: the job's `if:` and the reply step's parse and header. `references/github.md` `## Fork safety`: the invocation-gates bullet names the admitted shapes, and a sentence says every admitted command is answered, an unknown or unreadable one with the command list. `## Verification after installing`: the test pull request bullet adds `/spec foo` answered with the command list. | existing load sentence of `references/github.md` |
| OSW — Behavior: The produced comment-command reply names and links the head its content was read from | the same asset: the resolving step reads the head from the snapshot it just wrote, and the pull request's state still from `gh pr view`; a missing head or state fails that step before it writes any output. `references/github.md` `## Fork safety`: the reply names and links the head the snapshot read; "a `snapshot` that fails fails the run before any reply" gains "and so does one that records no head". | as above |
| Both, comment only | the same asset: the `pull-requests: read` comment no longer says the job reads "the request and its files" | — |
| SKW — both requirements, and the comment | the same places in `skills/sdd/spec-kit-workflow/`: `assets/github/workflow-spec-command.yml`, `references/github.md` `## Fork safety` and `## Verification after installing` | existing load sentence |

`SKILL.md` of either skill does not change. Its command list names what the agent uses (`show`, `status`), and the reply itself lists the commands to anyone who types another.

## External impact

- **Companion `spec-command-reply-harness`:** `.github/workflows/spec-command.yml` carries the same admission, parse, head, and comment, and `.agents/knowledge/github-checks.md` states the new admission in the `spec / command` row. Proof: the companion's verification plan, including a `diff` of the three workflow files.
- **Register row** in `.agents/knowledge/harness-maintenance.md`: the text stays true ("same steps, same `scripts/spec_changes.py` path") once the companion lands. Proof: readback.
- **No change to** scripts, snapshot schema, descriptions, symlinks, `marketplace.json`, catalog `CONTEXT.md`, or the `sdd` README pair. Proof: `git diff --stat origin/main...HEAD` lists none of them; `just validate`.

## Decisions

- **Two ADDED Behavior requirements per domain, no MODIFIED block** (both requirements).
  - Admission and answering form one requirement, the head another, so each is one statement with its own scenarios, as the schema asks.
  - The installation requirement already holds the comment workflow's other clauses and is one long compound block. #99 MODIFIES that block in `openspec-workflow`, and two in-flight MODIFIED copies of one block overwrite each other at archive; ADDED requirements stay disjoint from it.
  - Rejected: MODIFYING the installation requirement, for the archive collision and the compound statement.
  - Rejected: one ADDED requirement covering admission, answering, and the head, which states two independent guarantees in one requirement.
- **Admission is tightened to `/spec` followed by the end, a space, a tab, or a line break** (answers every command it admits; maintainer decision, option B of the scoping).
  - A word that only begins with `/spec`, above all Spec-Kit's `/speckit.*` commands, starts no privileged run and draws no reply.
  - The proposed condition replaces `startsWith(github.event.comment.body, '/spec')`:
    ```
    (github.event.comment.body == '/spec'
     || startsWith(github.event.comment.body, '/spec ')
     || startsWith(github.event.comment.body, fromJSON('"/spec\t"'))
     || startsWith(github.event.comment.body, fromJSON('"/spec\r"'))
     || startsWith(github.event.comment.body, fromJSON('"/spec\n"')))
    ```
  - Checked against GitHub's expressions reference (docs.github.com, "Expressions", read 2026-09-30): a string literal is single-quoted and has no escape but `''`; `fromJSON` converts any value representable in JSON, strings included; `startsWith` and `==` ignore case. So the tab, CR, and LF literals come from `fromJSON` on a JSON string, and a folded YAML scalar passes the backslashes through unchanged. The reference shows no string example of `fromJSON`, so the tab, CR, and LF branches remain **to be verified on a live run**; see Risks.
  - Because the comparison ignores case, `/SPEC status` is admitted; the next decision settles its answer.
  - Rejected: keeping `startsWith('/spec')` and answering every non-command with the help text (option A). It spends a privileged run and the shared API budget, and posts a help comment, under every `/speckit.*` comment.
  - Rejected: keeping admission and exiting green with no reply for a non-command (option C). An admitted command answered by silence is what #96 reports.
  - Rejected: a filter step inside the job, which still spends the privileged run.
- **Admission ignores case, the parse does not: `/SPEC status` gets the command list** (answers every command it admits; not among the settled scoping decisions, so the maintainer confirms it on the pull request).
  - The reply step matches `/spec` and the command words case-sensitively, as today, so any other spelling is answered with the command list, whose header `/spec help` and body show the right spelling.
  - Rejected: a case-insensitive parse that answers `/SPEC status` with the status table. Only part of the line could be folded, because change or feature names and document names are case-sensitive paths, so the grammar would gain a rule (fold `/spec` and the command word, keep the rest) in all three copies for a spelling nobody documents; the command list already answers it in one comment.
  - Rejected: a case-sensitive admission check. GitHub's expressions reference (read 2026-09-30) documents `==`, `startsWith`, and `contains` as ignoring case and names no case-sensitive comparison, so the `if:` cannot express it; the only other place is a filter step inside the job, rejected above for spending the privileged run.
- **The parse is total** (answers every command it admits).
  - No admitted body makes the step fail before the reply. A carriage return is dropped from the body before the command line is read, spaces and tabs both separate words, and a line that yields no command reads as `help`.
  - The command line stays the first line that is `/spec` alone or followed by a blank, today's rule widened to the tab, so a reply to `/spec foo` above a later `/spec status` line is the command list, as today.
  - No `|| true` wider than the no-match exit it is written for (R6); comment text still reaches the shell only through `env`, with globbing off.
  - Rejected: an explicit `shell: bash` without `-e`, which would hide the next failure as well.
  - Rejected: reading only the comment's first line. It changes which line counts for no gain once admission is tightened, and a body whose first line differs only in case would still need the fallback to `help`.
- **The header echoes the command as it was read** (answers every command it admits; maintainer decision).
  - The header shows the parsed words joined by one space, with backticks dropped as today, and `/spec help` for anything answered with the command list. The commenter's own comment sits directly above the reply, so nothing is lost.
  - Rejected: echoing the raw line, which carries CR and any other character into the code span and would need its own sanitizing.
- **A tab separates words** (answers every command it admits; maintainer decision). `set --` already splits on tabs, and a pasted command often carries one. Rejected: a space only, under which `/spec` + TAB + `status` is not admitted at all.
- **The reply's head comes from the snapshot's `head.sha`; the state still from `gh pr view`** (names and links the head).
  - The resolving step reads `head.sha` from the snapshot it just wrote, failing with a message that names the snapshot file and the key when the key is missing or null (`jq -er`, whose `-e` makes a null fail). The state read uses `jq -er` as well, so a missing state no longer reaches the reply as `null`. Both reads fail the resolving step before it writes any output. These are the reads the first requirement names in its exception, "once the reads of its pull request succeed (the snapshot, the head the snapshot records, and the pull request's state)", so no read that runs after the snapshot can leave an admitted comment unanswered outside that exception.
  - Rejected: keeping the exception at "whose snapshot succeeds" and ordering the state read before `snapshot`. A snapshot that records no head would still sit between "succeeded" and "failed", and the requirement would depend on a step order the design would have to freeze.
  - Rejected: carrying the state in the snapshot to cut the step to one read. It changes both management scripts, their `Script:` requirements, and the snapshot schema for a race whose only effect is the "(pull request is X)" note.
  - Rejected: having `show` and `status` print the header or build the links from the snapshot themselves. It changes the scripts' output contract and every `Script:` requirement.
  - Rejected: passing the first head to `snapshot` and failing on a mismatch. A harmless race would become a red run with no reply.
- **This change rewords the stale permission comment** (comment only; maintainer decision). "Reading the request and its files is a pull-request read" stops being true when #101 drops the file list. Reworded to what holds before and after #101 (reading the pull request is a pull-request read), here, because this change already edits the same lines of all three files. Rejected: leaving it to #101, which then edits a block this change touched.
- **Verified by a command harness over the assets, with no subagent outcome task** (both requirements; maintainer decision).
  - The scenarios describe what the produced workflow does, and the produced workflow is the asset with its placeholders resolved, so running the asset's steps exercises them directly.
  - Rejected: re-running the "install the automation on GitHub" outcome task per skill. It would show that the agent copies the asset, which this change does not alter, at the cost of two solver runs and a grader.
- **The live admission check runs on this repository after the merge, not before it in a throwaway repository** (answers every command it admits; not among the settled scoping decisions, so the maintainer confirms it on the pull request).
  - `spec / command` runs only from a default branch, so no run of this pull request can evaluate the new `if:`. Before the merge, the `if:` is checked only against the documented rules and a local model of them.
  - Rejected: the maintainer runs the `if:` once, before the merge, in a throwaway repository whose default branch carries the workflow. It is the only way to see the `fromJSON` branches and the evaluation-error mode (Risks) before the assets ship. It is rejected because it needs a repository outside this one, created and administered by the maintainer, and a collaborator's comments posted there. Neither is among the actions `.agents/knowledge/agent-authority.md` grants an agent, and this repository's process has no step for either. If the maintainer prefers to take that cost, the check after the merge (Verification plan) moves there unchanged, and the archive commit waits for its result.

## Risks / Trade-offs

- **[The `fromJSON` branches of the `if:` do not evaluate as intended]** → Two failure modes, both **to be verified**:
  - A branch evaluates to false, so `/spec` followed by a tab or a line break is not admitted. The space and bare branches do not depend on `fromJSON` and would still admit their shapes.
  - A branch raises an evaluation error. GitHub's expressions reference (read 2026-09-30) says nothing about how an evaluation error in an `if:` is reported or whether `&&` and `||` short-circuit. Such an error could reach every pull request comment that no earlier branch settles, not only `/spec` comments, and fail or error the job for each.
  - Before the merge: the documented rules are quoted with their date above, and a local model of them is run over the case table. That model shares the author's reading of the rules, so it does not settle either mode.
  - After the merge: the maintainer's check on a test pull request (Verification plan) covers every admission case.
  - Fallback: a failure found live is fixed by a new fix change against the merged skills, which at its narrowest reverts the `if:` to the space and bare branches. This change is archived by then and is not revised. Projects that installed the automation in between keep the faulty copy until they reinstall.
  - The alternative that catches both modes before the assets ship, a run in a throwaway repository, is recorded under Decisions with the reason it is not the plan.
- **[Case-insensitive admission admits `/SPEC …`, `/Spec …`]** → Accepted under Decisions: such a comment gets the command list, which shows the right spelling.
- **[It is not verified that GitHub's comment editor sends CRLF]** → The fix is harmless either way.
- **[The step order changes: the head is written after `snapshot`]** → A failed snapshot still fails the step before any output is written, as today.
- **[Coordination with #99 and #101, which also change `openspec/specs/sdd/*` and the same files]** → Planned merge order: #99, then this change, then #101.
  - After a preceding sdd pull request merges, this branch rebases on `main` and re-copies every MODIFIED requirement block from the then-current `main` before its archive commit, so no archive silently reverts another change's clause. This change carries no MODIFIED block today; the rule applies if the deliberation turns one of its requirements into one.
  - Its ADDED requirement names are unique in both domains, so neither #99 nor #101 can collide with them.
  - #101 edits the same resolving step and `## Fork safety`; it rebases onto this change and keeps `head.sha` in `spec-snapshot/2`.
- **[Projects that installed the automation earlier keep the old workflow]** → Accepted: installed copies are not updated in place. They reach the fix by reinstalling.

## Verification plan

Written before implementation; results go to the pull request's Validation section.

**Harness.** In an untracked scratch directory under the session scratch directory, for each skill's `assets/github/workflow-spec-command.yml`:
- The two `run:` bodies are extracted by step name: "Resolve the pull request and snapshot its head through the API" and "Run the command and reply".
- A stub `gh` on `PATH`: `pr view` prints `{"headRefOid": A, "state": S}`, and `pr comment` copies its `--body-file` aside.
- `scripts/` holds a wrapper that runs the skill's management asset, adding `--api-url` of a local stub HTTP server to `snapshot`. The server serves the pull request, its files, trees, and blobs from a fixture repository whose `pulls/{n}` reports head B ≠ A. The fixture's change or feature holds one document over 60000 characters.
- The resolving body runs with `bash -euo pipefail`, the reply body with `bash -e`, with `N`, `REPO`, `RUNNER_TEMP`, `GITHUB_OUTPUT`, and `COMMENT_BODY` set. `HEAD` and `STATE` come from the resolving step's `GITHUB_OUTPUT`.
- Every case also runs against `origin/main`'s asset first, to show it failing where the scenario says it did.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| OSW, SKW: Word that only starts with /spec (SKW: Kit command in a comment) | the `if:` read against the documented rules; a local model of those rules (case-insensitive `startsWith` and `==`, `fromJSON` of a JSON string) over `/specs`, `/specification`, `/speckit.plan`, `/spec`, `/spec status`, `/spec` + TAB, `/spec` + CR LF, `/spec` + LF, `/SPEC status` | the three words are not admitted (C); the other six are (C) | all | none (command) | model output | scratch directory |
| OSW, SKW: Command line ends in a carriage return | `COMMENT_BODY=$'/spec status\r\nthanks'` | step exits 0 (C); posted body starts with `` `/spec status` `` and holds no CR (C); body carries the status table (C) | all | none (command) | posted body, exit | scratch directory |
| OSW, SKW: Tab after /spec | `$'/spec\tstatus'` | exits 0; header `/spec status`, status table (C) | all | none | same | same |
| OSW, SKW: Bare /spec followed by more text | `$'/spec\r\nthanks'` | exits 0 (C); header `/spec help` and the command list (C) | all | none | same | same |
| OSW, SKW: Unknown command | `/spec foo`, and `/specs` fed to the step directly | exits 0 (C); header `/spec help` and the command list (C) | all | none | same | same |
| OSW, SKW: Command in another case | `/SPEC status` | exits 0 (C); header `/spec help` and the command list (C) | all | none | same | same |
| OSW, SKW: Head moves between the two reads | stub `gh` reports A, snapshot records B; `/spec show` | header names B's short SHA (C); every `[view]` link and the truncation link start with `https://github.com/o/r/blob/B/` (C); A appears nowhere in the body (C) | all | none | same | same |
| OSW, SKW: Status names the snapshot's head | same heads; `/spec status` | header names B (C) | all | none | same | same |
| OSW, SKW: Closed request | stub `gh` reports `CLOSED`; `/spec status` | header names B and "(pull request is CLOSED)" (C) | all | none | same | same |
| OSW, SKW: Snapshot without a head | the wrapper writes a snapshot with `head.sha` removed, then with it `null` | resolving step exits non-zero, with a message naming the snapshot file (C); `GITHUB_OUTPUT` holds no `head` (C); the reply step is not reached | all | none | exit, stderr, output file | same |

The admission halves of "Word that only starts with /spec" (SKW: "Kit command in a comment"), "Tab after /spec", "Bare /spec followed by more text", and "Command in another case" rest, before the archive, only on the local model of the documented rules. The Validation section says so, and their live evidence is the check after the merge below.

Regression, on the same harness:
- Installation requirement, "Command answered when the script fails": a wrapper whose `status` exits 1 gets its output posted and the step exits 1; one exiting 2 gets its output posted and the step exits 0.
- `/spec`, `/spec status`, and `/spec show nope` give the same bodies as on `origin/main`, apart from the head.
- Each asset parses as YAML; `grep -n 'git .*fetch\|checkout'` finds only the base checkout; the permissions block is unchanged apart from its comment.
- `diff` of the two assets shows only the vocabulary and the script name.

Readback: both `references/github.md` files, `## Fork safety` and `## Verification after installing`, state the admitted shapes, the answer to an unknown command, and the snapshot's head, and the unchanged "Command answered when the script fails" clause.

Checks: `just check-skill skills/sdd/openspec-workflow skills/sdd/spec-kit-workflow`, then `just check`.

After the merge, a maintainer action, since `spec / command` runs only from the default branch. On a test pull request, with bodies whose bytes are controlled posted through `gh api` (`repos/{owner}/{repo}/issues/{n}/comments` with a `body` field) rather than typed in the editor, whose line endings are not verified:
- Each of these gets exactly one reply naming the current head: `/spec status` (status table); `/spec foo` and `/SPEC status` (command list under `/spec help`); `/spec` + TAB + `status` (status table); `/spec status` + CR LF + `thanks` and `/spec status` + LF + `thanks` (status table, header without CR); `/spec` + CR LF + `thanks` and `/spec` + LF + `thanks` (command list).
- `/specs`, `/specification`, `/speckit.plan`, and a plain `thanks` each show the `spec / command` job as skipped in the Actions list, not failed or errored, and draw no reply.
- A failure here goes to the fallback under Risks.

Skipped:
- Trigger cases: no description changes.
- Subagent outcome tasks: the decision above.
- Live runs of the Spec-Kit asset: this repository runs only the OpenSpec workflow; the harness covers the Spec-Kit copy.
