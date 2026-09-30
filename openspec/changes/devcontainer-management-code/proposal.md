## Why

`devcontainer-authoring` hands a new Feature or Template collection a scaffold of workflows, composite actions, test and smoke scripts, and a `justfile`. That is management code, and the management-code change (#94) left it unread as a non-goal. Read against `skill-quality.md` `## Management code` (R5, R6), it turns failures into green runs: a jq error becomes an empty test matrix, a failed commit reads as "no documentation changes", a feature with no `duplicate.sh` passes its idempotency test without running one, and a template whose boolean option defaults to `false` is rejected as having no default. A developer who scaffolds a collection with the skill gets checks that pass when they should fail.

## What Changes

- The Feature scaffold's workflows, composite actions, test scripts, and recipes hide no unexpected failure:
  - the matrix and feature-list steps check the JSON they read and write no output built on a failed command;
  - `just validate` checks every manifest, names each malformed one, and fails once at the end;
  - a `compatibility.txt` that declares no image fails naming the file, in CI and in `just test-compat`; a missing one keeps the documented default image;
  - `just test-compat` tests every declared image whatever the CLI does with standard input;
  - the shared test action refuses a per-feature mode without a feature id;
  - the example tests keep the exit status of the command they assert;
  - the test workflows drop the guard on an output the path filter always sets;
  - the matrices that finish every leg before failing say so where they are written.
- Tests the Dev Container CLI would skip are no longer read as passes. The idempotency test script `test/<id>/duplicate.sh` is required: the shared test action and `just test-duplicate` fail naming it when it is absent, and the scaffold's example feature ships one. A feature without `scenarios.json` and a collection without `test/_global/scenarios.json` are skipped visibly, with a notice naming the file and a comment where the skip is written.
- The Template scaffold's smoke test hides no unexpected failure:
  - it parses the manifest once and fails naming it when it does not parse;
  - it substitutes every declared default, `false` and the empty string included, and still refuses an option with none;
  - it fails naming `test/<id>/test.sh` when the template has none;
  - it runs the test script with the interpreter its first line names;
  - its cleanup is marked best effort, names containers it could not remove, and keeps the test's exit status;
  - it runs whether or not the script keeps its exec bit.

  Its example test keeps the status of every command, and its test workflow and `just validate` get the same repairs as the Feature scaffold's.
- Both release workflows open a documentation pull request only when a regenerated README changed, and fail when the commit fails.
- `references/feature-testing.md` no longer shows an idempotency check that swallows the re-install failure it exists to catch (`install.sh || true`). It proves repeat installation with the duplicate mode and `duplicate.sh`. Its example checks keep the status of the command they assert. The references that describe the scaffolds are updated where the behavior they describe changes.

## Skills touched

- `engineering/devcontainer-authoring` (new): the management code the Feature and Template scaffolds deliver, how their tests treat files the CLI would skip, the release documentation step, and the feature tests the agent writes.

## Installed behavior

An agent that scaffolds or maintains a Feature or Template collection with the skill no longer delivers steps that read a failure as success. It no longer rejects a valid `false` or empty default, and no longer writes an idempotency check that cannot fail. Correcting wrong behavior makes this a `fix`.

## Impact

- No description, catalog `README.md` or `README.zh.md` row, symlink, `marketplace.json` entry, or catalog `CONTEXT.md` changes. No file listed in `scripts/validate_harness.py` or `.agents/knowledge/harness-maintenance.md` pairs with these assets.
- No project harness file changes. The assets are Bash and YAML, which `just lint` does not cover. A lint for Bash assets is proposed as a separate repository issue, not a companion change.
- Collections scaffolded earlier own their copies and do not receive these repairs.

## Non-goals

- The Feature's `src/*/install.sh`, which is the product the collection ships, not management code.
- The skill's description, the prebuilt-image branch, and the other `engineering` skills.
- Porting the scaffold scripts out of Bash.
- A validate workflow for the Template scaffold, and the Feature `test-multios` concurrency group, which are neither failure-hiding nor in the issue.
- Steps that already conform, such as the path-filter generation, the shellcheck severity threshold, the main-only release guard, and the empty feature list on a collection with no features. They stay as they are.

## Tracked work

Issue #98.
