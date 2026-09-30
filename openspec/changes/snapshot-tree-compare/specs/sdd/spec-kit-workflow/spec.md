## MODIFIED Requirements

### Requirement: Script: assets/spec_kit_features.py
The management script the automation deposits SHALL resolve a request's touched features (numbered directories under the kit's specs directory whose files the request touches) from either of two head sources — a base and head ref read with git plumbing, or a snapshot document read with `--snapshot` — and SHALL offer the `snapshot`, `check`, `show`, `status`, and `labels` subcommands and none that edits the working tree; `snapshot` SHALL build that document from the platform's REST API without fetching or checking out the head and without the platform's list of the request's changed files, SHALL take as touched every numbered directory under the specs directory that has a different tree at the head than at the merge base of the request's base and head, as the platform's comparison of those two commits reports that merge base, or is a directory at only one of the two, so that it finds the same touched features as the git source for a request of any size, and SHALL record in the document the head commit it read; `snapshot` SHALL check every response it relies on — its shape, the merge base, the truncation flag, the encoding, and text decoding — and SHALL fail naming the path or endpoint, writing no document, on a cap reached or on any partial or unexpected read; a snapshot of another schema, or built for a different specs directory, SHALL be refused; `check` SHALL fail when a touched feature lacks its specification or plan, or when a touched feature has an open task and the request is ready (a warning with `--draft`, the specification request admitted under `--shape split`), an unchecked box with no text counting as an open task; `show` SHALL print a feature's documents inside a fence and truncate at `--max-chars` with a link; `show` and `status` SHALL answer a feature name the request does not touch with the touched features' names, every name rendered as code; `labels` SHALL compute the desired progress label and the additions and removals against the current set, and SHALL report the managed label names with `--taxonomy`; the script SHALL exit 0 on success, 1 on a failure or a finding, and 2 on bad arguments, an unknown feature name included, and SHALL let an unexpected error end with its traceback.

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage naming the five subcommands and exits 0

#### Scenario: Representative run
- **WHEN** a touched feature has its specification, plan, and a fully ticked task list and `check --base <base> --head <head>` runs
- **THEN** it exits 0, and `status` with the same refs reports the feature as done

#### Scenario: Repeated run
- **WHEN** the identical `check` and `status` commands run a second time
- **THEN** the output is identical and nothing in the tree changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option

#### Scenario: Unknown feature named
- **WHEN** `show` is given a feature name the request does not touch
- **THEN** it exits 2 and lists the touched features, and every feature name in its output, the given one included, is rendered as code

#### Scenario: Missing plan
- **WHEN** a touched feature has a specification but no plan
- **THEN** `check` exits 1 and names the feature and the missing file

#### Scenario: Partial read refused
- **WHEN** `snapshot` reaches a cap, meets a truncated tree, or finds a document it cannot decode as text
- **THEN** it exits 1 naming the path and writes no document

#### Scenario: Unexpected API response
- **WHEN** a response `snapshot` relies on is not the documented shape — the comparison of the base and the head answering without a merge base commit included — or a tree that a listing named answers 404
- **THEN** `snapshot` fails naming the endpoint and writes no document, instead of reading the response as empty

#### Scenario: Snapshot source matches the git source
- **WHEN** `snapshot` runs against a pull request and `status`, `show`, and `labels` run against the resulting document
- **THEN** their output is identical to the same subcommands run with `--base` and `--head` over a clone of that pull request

#### Scenario: Request past the file-listing limit
- **WHEN** `snapshot` runs against a pull request that changes more than 3000 files, one of them a feature's task list, and `status` and `labels` run against the resulting document
- **THEN** `snapshot` exits 0, and the output of `status` and `labels` is identical to the same subcommands run with `--base` and `--head` over a clone of that pull request

#### Scenario: Nothing under the specs directory touched
- **WHEN** `snapshot` runs against a pull request that changes files only outside the specs directory, and `status --json` and `labels --current` run against the resulting document
- **THEN** `status --json` lists no touched feature, and `labels`, given every managed label as current, reports an empty desired set and each of them as a removal

#### Scenario: Feature edited only on the base branch
- **WHEN** a feature directory is edited on the base branch after the pull request's branch left it, the pull request does not touch that directory, and `snapshot` then `status` run
- **THEN** that feature is not among the touched features

#### Scenario: Feature removed by the request
- **WHEN** the pull request deletes a feature directory, and `snapshot` then `status` run
- **THEN** the feature is reported as removed, exactly as `status` with `--base` and `--head` reports it

#### Scenario: Snapshot of an earlier schema
- **WHEN** `status` is given with `--snapshot` a document whose schema is not the one this script writes
- **THEN** it exits 1 naming the schema, says to rebuild the snapshot, and reports no feature
