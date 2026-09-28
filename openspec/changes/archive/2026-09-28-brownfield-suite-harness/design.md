## Context

See proposal.md for motivation, and `openspec/changes/brownfield-suite/design.md` for the skills this change serves.

**Current state of the harness:**
- **`engineering/CONTEXT.md` `## Dependencies`** grants `core` only: "they are installed one at a time per project, so co-presence is never guaranteed".
- **`engineering/CONTEXT.md` `## Disambiguation`** routes "converting a prototype or brownfield codebase" to `spec-driven-development`.
- **`skill-quality.md`** lists the current grants: "`meta` builders on one another, `scaffold` builders on `meta`".
- **Register row.** `harness-maintenance.md` has a row that ties any `## Dependencies` grant to that sentence and to the `ARCHITECTURE.md` catalog bullet.
- **Validators.** `scripts/validate_harness.py` lists every mechanically checked pair in its docstring, and the register defers to that list. `scripts/validate_skills.py` already holds one skill-to-skill identity check, `check_meta_harness_methodology`, which is not in either docstring.

**Binding rules:**
- `check_pointers` fails when a backticked path in `ARCHITECTURE.md` or `AGENTS.md` does not exist.
- Branch commits are rebase-merged, so every commit must pass `just check` on its own.
- The harness commit lands before any `brownfield-*` directory exists.

## Placement

| What Changes bullet | File and section | Proof |
|---|---|---|
| Suite grant, override, and duplication note | `skills/engineering/CONTEXT.md` `## Dependencies`: a new bullet after the default-range bullet naming the five members | Readback: the grant names all five members and the installer route. A reader of the old default bullet can no longer conclude that members are forbidden to depend on one another. |
| Class list | `skills/engineering/CONTEXT.md`, the opening class list: a third class, the brownfield suite, described as methodology that transfers across stacks | readback |
| Suite prefix | `skills/engineering/CONTEXT.md` `## Naming` | Readback; `just validate` passes with no `CATALOG_NAME_PREFIXES` change |
| Split route | `skills/engineering/CONTEXT.md` `## Disambiguation`: the brownfield clause is split, and "the two skills here" names `session-retrospective` and `knowledge-deposition` | readback |
| `sdd` pointer | `skills/sdd/CONTEXT.md` `## Scope` (one clause) and `## Disambiguation` (one route) | readback |
| Grants sentence | `.agents/knowledge/skill-quality.md`, the "today:" parenthesis | readback against the `CONTEXT.md` grant |
| Architecture bullet | `ARCHITECTURE.md` `## Catalogs`, the `engineering` bullet. It names the suite by prefix, not by backticked path. | `just validate` (`check_pointers` passes before the skills exist) |
| README introduction | `skills/engineering/README.md` and `README.zh.md`: one paragraph plus one install example listing the five `--skill` names | readback of both files for identical content |
| Shared-material check | `scripts/validate_harness.py`: a new check in `main()` beside `check_copies`, and a docstring entry `suite` | See the verification plan |

## Decisions

- **The grant lives in `engineering`, scoped to the suite by name** (serves the grant bullet).
  - The grant lists the five members explicitly, and the prefix only names the suite.
  - A missing member is installed by name through the installer, whose handoff sentence lists every missing member. This is the meta pattern minus whole-catalog install, which the installer reserves for `meta`.
  - Rejected: granting on the prefix alone. A future `brownfield-` skill would enter the grant without review.
- **The check goes in `validate_harness.py`, not `validate_skills.py`** (serves the check bullet).
  - The synchronization register says that mechanically checked pairs live in `validate_harness.py` and are listed in its docstring.
  - Rejected: placing it beside `check_meta_harness_methodology`. It would be as undocumented as that check is.
- **What the check compares** (serves the check bullet):
  - It compares one item: the `## Evidence discipline` section, found by its exact heading. The workspace and record rules are a `###` subsection inside it, so they fall inside the compared text.
  - The source is `brownfield-investigation`, the first member to land.
  - The member set is every directory `skills/engineering/brownfield-*`.
  - A missing section, or members present without the source, is an error, never a skip. With no members present, the check passes, so this commit is valid on its own.
  - The section is cut at the next level-2 heading. Level-3 headings do not end it. The skill design keeps code fences out of the shared section, records included, so a `## ` line inside a fence cannot end the comparison early.
  - The error names the member file to overwrite and the source to copy from.
- **No register row** (serves the check bullet). The register already defers mechanically checked pairs to the validator docstring. Adding a row would state the pair twice.
- **`sdd/CONTEXT.md` changes in two places only** (serves the pointer bullet). The user asked that both catalogs point at each other. `spec-driven-development` itself stays untouched, because its adoption reference remains correct for adopting the loop.

## Risks / Trade-offs

- [The check's section cut misreads a fence] → The skill design forbids fences and templates in the shared section and writes its records as field lists. Verification step 2 edits the last subsection, which proves that the cut reaches the end of the section.
- [The grant weakens the "installed one at a time" rule for other engineering skills] → The grant names five skills explicitly, and the default bullet stays as it is for every other skill.
- [`ARCHITECTURE.md` names paths that do not exist yet] → The bullet names the suite by its prefix in prose, never as a backticked path.

## Verification plan

Per What Changes bullet:

- **`CONTEXT.md` grant, class list, naming, and disambiguation; `sdd/CONTEXT.md`; `skill-quality.md`; `ARCHITECTURE.md`; README pair.**
  - A clean-context readback confirms each placement row's statement.
  - `diff` of the README pair's structure: same headings, same rows, same code block.
  - `just validate` passes.
- **Shared-material check,** run in a disposable worktree of the branch after the five skills exist:
  1. `just validate` passes.
  2. Change one line in the `### Workspace and records` subsection of `brownfield-onboarding`'s `## Evidence discipline` section. `just validate` exits non-zero, and the error names `skills/engineering/brownfield-onboarding/SKILL.md` and the source. Revert.
  3. Rename the `## Evidence discipline` heading in `brownfield-migration/SKILL.md`. The error names the missing section and the file. Revert.
  4. Rename `brownfield-investigation` out of the way while other members exist. The error names the missing source. Revert.
  5. On the harness commit alone, before any member exists, `just validate` passes.
  6. `just lint` passes on the edited script.
- **Whole change:** `just check`, and `git diff --stat origin/main...HEAD` shows only the files listed in the Placement table plus the skill change's files.
