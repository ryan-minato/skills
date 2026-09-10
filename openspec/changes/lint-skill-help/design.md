## Context

See proposal.md. The skill lives in `core`, so its SKILL.md stays short and
it depends on nothing outside `core` (`skills/core/CONTEXT.md`). The script
is one file with PEP 723 metadata (`scripts/lint_skill.py` lines 1–7,
`pyyaml>=6.0,<7`, `requires-python >=3.11`) and `compatibility: Validation
requires uv.` in SKILL.md. The only PyYAML consumer is `parse_frontmatter`
(lines 68–84), called from `lint()` after `main()` has parsed the arguments
and resolved the skill path (lines 571–583); `--skill` pointing at a
non-skill path already exits 2 through `parser.error`, and a missing
SKILL.md exits 1. Precedent for a deferred import with an actionable
message: `skills/meta/meta-github-workflow/assets/check_taxonomy.py`
(`import_yaml()`, exit 2). Script rules: `.agents/knowledge/skill-quality.md`
Scripts (`--help` is the discovery path; exit codes 0/1/2 with deviations
documented in `--help`; errors say what, why, how). `ruff.toml` selects no
`PL` rules, so a function-local import passes `just lint`.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| Trigger: description | `SKILL.md` frontmatter `description` — unchanged; the block records the domain's baseline | — |
| Script: lint_skill.py | `scripts/lint_skill.py`: `import_yaml()` above `parse_frontmatter`, called as its first statement; module docstring exit codes; `main()` parser `formatter_class` and `epilog` | — |

## External impact

None. The description is unchanged, so the `core` README pair rows stay;
no file is added, moved, or removed, so the `.agents/skills/` symlink,
`marketplace.json` (`scripts/gen_marketplace.py` lists directories only),
`skills/core/CONTEXT.md`, and the mirrors in `scripts/validate_harness.py`
are untouched. Proof: `just validate` and `git status --short` listing only
the script and the change record.

## Decisions

- **Exit 2 for a missing PyYAML, not 1** (serves Script: lint_skill.py).
  Exit 1 is documented as "one or more errors found" and `--json` consumers
  read it as findings to parse; a missing interpreter dependency is an
  invocation problem of the same family as bad arguments. Matches
  `check_taxonomy.py`. The deviation from the bare 0/1/2 meanings is
  written into `--help`, as the script rules require.
- **Import inside `parse_frontmatter`, not at the top of `main()`**: the
  issue asks for the import "only when YAML is actually parsed", and it
  keeps `main()` untouched; the argument and path checks run first either
  way. Alternative rejected: importing in `main()` after `parse_args()`,
  which would still fail the missing-path case before its own diagnostic.
- **Exit codes in the argparse epilog with `RawDescriptionHelpFormatter`**
  rather than `description=__doc__`: `--help` currently prints only the
  description and epilog, so the docstring alone would not satisfy
  "document deviations in `--help`"; the epilog is the smallest visible
  change.
- **Missing skill path keeps exit 1** and is enshrined as a scenario;
  changing it is out of scope.

## Risks / Trade-offs

- [`uv run` cannot resolve PyYAML offline] → `uv run --offline` works here
  (cached wheels); without cache or network the representative and
  repeated runs are blocked, not skipped, and the run waits.
- [The host has PyYAML, so "cannot import yaml" cannot be observed] → a
  fresh `python3 -m venv` (no site packages) is the interpreter.
- [Trigger solvers see this repository's whole skill list, including
  `skill-authoring`, which pairs with the target] → the target's own load
  decision is what is graded; the solver works in a fixture project outside
  the repository so it cannot answer from `.agents/knowledge/`.

## Verification plan

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| Trigger: Skill authoring request | "Write a SKILL.md so our agents follow the release checklist every time" in the fixture project | loads `great-skill-writing` (critical) | 1/1 | Sonnet-class | appended neutral `SKILLS_LOADED:` self-report | fresh clean-context subagent per prompt, fixture project in scratch, one attempt (up to three on an invalid observation) |
| Trigger: Misbehaving skill, indirect phrasing | "the instruction package I gave my agent for changelog entries never gets picked up — fix it" (fixture carries `.agents/skills/changelog-entries/SKILL.md` with a vague description) | loads (critical) | 1/1 | same | same | same |
| Trigger: Human documentation (near-miss) | "write a README that explains how to run the release checklist" | does not load (critical) | 1/1 | same | same | same |
| Trigger: Human skills (near-miss) | "which skills should a junior engineer build first?" | does not load (critical) | 1/1 | same | same | same |

Script and tool harnesses (`S=skills/core/great-skill-writing/scripts/lint_skill.py`;
`NOY` is the interpreter of a fresh venv where `import yaml` fails):
- Help: `$NOY $S --help` → usage on stdout naming the exit codes and the
  PyYAML case; exit 0; no traceback.
- Bad arguments: `$NOY $S --bogus` → stderr names `--bogus`; exit 2. Also
  `$NOY $S` (no `--skill`) → stderr names `--skill`; exit 2.
- Missing skill path: `$NOY $S --skill /nonexistent/SKILL.md` → stderr
  `Error: '/nonexistent/SKILL.md' does not exist.`; exit 1; no traceback.
- Missing dependency: `$NOY $S --skill skills/core/great-skill-writing` →
  stderr names PyYAML, the frontmatter, `uv run`, and `pip install
  'pyyaml>=6.0,<7'`; exit 2; stdout empty; no traceback.
- Representative run: `uv run --offline $S --skill skills/core/great-skill-writing`
  → `OK  skills/core/great-skill-writing: all checks passed.`; exit 0;
  `--json` prints a JSON array; `uv run --offline $S --help` exits 0.
- Repeated run: the representative command twice into files; `diff` empty;
  `git status --short` unchanged.
- `just check-skill skills/core/great-skill-writing`, `just lint`,
  `just spec-validate`, `just check`.

Skipped: none.
