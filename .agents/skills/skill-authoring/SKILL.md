---
name: skill-authoring
description: End-to-end workflow for creating or modifying a skill in this repository — subagent-assisted behavioral evaluation, isolated script tests, scaffolding, symlinks, marketplace manifest, catalog READMEs, validation, and commit. Use when asked to "create a skill", "add a skill to a catalog", "move a skill", "remove a skill", or when modifying any skill under skills/ or .agents/skills/.
metadata:
  internal: true
---

# Skill Authoring

Project-only workflow skill for this repository. Follow it whenever a skill
is created, modified, moved, or removed.

## Define the skill first

Answer these before creating any files; they decide whether the skill is
worth building and where it belongs:

1. What task does the skill cover? Is it one coherent unit of work (not a
   grab-bag, not a sliver of a larger workflow)?
2. When should it trigger — including indirect phrasings a user might say
   without naming the domain?
3. What does the agent lack without it? If the agent already handles the
   task well, do not build the skill.

## Before writing anything

1. Read `.agents/knowledge/skill-quality.md` — the quality bar every skill
   must meet (scoping, structure, spec limits, self-containment,
   description/body standards, instruction patterns, progressive
   disclosure, script rules).
2. Decide where the skill lives:
   - **Public skill** (distributable): `skills/<catalog>/<skill-name>/`.
     Pick the catalog from the list in `ARCHITECTURE.md` (Catalogs section)
     and read that catalog's `CONTEXT.md` for catalog-specific requirements.
   - **Project-only workflow skill** (serves this repo itself):
     `.agents/skills/<skill-name>/` as a real directory.
3. Apply the subagent gate below. If it passes, read
   [references/testing.md](references/testing.md) and design its behavioral
   tests before editing. If it fails, do not read the reference; record the
   skipped behavioral tests and missing capability for the Linear milestone
   comment and handoff. The `issue-workflow` skill must already have placed the
   current worktree on the issue branch; edit the skill there.

Done when: the target location is recorded and either the behavioral cases,
rubric, critical failures, and passing threshold are defined or the missing
subagent capability is recorded.

## Gate behavioral tests by subagent support

The gate passes when the invoking agent can dispatch clean-context subagents;
this is the only required capability. Do not maintain a framework allowlist or
denylist. When it passes, try disposable candidate worktrees and snapshots for
each test run; if either is unavailable, use the best available environment and
record the isolation degradation. Prefer framework-native conversation history
or skill-load telemetry to observe invocation. If neither is available, the
testing reference supplies neutral self-report instrumentation. If subagents
are unavailable, do not load the reference or use the authoring agent as a
solver or grader; record the skipped tests and missing capability in the Linear
milestone comment and handoff.

## Creating a public skill

1. Scaffold `skills/<catalog>/<skill-name>/SKILL.md` with `name` (must equal
   the directory name) and `description` frontmatter. Add `scripts/`,
   `references/`, or `assets/` only when a file is going into them. No
   README.md in the skill root; content in English; nothing may reference
   paths outside the skill directory. While drafting, lint the skill in
   isolation:

   ```bash
   just check-skill skills/<catalog>/<skill-name>
   ```

2. Symlink it into the agent-visible directory (relative link, from repo
   root):

   ```bash
   ln -s ../../skills/<catalog>/<skill-name> .agents/skills/<skill-name>
   ```

3. Run `just gen-marketplace` to sync the skill into its catalog plugin's
   `skills[]` in `.claude-plugin/marketplace.json` (the validator enforces the
   list matches the catalog exactly). A brand-new catalog also needs a
   hand-added plugin entry —
   `{ "name": "<catalog>", "source": "./", "strict": false, "skills": [] }`
   with a `description` — before running the generator.
4. Add the skill to the catalog's `README.md` **and** `README.zh.md` tables.

## Creating a project-only skill

1. Create `.agents/skills/<skill-name>/SKILL.md` (real directory, no
   symlink, never listed in marketplace.json or catalog READMEs).
2. Repo path references are allowed here, but all other quality standards
   still apply.

## Removing or moving a skill

Remove/update the symlink in `.agents/skills/`, the catalog README.md +
README.zh.md entries, and run `just gen-marketplace` to resync
`.claude-plugin/marketplace.json` (remove a plugin entry by hand only if its
catalog became empty). The validator catches anything missed.

## Test bundled scripts without subagents

For every added or changed bundled script, first try a detached disposable
candidate worktree and complete temporary snapshot immediately before testing.
If either is unavailable, generate the untracked test harness outside version
control in the best available environment and record the isolation degradation.
Verify:

- `--help` exits 0 and includes a usage example;
- a representative invocation exits 0 with expected output;
- repeating it proves idempotence;
- bad arguments exit 2 with an actionable diagnostic.

Record commands and results in the Linear milestone comment and handoff, then
remove the worktree, harness, fixtures, and outputs. This test does not require
subagents or loading the behavioral-testing reference.

## Finish

1. If the subagent gate passed, run the candidate tests from
   [references/testing.md](references/testing.md), applying the best available
   isolation, then apply the smallest general fix for each failure in the
   current worktree and rerun the complete affected evaluation. Otherwise
   confirm the skipped tests and missing subagent capability are recorded.
2. Walk the "Checklist before committing" in
   `.agents/knowledge/skill-quality.md`.
3. Run `just check` (repo-wide validation, including symlinks, the
   marketplace manifest, and catalog consistency) and fix everything it
   reports; warnings deserve a look even though they don't fail.
4. Commit using the repository convention (`.gitmessage`): Conventional
   Commits, scope = the skill name. Classify a distributable skill change by
   its effect across SKILL.md, references, assets, and scripts. First ask
   whether it corrects wrong, misleading, overly restrictive, or overly
   permissive installed behavior; if so, use `fix` even though the result is
   an improvement. Otherwise use `feat` for a new capability and `refactor`
   for a behavior-preserving restructure. Reserve `docs` for supporting
   documentation that does not change an installed skill; Markdown alone does
   not make a change documentation-only. Examples:
   `feat(<skill-name>): add <skill-name> skill` and
   `fix(<skill-name>): relax incorrect validation rule`.
