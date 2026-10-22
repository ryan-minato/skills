---
name: code-review
description: >-
  Project-calibrated code review for this skill library — reviews a diff, PR
  number, branch, or path as instructions executed by agents, treats leaked
  secrets or private data as the most severe finding class, applies the
  repository's severity ladder, script threat model, and harness-sync checks,
  then reports verified findings ranked most severe first. Use when reviewing
  changes in this repository: "review this PR / diff / branch", "look over
  these changes before merge", "check the PR for leaked secrets or content
  that should not be committed", or any /code-review request here. Replaces
  the generic review behavior and carries its baseline (effort levels,
  find-verify-report, --fix, --comment). Not for authoring or testing skills
  (skill-authoring), the commit and PR workflow itself (change-workflow), or
  a standalone scan of arbitrary material (sensitivity-check).
metadata:
  internal: true
---

# Project Code Review

Public skills in this repository are instructions executed verbatim by LLM
agents of varying capability. Review behavior, not prose: the question is
never "is this well written" but "which reader, on what input, does the
wrong thing". This file is also read by GitHub Copilot code review from its
git path; every rule below applies to any reviewer, human or machine.

## Baseline procedure

1. **Resolve the target.** Default is the current branch's diff against the
   main branch (staged and unstaged included); a PR number, branch, or path
   argument overrides it.
2. **Calibrate effort.** `low`/`medium`: fewer, high-confidence findings
   only. `high` and above: broader coverage; uncertain findings allowed but
   marked as such. When no level is given, reuse the last one used.
3. **Find, verify, report.** Draft candidate findings, verify each one
   (see "Verifying findings"), then report exactly once: through the
   host's typed findings tool when the harness provides one — ranked most
   severe first, empty if nothing survived, never duplicated as prose —
   otherwise as a ranked list.
4. **Arguments.** `--fix`: after reporting, apply the surviving findings to
   the working tree. `--comment`: post findings as inline PR comments — a
   remote write, so `agent-authority.md ## Publishing` applies to the
   exact comment payloads.
5. A deep multi-agent cloud review ("ultra") is a host feature, not part of
   this skill; when it is unavailable, run at maximum effort locally and
   say that is what happened.

## Severity ladder

1. **Critical — leaked secrets, private data, or content that must not be
   committed.** Always the top of the report, at every effort level.
2. **Blocking — wrong installed behavior**: an agent following a published
   skill's text takes a wrong or unsafe action; agent authority widened by
   default (ready/review/merge granted on green checks, a weakened
   no-self-escalation rule); a builder-defined model noun inside a
   deposited platform file, or a platform object decided during the
   platform-neutral design stage; a contradiction between pieces that load
   together; a breach of
   `skill-quality.md ## Self-containment and dependencies`.
3. **Recommended**: a latent ambiguity with a plausible failure path; a
   harness desync (see the quality gradient).
4. **Note**: grammar, terminology, phrasing. Never marked blocking, and
   never proposed in a way that violates the constraint surface below.

## Pass one, on every review: sensitive and misplaced content

Before any quality finding, re-examine the full added content — prose,
fixtures, and data included, not just code — for material that should not
be committed:

- Credentials of any kind: tokens, API keys, private keys, passwords,
  signed URLs, cloud account identifiers.
- Private or personal data: real names and emails outside intended
  authorship metadata, internal hostnames or URLs, customer or user data,
  anything pasted from a private conversation.
- Content that belongs outside version control: session transcripts, eval
  fixtures and outputs, personal harness configuration
  (`.claude/settings.local.json` and similar; the shared
  `.claude/settings.json` allowlist is tracked on purpose), scratch files,
  large generated artifacts.

The pre-commit hooks (gitleaks, detect-secrets) catch known token patterns;
this pass hunts what pattern matching misses — novel token formats, secrets
inside prose or test data, private information that is not a credential.
When the diff carries risky material, run the `sensitivity-check` skill over
it for the deep scan. Anything found is the most severe finding in the
report, and if it is already in a pushed commit, say plainly that removal
requires a history rewrite plus rotation of the exposed secret — a
follow-up commit deletes nothing.

## Quality gradient

- `skills/<catalog>/` is the published product: users copy these
  directories out of the repository, where they lose access to everything
  else. Highest scrutiny; installed-behavior defects live only here.
- Everything else (`.agents/`, `.github/`, `scripts/`, `justfile`, READMEs,
  `ARCHITECTURE.md`) is repository harness, not published content. Review
  it for synchronization, not polish: the finding shape is desync — a pair
  in `.agents/knowledge/harness-maintenance.md` changed on one side only,
  a documented command that no longer runs, a pointer to a moved file,
  guidance drifted from the implementation it describes. Stale harness is
  entropy that misleads every later agent; report it, capped at
  recommended severity.
- `openspec/` is the specification layer. A pull request that changes a
  public skill's behavior without a delta spec, or that is ready with a
  Validation section naming no scenario, is blocking. A repository change
  needs its `skip_specs` record instead of a delta spec
  (`spec-workflow.md ## Domains`). A hand edit to a generated `openspec-*`
  skill is a finding (`spec-workflow.md ## Artifacts`).
- Never block a ready pull request for being unarchived: during its second
  deliberation that is the expected state, and `checks / spec` blocks the
  merge until the freeze (`spec-workflow.md ## Archive executor and the
  freeze`).

## Scripts: the threat model

Skill scripts are one-off developer tooling run locally by the coding
agent, on inputs the agent itself supplies. They are not services and have
no untrusted callers — an agent that wanted to act unsafely could act
directly, so a script adds no attack surface. Do not report input
sanitization, injection, or hardening findings against them as security
issues. What does matter:

- Protecting the human from agent mistakes: destructive or remote-writing
  paths default to dry-run and require an explicit `--apply`, `--delete`,
  or `--confirm`.
- Secrets are never printed or logged.
- The script contract in `skill-quality.md ## Scripts` holds.

The rules each kind of script is reviewed against have one owner:

- This repository's code — `scripts/`, workflow shell, the project
  skills' scripts — `.agents/knowledge/management-code.md`; a privileged
  job handles content the request's author wrote, so read its
  `## Workflow security` before reviewing one.
- The Python scripts and script assets public skills ship —
  `skill-quality.md ## Script assets`.

## Do not report what machines catch

`just check` already enforces frontmatter limits, name/directory match,
symlinks, marketplace sync, catalog files, the
`## Harness Methodology` copy between `core/meta-harness` and
`meta-harness-architecture`, label and form consistency, knowledge-file
pointers, the parsed workflow rules (permissions, pins, timeouts,
privileged triggers), line and table budgets, and strict OpenSpec
validation (`scripts/validate_skills.ts`, `scripts/validate_harness.ts`,
`just spec-validate`). Run it (or trust its CI result) and spend review
attention only on what it cannot see. Its self-containment check catches
only Markdown links that escape a skill's root and install commands;
review every other reference yourself — paths in prose or code, skill
names outside the dependency range, skills of other repositories. A name
that departs from the recommended `[<prefix>-]<body>[-<suffix>]` shape is
not a defect either: the shape is advisory and the chosen name stands.

## Repository-specific checks

- **Load-graph consistency.** A skill is read piecewise: description →
  SKILL.md → the one reference whose stated condition fired → asset. A
  contradiction between pieces that load together is blocking; wording
  drift between pieces that never co-load is entropy, not a defect.
- **Assets outlive their builders.** For disposable builders (`meta-`,
  `scaffold-`), the deposited assets survive in target projects after the
  builder is deleted, so a defect in `assets/` outranks the same defect in
  SKILL.md. Check that content meant to survive adaptation never sits
  inside placeholder markup a header orders deleted.
- **Contract flow is one-way.** Contract builders design in the
  platform-neutral model and deposit in the platform's vocabulary;
  platform builders implement settled decisions and never re-decide or
  re-ask them, and stay paradigm-neutral — a specification word, artifact,
  or gate inside a platform builder is a finding; paradigm builders
  (`meta-spec-workflow`) deposit their contract first and shape the
  platform base's registered extension slots in a second phase, so their
  slot names and locations must match the bases' slot registers. This
  repository's own harness is GitHub-specific by decision
  (`.agents/knowledge/github-workflow.md`), so GitHub vocabulary in
  `.agents/knowledge/` is not a finding.
- **Cross-skill references by name that hand off through
  `ryan-minato-skills-installing` are the required pattern**, not a defect;
  the defects are those `skill-quality.md ## Self-containment and
  dependencies` names.

## Before suggesting an edit

Check the constraint surface first: the limits in
`skill-quality.md ## Spec limits (mechanically enforced)`; English only;
the disposable marker of `meta-`/`scaffold-` descriptions
(`ARCHITECTURE.md`), which must not change; the `README.zh.md` twin of
every `README.md` (`AGENTS.md`). Trigger phrases in descriptions are
load-bearing — never trade one away for style.

## Verifying findings

Every reported finding names its failure script: which reader (agent tier
or human), on what input, takes which wrong action — a finding without one
is a note or is dropped. Verify by tracing the load path or running the
named check: `just check-skill <dir>` for spec claims, a file's own grep
command for vocabulary claims, the resolved symlink chain for structure
claims. For trigger-surface claims, read the behavioral-test evidence
recorded in the PR before re-arguing from intuition — a description that
tested clean at its stated tier is not a finding.
