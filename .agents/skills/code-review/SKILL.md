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
   the working tree. `--comment`: post findings as inline PR comments —
   that is a remote write, so it needs the explicit user authorization the
   `change-workflow` skill requires, and the exact comment payloads are
   drafted and reviewed before publishing.
5. A deep multi-agent cloud review ("ultra") is a host feature, not part of
   this skill; when it is unavailable, run at maximum effort locally and
   say that is what happened.

## Severity ladder

1. **Critical — leaked secrets, private data, or content that must not be
   committed.** Always the top of the report, at every effort level.
2. **Blocking — wrong installed behavior**: an agent following a published
   skill's text takes a wrong or unsafe action; agent authority widened by
   default (ready/review/merge granted on green checks, a weakened
   no-self-escalation rule); platform vocabulary inside a platform-neutral
   contract layer; a contradiction between pieces that load together; a
   path reference escaping a public skill's directory; a hard dependency
   outside the catalog's granted range, or a recommendation of another
   repository's skill.
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
  fixtures and outputs, local harness configuration (`.claude/settings.json`
  and similar), scratch files, large generated artifacts.

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
  it for synchronization, not polish: the finding shape is desync — one
  side of an `AGENTS.md` Keep In Sync pair changed without the other, a
  documented command that no longer runs, a pointer to a moved file,
  guidance drifted from the implementation it describes. Stale harness is
  entropy that misleads every later agent; report it, capped at
  recommended severity.

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
- The repository script contract holds: non-interactive, `--help`, data to
  stdout and diagnostics to stderr, exit codes 0/1/2, idempotent.

Scripts copied into target projects for CI follow the same calibration:
their input is repository content, not adversarial traffic.

## Do not report what machines catch

`just check` already enforces frontmatter limits, name/directory match,
symlinks, marketplace sync, path self-containment, catalog files, and
byte-identical script copies. Run it (or trust its CI result) and spend
review attention only on what it cannot see. A name that departs from the
recommended `[<prefix>-]<body>[-<suffix>]` shape is not a defect either: the
shape is advisory and the chosen name stands.

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
- **Contract flow is one-way.** Contract builders stay platform-neutral;
  platform builders map settled contract decisions and never re-decide or
  re-ask them.
- **Cross-skill references by name that hand off through
  `ryan-minato-skills-installing` are the required pattern**, not a defect.
  The defects are a hardcoded install command in a public skill; a
  dependency the skill cannot work without on anything outside its allowed
  range (`core` always; other catalogs only by the catalog `CONTEXT.md`
  grant); a recommendation of a skill from another repository; and a path
  reference across skill roots.

## Before suggesting an edit

Check the constraint surface first: descriptions warn above 900 characters
(error at 1024); bodies warn above 500 lines; all content is English;
`meta-`/`scaffold-` descriptions open with the shared disposable marker
(`Disposable builder skill (delete after the harness is built):`), which
must not change; every `README.md` change needs its content-identical
`README.zh.md` twin; trigger phrases in descriptions are load-bearing —
never trade one away for style.

## Verifying findings

Every reported finding names its failure script: which reader (agent tier
or human), on what input, takes which wrong action — a finding without one
is a note or is dropped. Verify by tracing the load path or running the
named check: `just check-skill <dir>` for spec claims, a file's own grep
command for vocabulary claims, the resolved symlink chain for structure
claims. For trigger-surface claims, read the behavioral-test evidence
recorded in the PR before re-arguing from intuition — a description that
tested clean at its stated tier is not a finding.
