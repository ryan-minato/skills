# Harness Maintenance

Read this when a file that another file mirrors changes, on the first pull
request of each quarter, and whenever a catalog is added or removed. It
holds the register of copied facts, the layer calibration, and the
periodic entropy review.

## Synchronization register

Mechanically checked pairs live in `scripts/validate_skills.ts` and
`scripts/validate_harness.ts` (run by `just validate`, pre-commit, and
`checks / quality`); each script's header lists its checks. The rows below
are the pairs no script checks, or checks only in part.

| Source | Mirror | Owner | Check or trigger |
|---|---|---|---|
| any `README.md` | its `README.zh.md`, content-identical | author | same pull request |
| a catalog `CONTEXT.md` `## Dependencies` grant | the grants sentence in `skill-quality.md`; the catalog bullet in `ARCHITECTURE.md` | author | same pull request |
| the handoff template in `skill-quality.md` | the description triggers of `skills/core/ryan-minato-skills-installing` | author | same pull request |
| NGC access in `skills/meta/meta-gpu-container/scripts/` | `skills/core/devcontainer-setup/scripts/list_sources.py` (same endpoint, separate code) | author | a registry change breaks one |
| the agentskills MCP endpoint | `.mcp.json`, `.codex/config.toml`, `.devcontainer/devcontainer.json` | maintainer | the endpoint moves |
| `openspec_version` in the `justfile` | `.agents/skills/openspec-*` (`just spec-sync`); the artifact table in `spec-workflow.md` | maintainer | `tools` check; `checks / spec` |
| `scripts/spec_changes.ts` subcommands, flags | `spec-check`, `spec-changes` recipes; `spec-workflow.md` `## Archive executor and the freeze` | author | a named flag changes |
| `## Extension slots` of `meta-github-workflow`, `meta-gitlab-workflow` | slot tables and asset sections of `skills/meta/meta-spec-workflow/` | author | a slot is added, renamed, or moved |
| `## Deliverable and pre-authorized actions` of the proposal template | `schema.yaml`; `spec-workflow.md`; the PR template; `agent-authority.md` | author | the section is renamed or moved |
| `.github/PULL_REQUEST_TEMPLATE.md` sections and reserved lines | the payload and ready steps of the `change-workflow` skill; `spec-workflow.md` | author | the template changes |
| a workflow's required-check name | the `github-checks.md` table (`checks` check); the ruleset row in `github-settings.md` | maintainer | a job rename |
| a third-party `uses:` in a workflow or `.github/actions/*` | the Actions allowlist row in `github-settings.md` | author | an action is added or dropped |
| a privileged trigger in a workflow | the `## Privileged workflows` line in `github-checks.md` (`workflows` check) | author | a workflow adopts or drops one |
| a workflow job holding a write token | the write-token script list in `management-code.md` `## Workflow security` | author | a job gains a write permission |
| an `allow` or `deny` entry in `.claude/settings.json` | the same rule in `.codex/rules/default.rules`; `agent-authority.md` `## Permission entries` | author | an entry is added or removed |
| a remote setting | its row in `github-settings.md` | maintainer | every applied setting, read back |
| `skills/machine-learning/experiment-provenance/assets/run_manifest.py` | `skills/scaffold/scaffold-ml/assets/run_manifest.py`, byte-identical | author | `cmp` them when either changes |
| `##` headings of `research-workflow/assets/research-spec.md` | `skills/scaffold/scaffold-ml/assets/openspec/research-task/templates/research.md` | author | a research-spec field changes |
| a catalog added or removed | catalog scaffold, `ARCHITECTURE.md` catalogs, `just gen-marketplace`, root README pair | author | same pull request; validators check part |
| a catalog added or removed | `catalog/*` in `labels.json`, issue-form Catalog options, prefixes in `validate_skills.ts` | author, maintainer | validators; removed label deleted by hand |

`scripts/spec_changes.ts` and `.github/workflows/spec-labels.yml`
began from the `openspec-workflow` skill's assets; they are this
repository's own port and bind nothing. A change to the assets needs no
change here, and the reverse.

## Layer calibration

Read before changing the thickness of any layer. Rated 2026-10-20 for the
rebuilt harness; the variables: error cost high (every skill is installed
by other projects), maintenance horizon indefinite, isolation a dev
container with no credential beyond the forwarded `gh` login, autonomy
attended at H1, team one maintainer plus agents.

| Layer | Rating | Variable and reason |
|---|---|---|
| Environment and sandbox | medium | isolation: container with the latest tools, no mounts, `gh` only; `checks / devcontainer` builds it |
| Knowledge and routing | thick | horizon: one owner per rule, entrypoint in budget; `pointers` and `budget` checks |
| Hard gates | thick | error cost: pre-commit, tested management scripts, parsed `workflows` check, commit-range scan, three required checks |
| Authorization boundary | medium | autonomy attended: H1, the agents' classifiers, deny entries, approval label kept by automation; thick only if a run is unattended |
| Specification and completion criteria | thick | error cost and horizon: OpenSpec strict, two deliberations, the approval label recomputed |
| Entropy management | medium | horizon: this register, two validators, budget check; duplication is found only by the review below |

## Periodic entropy review

Cadence: quarterly, on the first pull request of the quarter, and after
any catalog change.

1. Run `just check`; every command `AGENTS.md`, `ARCHITECTURE.md`, and
   the project skills name must still run.
2. Read `AGENTS.md` for stale paths and for an always-on rule missing its
   owner; the `budget` check holds its length. Every knowledge file must
   have a pointer whose trigger can fire.
3. Duplication grep: for each rule with an owner (the owner map in the
   archived `repo-harness-alignment` design), grep its key phrase — the
   freeze, draft timing, the archive executor, `skip_specs`, the publish
   gate, the merge method, branch naming, README pairs, tool pins — across
   `AGENTS.md`, `ARCHITECTURE.md`, `.agents/`, `.github/`, and
   `openspec/config.yaml` and the schema. A full statement outside its
   owner becomes a pointer.
4. Compare each knowledge file's "Update this file when" list with what
   changed since the last run; update or delete stale content.
5. Settings readback: read back every row of `github-settings.md` with
   its command and record it as its `## Applying and reading back` says.
6. Confirm Dependabot's `github-actions`, `devcontainers`, and
   `pre-commit` updates are merged or declined; none sits open.
7. Check the "not used" triggers in `github-workflow.md`; enable only what
   fired.
8. Ask of each constraint whether it caught anything since the last run;
   one that catches nothing is a thinning candidate. Look for project
   skills whose trigger never fired. Re-read the rules held by review
   alone (the non-goals of the `repo-harness-alignment` proposal) and
   whether one now earns a check.
9. Record findings as keep / update / remove, apply the unambiguous ones,
   and update "Last run" below in the same pull request.

Last run: 2026-10-20, the rebuild `repo-harness-alignment`, as the
quarter's review. Update: every duplicated rule given one owner, stale
passages removed, wide tables rebuilt. Settings: four rows differ, applied
by the maintainer after the archive. Hook revisions: Dependabot's
`pre-commit` ecosystem added.
