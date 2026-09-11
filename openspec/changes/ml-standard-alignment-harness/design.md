## Context

See proposal.md. `skills/scaffold/CONTEXT.md` `## Dependencies` grants
`meta` and `core` and says nothing about durable topic skills of another
catalog; `.agents/knowledge/skill-quality.md` already defines the
optional-handoff-by-role rule, so the catalog sentence points at it rather
than restating it. `.agents/knowledge/harness-maintenance.md` holds the
register of mirrored pairs with trigger and owner columns; two new pairs
appear with the skill change. `.claude-plugin/marketplace.json`'s
`scaffold` description is a human-owned field the generator never
touches.

## Placement

| What Changes bullet | File | Check that proves it |
|---|---|---|
| Cross-catalog naming rule | `skills/scaffold/CONTEXT.md` `## Dependencies`, one sentence after the existing grant bullets | read-through; `just validate` (`check_catalogs`) |
| Register rows | `.agents/knowledge/harness-maintenance.md`, two rows in the mirrored-pairs table: the manifest module and the research-spec headings | `diff skills/machine-learning/experiment-provenance/assets/run_manifest.py skills/scaffold/scaffold-ml/assets/run_manifest.py` shows only the docstring's first line; `grep -c '^## ' skills/machine-learning/research-workflow/assets/research-spec.md` equals the template's heading count; `just validate` (`validate_harness.py` register checks) |
| Marketplace description | `.claude-plugin/marketplace.json` `scaffold` plugin `description` | `just gen-marketplace` then `git diff --exit-code` on the `skills` arrays (only the description line may differ); `just validate` |

## External impact

- The skill change `ml-standard-alignment` on the same branch creates
  both mirrored files; the register rows land after them.
- No validator, workflow, ruleset, `meta` file, or `machine-learning`
  file changes; proof `git diff --stat origin/main...HEAD -- scripts .github skills/meta skills/machine-learning` empty.

## Decisions

- **Register rows, not a mechanical check** (serves Register rows): the
  manifest copy is the only byte-level mirror and it is small; a
  `validate_harness.py` check is added only if the mirror drifts once.
- **One sentence in the catalog context pointing at the quality file**:
  the rule already exists repo-wide; the catalog says only that its
  scaffolds use it for the durable topic skills.

## Risks / Trade-offs

- [The manifest copies drift silently] → the register row names the
  `diff` command and the trigger ("the durable skill's module changes");
  the reviewer's harness-sync check reads the register.

## Verification plan

- Cross-catalog rule: read-through of `skills/scaffold/CONTEXT.md`;
  `just validate` green.
- Register rows: the two commands in the placement table run green;
  `just validate` green.
- Marketplace: `just gen-marketplace` then `git diff --exit-code
  .claude-plugin/marketplace.json` after the description edit is
  committed; `just validate` green.
- `just check` at the end.

## Open Questions

None.
