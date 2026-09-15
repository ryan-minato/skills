## Context

See proposal.md. The five skills live in `skills/machine-learning/` under
the catalog's `CONTEXT.md`: assets are drop-in modules or section
skeletons that work as written, PyTorch is the illustrative default with
framework-specific hooks behind a load condition, and no skill depends on
another by name. Mirrors that bind this change:
`experiment-provenance/assets/run_manifest.py` is byte-identical to
`scaffold-ml/assets/run_manifest.py`, and the `##` headings of
`research-workflow/assets/research-spec.md` equal those of
`scaffold-ml/assets/openspec/research-task/templates/research.md`; both
are register rows in `.agents/knowledge/harness-maintenance.md`. Size
limits: bodies under 500 lines (all five are). The asset ceiling this
change applies is the one `ml-standard-alignment` records for the
scaffold.

## Placement

| Requirement | File and section | Load trigger (references only) |
|---|---|---|
| Behavior: Run identity is composed of four immutable parts plus a run id (modified) | `assets/run_manifest.py` `start_manifest` (a declared-but-empty `IMAGE_DIGEST` appends `no_image_digest`; `sha256_file` public) and its docstring's degraded list; `SKILL.md` `## The manifest` paragraph points at `references/run-record.md` for the canonical field list instead of restating it | "Read `references/run-record.md` when writing or reviewing a run record." (existing) |
| experiment-provenance structure | `SKILL.md` `## The run equation` (the resolve-later validator question), `## Where the facts live` (tracker aliases move), `## Gotchas` (two bullets) | — |
| research-workflow structure | `assets/research-spec.md` (bare skeleton; headings unchanged), `SKILL.md` `## Gotchas` (one bullet) | — |
| experiment-code-conventions structure | `references/config-surface.md` (mechanism block, one sentence), `references/vendoring-research-code.md` (procedure steps 1–2, modes table; no Gotchas), `references/tensor-tests-and-docs.md` (the positive rule only), `SKILL.md` `## Gotchas` (three bullets) | existing load sentences unchanged |
| training-instrumentation structure | `SKILL.md` `## Health measurements in the loop` (order sentence and the asset pointer), `references/model-health-metrics.md` (one sentence in place of the block), `## Gotchas` (one bullet) | existing load sentences unchanged |
| training-diagnosis structure | `SKILL.md` `## The evidence chain` (pointer and the two-cause rule), `## Numerical instability` (one resident rule and the symptom-index pointer), `## Gotchas` (four bullets) | "Read `references/numeric-instability.md` when the loss spikes or goes NaN" (existing symptom-index row) |

## Dependencies and handoffs

Unchanged: no skill names another; the roles named in handoffs stay.

## External impact

- `skills/scaffold/scaffold-ml/assets/run_manifest.py`: changed in the
  same commit; proof `cmp` of the two copies is silent.
- `skills/scaffold/scaffold-ml/assets/openspec/research-task/templates/research.md`:
  proof the `## ` heading lists of the two research-spec files are
  identical.
- README pairs, symlinks, `marketplace.json`: unchanged; proof
  `just gen-marketplace` then `git diff --exit-code .claude-plugin/marketplace.json`.

## Decisions

- **A declared-but-empty digest is degraded; an absent variable is not**
  (serves Run identity): the presence of the variable is the container
  signal (the Compose asset always declares it), and a container run
  without its image digest has lost its environment identity even when a
  lock file is visible through the source mount. Alternative rejected:
  change only the Compose comment, which leaves a container run citable
  as complete.
- **One home per fact, chosen by the branch where an agent gets it
  wrong** (serves the structure bullets): seed-versus-determinism stays
  with provenance (it records the determinism flags), utilization-
  versus-efficiency with diagnosis (where a utilization number is read),
  allocator blindness with instrumentation (where a second memory source
  is designed in), mocking with code conventions. Alternative rejected:
  keep every cross-skill duplicate, which costs every invocation of both
  skills.
- **Skeleton assets carry slots, not rules** (serves research-workflow
  structure): the field rules already live in `SKILL.md`'s section table
  and the fields reference; a skeleton whose placeholders are instructions
  is guidance in disguise and drifts from the reference.
- **Code blocks illustrate; assets are copied** (serves instrumentation
  and code-conventions structure): the call site is the asset's
  docstring, the Adam measurement is the asset's function, and the
  configuration reference shows the mechanism, not a schema with
  defaults a project would inherit.

## Risks / Trade-offs

- [A moved fact is lost on the branch that needed it] → the readback
  quotes every remaining scenario of the five domains against the
  finished skills and checks each deleted Gotchas bullet against the
  section named for it.
- [The mirror drifts] → both copies change in one commit; `cmp` is a
  task.

## Verification plan

Solver tier: none (no Trigger or description change). Observation and
isolation: not applicable.

| Scenario | Case (prompt or task) | Rubric and critical failures | Pass threshold | Solver tier | Observation | Isolation |
|---|---|---|---|---|---|---|
| experiment-provenance: Empty image digest declared | scratch fixture with the module: `IMAGE_DIGEST= python -c "…start_manifest(…)"` and the same with `IMAGE_DIGEST=sha256:abc` and with the variable unset | `no_image_digest` under degraded and on stderr only in the first case (critical) | 3/3 | — | command output | scratch directory outside the repository |

Readback (one clean-context subagent reads the five finished skills and
the main specs, quotes the passage behind every scenario's THEN, reports
PRESENT / WEAK / GAP with no GAP allowed, and for every Gotchas bullet
this change removes quotes the section that now carries the fact).

Script and tool harnesses:
- `cmp skills/machine-learning/experiment-provenance/assets/run_manifest.py skills/scaffold/scaffold-ml/assets/run_manifest.py`; `diff` of the two research-spec `## ` heading lists; `python3 -m py_compile`, `ruff check`, `ruff format --check` on the changed module; `just check-skill` on the five skills; `just check`.

Skipped:
- Every scenario other than the new one: no behavior changed; covered by the readback.

## Open Questions

None.
