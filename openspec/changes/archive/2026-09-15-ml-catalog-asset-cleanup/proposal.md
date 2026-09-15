## Why

The review of `scaffold-ml`'s assets on this branch (`ml-standard-alignment`)
set an asset ceiling — an asset is copied as is or filled at its
placeholders, guidance is prose, a branch's code fragment is a fenced block
in that branch's reference, and a Gotchas section holds only what no step
or reference states. The five `machine-learning` skills that the scaffold
installs were written before that ceiling: their Gotchas sections restate
rules their own sections carry, several facts are stated in two or three
skills, a research-spec skeleton carries its field rules as placeholders,
a training-loop call site and an optimizer-health function each live in
two places, and a configuration reference prescribes a dataclass schema
with concrete defaults. The manifest module the scaffold mirrors also
gains the review's correction — a declared-but-empty `IMAGE_DIGEST` marks
the run degraded — which the durable skill's copy must share. Now, on the
same branch, so the catalog and the scaffold ship one standard.

## What Changes

- `experiment-provenance`: `assets/run_manifest.py` marks a run whose
  environment declares `IMAGE_DIGEST` but leaves it empty as degraded
  (`no_image_digest`) instead of silently falling back to the lock hash,
  and exposes its chunked file hash as `sha256_file` for callers that
  hash large artifacts; the copy stays byte-identical to the scaffold's.
  Structure only, no observable behavior change: the manifest field table
  leaves `SKILL.md` for the reference that is its canonical home; the
  Gotchas keep the two facts stated nowhere else (a seed is not
  determinism; a presigned URL is a credential and a mutable reference)
  and the rest move into the sections that own them.
- `research-workflow` (no observable behavior change): the research-spec
  skeleton becomes a bare skeleton — headings, a status line, one slot
  per required section — with the field rules left in `SKILL.md` and
  `references/research-spec-fields.md`; the Gotchas keep the one fact
  stated nowhere else (a research spec is not a requirements spec).
- `experiment-code-conventions` (no observable behavior change): the
  configuration reference's prescriptive dataclass schema becomes the
  three-line merge mechanism plus one sentence; the Gotchas keep the three
  detectors (`_target_`/`class:`, `skipif(no GPU)`, mocking tensors) and
  the vendoring reference's Gotchas dissolve into its procedure and modes
  table.
- `training-instrumentation` (no observable behavior change): the PyTorch
  call-site block leaves `SKILL.md` for the asset's own docstring, the
  duplicated `adam_health` block leaves the model-health reference, and
  the Gotchas keep the allocator-statistics fact only.
- `training-diagnosis` (no observable behavior change): the evidence-chain
  slot list and the numerical-instability section leave `SKILL.md` for the
  asset and the reference that hold them, keeping one resident rule (no
  learning-rate change before the replay); the Gotchas keep the four facts
  stated nowhere else in the skill.

## Skills touched

- `machine-learning/experiment-provenance` (modified): the run-identity
  requirement gains the declared-but-empty image digest scenario.

## Installed behavior

`experiment-provenance`: a run launched from a Compose service whose
`IMAGE_DIGEST` was never exported is now cited as degraded, and a caller
can hash a checkpoint through the module → `fix`. The other four skills
behave the same; their edits restructure where facts live → `refactor`.

## Impact

- `skills/scaffold/scaffold-ml/assets/run_manifest.py`: the byte-identical
  mirror, changed in the same commit (register row in
  `.agents/knowledge/harness-maintenance.md`).
- `skills/scaffold/scaffold-ml/assets/openspec/research-task/templates/research.md`:
  keeps the same `##` headings as the trimmed research-spec skeleton
  (register row).
- README pair rows, symlinks, `marketplace.json`, catalog `CONTEXT.md`:
  unchanged (no description moves, no directory added or removed).

## Non-goals

- Any new capability in the five skills, any description change, any
  change to their scripts.
- The hyperparameter / search-variable vocabulary each skill restates:
  the skills are self-contained and each states it for its own branch.
- The signal-catalog tables of `training-instrumentation` and the
  research-spec section table of `research-workflow`: entry summaries
  with reference detail behind them are a valid layering.

## Tracked work

No issue: raised by the review of `ml-standard-alignment` in conversation.
