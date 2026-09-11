# The Run Record

Read when a run is cited as evidence in a pull or merge request, a report,
or a promotion, or when a committed run record is required. This file is
the canonical field list; the manifest module and any committed record
follow it.

## When a record is committed

The tracker holds the interactive view of every run. A **committed run
record** is written into the repository only for a run that carries
weight beyond the task: a promoted model, a result a report cites, a
baseline other tasks compare against. It is what a future agent can trust
from the checkout alone when the tracker is gone. Its shape is the
skill's `assets/run-record.md`; fill every line or delete it, never leave
a slot.

## Fields

| Field | Immutable form | Why |
|---|---|---|
| Run id | the minted id | one execution; not the commit |
| Status, start, end | timestamps, final status | reproducibility of the timeline |
| Source snapshot | repository identity + commit; dirty flag; patch hash when dirty | what code ran |
| Snapshot retention | the tag or ref that keeps the commit reachable | the record must not dangle |
| Resolved configuration | path of the resolved dump and its hash | what choices were in effect |
| Environment | image digest, or lock digest plus interpreter version | what software ran |
| Host | GPU model and count, driver, runtime versions, node | required for any performance claim |
| Inputs | one line per input: name, kind, identity (revision, checksum, version id), preprocessing version | what data and models were used |
| Randomness | seed(s), determinism flags | what "same seed" means |
| Parent run | the run id resumed or fine-tuned from | lineage |
| Evaluation identity | the evaluation set's identity and the evaluation code's commit | what the metrics mean |
| Metrics | name, definition, value, and the threshold agreed before the run when one exists | the evidence |
| Artifacts | governed location plus immutable identifier per artifact | where the outputs are |
| Tracker run | the run's URL or id | the interactive view |
| Decision | promoted, baseline, rejected, superseded by run id | why the record exists |

## Rules

- A field whose value could resolve differently later is not filled in
  yet: resolve it first.
- Never commit weights, datasets, secrets, or raw samples to make them
  discoverable; record governed locations and identifiers.
- A record is never edited after the run it describes ends. A later
  finding becomes a note appended with its own date, or a new record.
- When the project's platform builder deposited a record template, that
  template carries these fields in the platform's vocabulary; do not
  keep two shapes.
