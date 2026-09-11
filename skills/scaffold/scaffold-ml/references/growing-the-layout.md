# Growing the Layout

Read when two entry scripts need the same code or the project grows a
package, tests, or a docs directory.

## The extraction rule

Move code into a shared package only when the copies must stay
logically consistent — the same preprocessing in `train.py` and
`eval.py` is the canonical case, because a drifted copy silently
corrupts the comparison. Repetition alone is not a reason: tolerate
copy-paste that keeps each script readable top to bottom. Abstract the
stable mechanism; duplicate the unstable policy.

## The package

One package named after the repository, at the root
(`<project_name>/`), imported by the entry scripts with absolute imports;
no `src/` layout, so `python train.py` works from the root with nothing
to install. Interfaces the entry scripts call get docstrings (arguments,
returns, raised errors, and for tensors the shape, dtype, device, and
mask conventions); private helpers get a one-line purpose; comments in
the scripts explain long logic passages, never single statements.

## Tests

`tests/` protects contracts whose silent failure would corrupt the
experiment: a metric, a data filter, a sampler, a custom layer, the
integration assumptions around a library (shapes, masks, padding, label
alignment, reductions). Small real tensors on the CPU; mocking tensor
operations tests the mock. Anything needing an accelerator carries
`@pytest.mark.gpu` and runs only through `just test-gpu`, failing
without hardware; anything needing a download or minutes carries
`@pytest.mark.slow` and runs by hand.

## Data directories

`data/raw/` holds the immutable inputs the manifest identifies; derived
data lands under `data/interim/` and `data/processed/` and is
regenerable; nothing under `data/` is committed. `docs/data.md` records
each source, its identity, and how to fetch it on a fresh machine.
