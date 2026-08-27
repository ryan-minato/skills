# Shared Code in an Experiment

Read when two entry scripts need the same code.

- Create one package named after the repository, at the root
  (`<project_name>/`), and import it from the entry scripts. No `src/`
  layout — the flat form keeps `python train.py` working from the root
  with nothing to install.
- The extraction test: move code into the package only when the copies
  must stay logically consistent — the same preprocessing in `train.py`
  and `eval.py` is the canonical case, because a drifted copy silently
  corrupts the comparison. Repetition alone is not a reason; tolerate
  copy-paste that keeps each script readable top to bottom without
  jumping between files.
- Docstrings in the package: interfaces the entry scripts call document
  their arguments, return values, and raised errors; private helpers get
  a one-line purpose. In the scripts themselves, comment long logic
  passages, never obvious single statements — prefer names that make the
  comment unnecessary.
- When a shared component has a contract whose silent failure would
  corrupt the experiment (a metric, a data filter, a sampler), protect
  exactly that contract with a pytest test under `tests/`. Anything
  needing a GPU, a download, or minutes of runtime is marked
  `@pytest.mark.slow` and runs only by hand.
