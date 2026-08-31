# Colab Environment Baseline and Debugging

Read before adding, pinning, or upgrading any dependency, and whenever
behavior in the local container differs from real Colab.

## Dependency policy

The Colab preinstalled environment is the baseline. Write against what is
preinstalled as far as possible and install only what is missing, in the
notebook's first code cell (the doctrine reference owns the cell rules).
Upgrading a preinstalled package is allowed when genuinely necessary, but
the environment is a tightly coupled set — one pip upgrade can cascade
through shared dependencies and break unrelated packages — so prefer
adapting the code to the preinstalled version over upgrading it.

## Ground truth: backend-info

The first-party record of the real Colab environment is
https://github.com/googlecolab/backend-info. Fetch its text files raw:

    https://raw.githubusercontent.com/googlecolab/backend-info/main/<file>

The files cover OS info, apt package lists, and pip freezes, each with GPU
and TPU variants alongside the CPU default (names follow `os-info*.txt`,
`apt-list*.txt`, `pip-freeze*.txt`). If a guessed name returns 404, list the
repository tree instead of retrying blind. The repo's history keeps
snapshots of past runtime versions — pin to a commit when the project must
match an older runtime.

## Debug flow

1. Reproduce the difference in the local runtime container.
2. Fetch the matching backend-info listing and diff it against the container
   (`pip freeze`, `apt list --installed`, Python and OS versions).
3. Decide: add the missing package, pin to the Colab version, or — last
   resort — upgrade. Re-run the notebook in the container, then on real
   Colab.
4. Record the decision and its reason in the target `AGENTS.md` so future
   work does not re-litigate it.
