# <project name>

<Two or three sentences: what the notebook(s) teach or demonstrate, and who
opens and runs them in Colab.>

## Files

- `notebook.ipynb` — the product; local mirror of the remote Colab notebook
  <single-notebook projects; otherwise list each root .ipynb with its remote
  counterpart>
- `justfile` — local runtime and validation commands
- `.devcontainer/` — official Colab runtime image; the local first-pass
  validation environment

## Mirror rule

Every root `.ipynb` mirrors a remote Colab notebook. Edit the local file,
push through colab-mcp <or `colab upload`>; after any remote edit, pull it
back before touching the local file again. A change on either side is
synchronized immediately — the two sides never drift.

## Commands

| Command | Does |
|---|---|
| `just runtime` | start a local Colab runtime the web UI can connect to |
| `just validate <nb>` | execute a notebook headlessly in the runtime image |
| `just lint` | ruff over any helper scripts |

## Dependencies

- The Colab preinstalled environment is the baseline; notebooks install only
  what is missing, in the first code cell.
- The real environment is recorded at
  https://github.com/googlecolab/backend-info (raw txt files); consult it
  before adding or pinning anything.
- Avoid upgrading preinstalled packages — upgrades cascade through the
  coupled dependency set. <record agreed exceptions here>

## Notebook rules

- A notebook is first a document for human reading, second a runtime.
- All installs in the first code cell (`%pip install -q` or `%%capture`).
- Re-import what a cell uses in that cell; cells stay movable.
- Parameters near first use as `# @param` controls of the precise type;
  never ipywidgets. `# @title` and form comments at the top of the cell.
- Atomic cells — one thing each; markdown explains why, not what.
- Imperative code; reuse belongs in Python libraries, not notebooks.
- Spec first: markdown structure plus TODO-comment code cells (what the code
  must do, how to verify), then implement TODO by TODO.

## Validation

1. `just validate <nb>` — preliminary, in the local runtime container.
2. Run on real Colab via colab-mcp — the confirmation that counts.
