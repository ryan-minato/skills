# Notebook Doctrine

Read before writing or reviewing any notebook cell. A good Colab notebook is
first a document for human reading, second a runtime environment; every rule
below serves that order.

## Cell rules

- **Installs first, together, quietly.** All dependency installs go in the
  very first code cell — a reader learns the notebook's requirements in one
  place, and nothing installs later. Use `%pip install -q` for a few
  packages; start the cell with `%%capture` when install output would drown
  the page.
- **Atomic code cells.** One cell does exactly one thing. A markdown cell
  may introduce a group of related cells; then give each member a `# @title`
  line so the group reads as labeled steps. Title and form-control comments
  sit at the top of the cell.
- **Re-import per cell.** When a cell uses an external module, import it in
  that cell even if an earlier cell already did. Repeated imports cost
  nothing at Colab scale, and every cell stays movable without breaking.
- **Parameters near first use.** Define a parameter immediately before — or
  in — the cell that first consumes it, as a `# @param` control of the
  precise type: a slider for a bounded number, a dropdown for an
  enumeration, never a bare string for everything. A reader-facing notebook
  almost always has at least one knob worth exposing (a split ratio, a
  hyperparameter, a sample size); if no cell carries a form control, check
  whether a hard-coded constant should be one. Read
  [colab-forms.md](colab-forms.md) for the syntax.
- **Helpers where they are needed.** Never collect utility functions into
  one toolbox cell. A lightly used helper is defined right before its first
  use; a helper that carries important logic gets its own cell, introduced
  by markdown and demonstrated with an example.
- **Long cells are the fallback.** When a cell genuinely cannot be split,
  explain its phases with in-code comments — but atomic cells come first.

## Prose rules

- Markdown cells carry the point, the principle, and the why — never a
  play-by-play of the code. The reader is not a fool: they can read code,
  and narrating "now we call fit()" wastes their attention.
- Read [colab-markdown.md](colab-markdown.md) when writing markdown cells
  for what Colab actually renders.

## Style

- Imperative code, at most lightly procedural. Classes, factories, and
  design patterns are wasted effort in a notebook; logic that deserves
  abstraction deserves to be a Python library the notebook imports.

## Spec-driven notebook development

Design the document before the code:

1. Write the complete markdown structure — headings, explanations, the
   argument the notebook makes — as markdown cells.
2. Where code will go, leave a code cell holding only a TODO comment that
   states what the code must do and how to verify it:

       # TODO: load the dataset from <url>; verify: prints (n_rows, n_cols)

3. Implement the TODOs one by one, checking each against its stated
   verification before moving on.

The spec pass keeps the prose in charge of the notebook and turns
implementation into a sequence of small, individually verifiable steps.

## Shape example

A markdown cell heading a two-cell group, each member atomic, titled, and
self-importing:

    [markdown]  ## Fitting the baseline

                Ridge regression is the baseline: linear, fast, and its
                single knob isolates the effect of regularization strength.

    [code]      # @title Fit
                from sklearn.linear_model import Ridge
                alpha = 1.0  # @param {type:"number"}
                model = Ridge(alpha=alpha).fit(X_train, y_train)

    [code]      # @title Evaluate
                import numpy as np
                rmse = float(np.sqrt(((model.predict(X_val) - y_val) ** 2).mean()))
                print(f"validation RMSE: {rmse:.3f}")
