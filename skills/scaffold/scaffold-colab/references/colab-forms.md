# Colab Form Controls

Read when adding parameters, cell titles, or hidden-code forms to code
cells. Bundled because the official reference (forms.ipynb) sits behind a
Colab login that agents cannot pass. Form comments are plain Python
comments: they render as native controls in Colab and degrade to harmless
comments in any other runner — which is why they beat ipywidgets for
Colab-centric work.

All form comments live at the top of the cell; a `@param` shares its line
with the assignment it controls.

## @param types

Pick the precise type for the value's shape, never a generic one.

    text     = 'value'       # @param {type:"string"}
    text2    = ''            # @param {type:"string", placeholder:"enter a value"}
    choice   = 'adam'        # @param ["adam", "sgd", "rmsprop"]
    choice2  = 'adam'        # @param ["adam", "sgd"] {allow-input: true}
    number   = 10.0          # @param {type:"number"}
    ratio    = 0.5           # @param {type:"slider", min:0, max:1, step:0.1}
    count    = 10            # @param {type:"integer"}
    level    = 3             # @param {type:"slider", min:0, max:10, step:1}
    flag     = True          # @param {type:"boolean"}
    day      = '2026-01-01'  # @param {type:"date"}
    anything = None          # @param {type:"raw"}
    literal  = [1, 2]        # @param ["1", "[1, 2]", "None"] {type:"raw"}

An all-integer `min`/`max`/`step` makes the slider integer-valued. `raw`
skips string coercion — the entered text or dropdown choice is evaluated as
a Python literal.

## Cell title and interleaved markdown

    # @title What this cell does { display-mode: "form" }
    # @markdown Caption rendered inside the form.

`# @title` names the cell — the label the doctrine's atomic-cell groups rely
on. Append `{ display-mode: "form" }` to show the form and hide the code by
default, or `{ run: "auto" }` to re-run the cell whenever a control changes
(only after its first manual run). `# @markdown` renders markdown inside the
form; use it for one-line control captions, not for the notebook's prose —
that belongs in markdown cells.

## Hiding code

View → Show/hide code (or the cell's context menu) toggles form-only,
code-only, or both. Ship reader-facing notebooks with noisy setup cells in
form-only mode under a `# @title` that says what the cell does.
