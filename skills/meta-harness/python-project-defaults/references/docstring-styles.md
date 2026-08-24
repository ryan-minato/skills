# Docstring Styles

Read when choosing the project's docstring style because no existing
convention or user preference decides it, or when writing the chosen
style's rules into the harness.

## The Shared Baseline: PEP 257

Every style below builds on PEP 257, which is stable and worth recording
as-is:

- Every docstring is a string literal directly under the definition,
  enclosed in triple double quotes.
- The first line is a one-sentence summary in the imperative mood
  ("Return the parsed config", not "Returns" or "This returns"), ending
  with a period.
- A one-liner keeps the quotes and text on one line. A multi-line
  docstring is the summary line, a blank line, then the body; the closing
  quotes go on their own line.

Upstream: <https://peps.python.org/pep-0257/>.

## Choosing a Style

| Style | Choose when |
|---|---|
| Google (default) | No signal points elsewhere. Readable in plain source, well supported by renderers and linters. |
| NumPy | The project sits in the scientific stack (numpy/scipy/pandas ecosystem) or already depends on numpydoc. |
| reST / Sphinx fields | The project is already deep in Sphinx with plain autodoc and reST field lists. Rarely the right choice for new code. |

The three are recognizable at a glance:

- **Google** — indented named sections:
  `Args:` / `Returns:` / `Raises:` / `Yields:` / `Attributes:` /
  `Examples:`, each entry as `name (type): description`.
- **NumPy** — underlined section headings:
  `Parameters` / `Returns` / `Raises` over `----------` rules, each entry
  as `name : type` with the description indented below.
- **reST** — inline field lists:
  `:param name:` / `:type name:` / `:returns:` / `:rtype:` /
  `:raises Exc:`.

These sketches identify a style; do not author from them. Fetch the full
section rules from the upstream before writing the convention into the
harness:

- Google style — the docstring section of
  <https://google.github.io/styleguide/pyguide.html>
- NumPy style — <https://numpydoc.readthedocs.io/en/latest/format.html>
- reST fields — <https://www.sphinx-doc.org/en/master/usage/domains/python.html>
- Rendering Google/NumPy under Sphinx — the napoleon extension:
  <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>

## Doctest Examples

Any style can embed doctest examples (`>>>` blocks). They earn their keep
only for stable, deterministic, illustrative APIs: an example that needs
network, time, or randomness will flake, and an example of a churning API
rots. Keep every example copy-runnable exactly as printed. Examples only
execute if the project's test setup collects them — pair the convention
with that decision, and if nothing runs them, record them as illustrative
only. Upstream: <https://docs.python.org/3/library/doctest.html>.
