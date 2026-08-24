# Documentation Generators

Read when the project will publish generated documentation and the
generator must be chosen — skip entirely when it will not.

## The Tools

| Tool | One line |
|---|---|
| Zensical (default) | Static site generator from the Material for MkDocs team, built as its successor |
| MkDocs | Markdown-first documentation site generator |
| Material for MkDocs | The dominant MkDocs theme, effectively a distribution |
| Sphinx | The reST-native generator with the deepest Python ecosystem |
| mkdocstrings | API reference rendered from docstrings, for the MkDocs family |

**Zensical maturity caveat:** Zensical is young. Fetch its docs first and
confirm its current status and feature coverage; when the user needs a
settled tool, record Material for MkDocs instead.

Fetch current install commands and site configuration from the selected
generator's official first-party documentation before writing them into the target.

## Selection Rules

- Align with the docstring convention the project already records: the
  Markdown family pairs with any docstring style through mkdocstrings;
  Sphinx pairs natively with reST fields and with Google/NumPy styles
  through its napoleon extension.
- API-reference-from-docstrings requires a bridge — mkdocstrings in the
  MkDocs family, autodoc in Sphinx; without one, docstrings never reach
  the site.
- Sphinx wins when the project needs its ecosystem: intersphinx
  cross-project linking, scientific-stack conventions, or an existing
  reST corpus.
- No generator without content to publish: a scaffolded empty site is
  maintenance debt, not documentation.
