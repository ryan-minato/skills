# Dependency Managers

Read when choosing or recording how the project manages dependencies and
environments, or when the target matches one of the three usage shapes
below.

## Three Shapes, Three Working Modes

The shape of the target decides the mode before any tool preference does:

- **One-off script.** No project file at all: declare dependencies in the
  script itself with PEP 723 inline metadata and run it with `uv run`.
  The script stays a single portable file.
- **Pinned-requirements workflow.** The pip-tools model: a human-edited
  input file compiled into a fully pinned output (`uv pip compile`), for
  projects that want lockfile discipline without adopting full project
  management — common when deployment tooling expects a
  `requirements.txt`.
- **Full project.** `pyproject.toml` managed end to end — init, add,
  lock, sync — with a committed lockfile. The default shape for anything
  with more than one file and a lifetime beyond a week.

## The Tools

| Tool | One line |
|---|---|
| uv (default) | Fast package and project manager covering all three shapes above |
| Poetry | Project manager with its own lockfile and publishing flow |
| PDM | PEP-standards-focused project manager |
| Hatch | Project manager and build backend with environment matrices |
| pip + pip-tools | The baseline installer plus compile-to-pin discipline |
| mamba / micromamba | Conda-ecosystem package manager for compiled scientific stacks |

Fetch current install commands and lockfile workflows from the selected
tool's official first-party documentation before writing them into the target.

## When an Alternative Wins

- An existing lockfile decides by itself: its format names the manager
  (`poetry.lock` → Poetry, `pdm.lock` → PDM, `uv.lock` → uv), and the
  project keeps it.
- Dependencies only distributed through conda channels (compiled
  scientific packages) → micromamba, whatever the default says.
- Everything else on the list is a legitimate user preference, not a
  project need — record the preference without arguing.
