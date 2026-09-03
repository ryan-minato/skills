# scaffold — Catalog Context

Rules for disposable builders that establish a project of a specific topic.

## Contract

- Install this catalog per project, never globally by default.
- Install exactly one topic builder from this catalog — the one whose topic
  matches the project being built. The topic builders are alternatives to
  one another, not layers that stack.
- Every skill directory and `name` begins with `scaffold-`.
- Every skill description begins exactly with:
  `Disposable builder skill (delete after the harness is built):`
- The marker identifies temporary builders. No asset or generated target
  artifact may carry it.
- These builders exist only for project initialization. Their marker and the
  `meta` catalog's `meta-disposal` skill make removal after verification
  explicit, so they do not consume context during normal development; a
  scaffold builder names that skill in its closing step and never deletes
  builders by hand.
- Prefer complete, durable initialization guidance over minimizing this
  catalog's own instruction budget. The builders are temporary; their durable
  output belongs in the target project before disposal.
- Preserve working project choices; scaffolding is not permission to migrate.
- Builders investigate before editing, and present the concrete project shape
  they intend to create before creating it.
- A topic builder loads before the `meta` entry workflow on a project of its
  topic — its description claims the project kind — and then names
  `meta-harness-building` for the rest of the harness and the closing of the
  build, the same way it names other `meta` builders.
- Disposable builders are never committed to the target repository. Deletion
  happens after verification and before the work goes to review, only on the
  user's explicit decision, and only through `meta-disposal`.
- Durable rules must land in the target repository, registered tools, or other
  reachable sources before disposal.
- Do not ship documentation indexes, URL registries, docs-navigation tables, or
  static external inventories. Resolve volatile facts from first-party sources
  when the selected procedure needs them.
- Assets are raw starting shapes. Rework every line and remove every
  placeholder.

## Dependencies

- Builders here may depend on `meta` builders by name (work tracking, agent
  authority, GPU containers) and on `core` skills. The expected install is
  one scaffold builder plus the whole `meta` catalog, both at project scope
  and disposed together.
- A missing `meta` builder is handed off through
  `ryan-minato-skills-installing`, installing the whole `meta` catalog —
  never one builder alone.
- No grant between scaffold builders (they are alternatives), and no
  dependency on or recommendation of skills from other repositories.

## Naming

Catalog prefix `scaffold-` on every directory and `name`, enforced by
`CATALOG_NAME_PREFIXES`; the body names the project topic
(`scaffold-data-science`, `scaffold-colab`).

## Scope

This catalog owns topic-specific project scaffolding methodology: what a
project of this kind must contain, how its data, code, and outputs are
organized, and which conventions its agents inherit. A topic builder earns a
place here only when its topic selects the whole project shape, so that adding
it excludes the catalog's other topic builders.

Generic, pluggable harness machinery — complete harness architecture,
platform-neutral contract builders (branching, project workflow, agent
authority), platform lifecycle workflows, language-level
convention defaults — belongs to the `meta` catalog, whose skills stack
alongside whichever scaffold is chosen. Durable
cross-project methodology remains in `core/meta-harness`.

Both catalogs are disposable and are removed together once the project is
initialized and verified, through `meta-disposal`.
