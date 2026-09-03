# scaffold — Catalog Context

Rules for disposable builders that establish a project of a specific topic.

## Contract

- Install this catalog per project, never globally by default.
- Install exactly one topic builder from this catalog — the one whose topic
  matches the project being built — together with `scaffold-disposal`. The
  topic builders are alternatives to one another, not layers that stack;
  `scaffold-disposal` is the catalog's own tool and accompanies whichever one
  is chosen.
- Every skill directory and `name` begins with `scaffold-`.
- Every skill description begins exactly with:
  `Disposable scaffold skill (delete after the harness is built):`
- The marker identifies temporary builders. No asset or generated target
  artifact may carry it.
- These builders exist only for project initialization. Their prefix and the
  `scaffold-disposal` skill make removal after verification explicit, so they
  do not consume context during normal development.
- Prefer complete, durable initialization guidance over minimizing this
  catalog's own instruction budget. The builders are temporary; their durable
  output belongs in the target project before disposal.
- Preserve working project choices; scaffolding is not permission to migrate.
- Builders investigate before editing, and present the concrete project shape
  they intend to create before creating it.
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
it excludes the catalog's other topic builders. The catalog also holds
`scaffold-disposal`, which is not a topic builder: it is the tool that removes
them, and it ships here so the catalog can be installed without `meta`.

Generic, pluggable harness machinery — complete harness architecture,
platform-neutral contract builders (branching, project workflow, agent
authority), platform lifecycle workflows, language-level
convention defaults — belongs to the `meta` catalog, whose skills stack
alongside whichever scaffold is chosen. Durable
cross-project methodology remains in `core/meta-harness`.

Both catalogs are disposable and are normally removed together once the project
is initialized and verified, each through its own disposal skill.
