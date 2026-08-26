# meta — Catalog Context

Rules for disposable builders that create durable agent harnesses.

## Contract

- Install this catalog per project, never globally by default.
- Every skill description begins exactly with:
  `Disposable meta-skill (delete after the harness is built):`
- The marker identifies temporary builders. No asset or generated target
  artifact may carry it.
- These builders exist only for project initialization. Their prefix and the
  `meta-disposal` skill make removal after verification explicit, so they do
  not consume context during normal development.
- Prefer complete, durable initialization guidance over minimizing this
  catalog's own instruction budget. The builders are temporary; their durable
  output belongs in the target project before disposal.
- Preserve working project choices; scaffolding is not permission to migrate.
- Builders investigate before editing. The complete architecture builder must
  present its concrete harness plan and receive user approval before construction.
- Durable rules must land in the target repository, registered tools, or other
  reachable sources before disposal.
- Do not ship documentation indexes, URL registries, docs-navigation tables, or
  static external inventories. Resolve volatile facts from first-party sources
  when the selected procedure needs them.
- Assets are raw starting shapes. Rework every line and remove every placeholder.

## Scope

This catalog owns complete harness architecture and disposable project
scaffolds, including platform-specific builders whose complete lifecycle output
is deposited into the target project before disposal. Durable cross-project
methodology remains in `core/meta-harness`; ordinary day-to-day platform skills
belong outside this catalog unless a builder generates them as project-specific
durable output.
