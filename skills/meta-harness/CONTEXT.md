# meta-harness — Catalog Context

Rules for disposable builders that create durable agent harnesses.

## Contract

- Install this catalog per project, never globally by default.
- Every skill description begins exactly with:
  `Disposable meta-skill (delete after the harness is built):`
- The marker identifies temporary builders. No asset or generated target
  artifact may carry it.
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
scaffolds. Durable cross-project methodology remains in
`core/meta-harness`; platform-specific durable authoring and operations
remain in the engineering and ops catalogs.
