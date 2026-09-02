# meta — Catalog Context

Rules for disposable builders that create durable agent harnesses.

## Contract

- Install this catalog per project, never globally by default.
- Install the catalog whole: its skills stack alongside one another.
- Every skill directory and `name` begins with `meta-`.
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
- Contract flow is one-way. Contract builders (branching model, workflow
  design, agent authority) deposit platform-neutral contracts into the target
  project; platform builders consume those contracts, never reopen a settled
  decision, and never let a platform capability reshape the upstream model.
- Durable rules must land in the target repository, registered tools, or other
  reachable sources before disposal.
- Do not ship documentation indexes, URL registries, docs-navigation tables, or
  static external inventories. Resolve volatile facts from first-party sources
  when the selected procedure needs them.
- Assets are raw starting shapes. Rework every line and remove every placeholder.

## Scope

This catalog owns generic, pluggable harness machinery: complete harness
architecture, platform-neutral contract builders (branching, project
workflow, agent authority) whose contracts the platform builders
consume, platform-specific lifecycle builders whose complete output is
deposited into the target project before disposal, and language-level
convention defaults. Its skills stack alongside one another, so the catalog is
installed whole. A builder whose topic selects the entire project shape — and
therefore excludes the catalog's other builders — belongs to `scaffold`
instead. Durable cross-project methodology remains in `core/meta-harness`.
Day-to-day platform operation guidance is not published as a standing public
skill in this repository: a platform builder generates it as project-specific
durable output before the builder is disposed of.
