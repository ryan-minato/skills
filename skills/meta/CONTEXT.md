# meta — Catalog Context

Rules for disposable builders that create durable agent harnesses.

## Contract

- Install this catalog per project, never globally by default.
- Install the catalog whole: its skills stack alongside one another.
- Every skill directory and `name` begins with `meta-`.
- Every skill description begins exactly with:
  `Disposable builder skill (delete after the harness is built):`
- The marker identifies temporary builders. No asset or generated target
  artifact may carry it.
- These builders exist only for project initialization. Their marker and the
  `meta-disposal` skill — which removes the `scaffold` builders too, since the
  two catalogs share the marker — make removal after verification explicit,
  so they do not consume context during normal development.
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

## Dependencies

- Builders here may depend on one another by name: the catalog is installed
  whole, so a sibling builder is present whenever this one is. They may also
  depend on `core` skills.
- A missing sibling is still handed off through
  `ryan-minato-skills-installing`, installing the whole `meta` catalog at
  project scope — never one builder alone.
- No grant to `scaffold` or any other catalog, and no dependency on or
  recommendation of skills from other repositories.
- `meta-harness-architecture` carries the `## Harness Methodology` section of
  `core/meta-harness` verbatim (validator-enforced) so it works when the core
  skill is absent; that is duplication, not a dependency.

## Naming

Catalog prefix `meta-` on every directory and `name`, enforced by
`CATALOG_NAME_PREFIXES`; the body after the prefix follows the default
`<subject>-<action>` shape (`meta-git-branching`, `meta-gpu-container`).

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
