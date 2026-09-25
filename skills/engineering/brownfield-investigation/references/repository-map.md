# Repository Map

Read when the question needs the system's components, entry points, runtime
units, dependencies, or boundaries, or when no map exists at the pinned
revision. The map describes what is built today; it states no rule about
what must be built.

## Procedure

1. **Runtime units.** Start from what gets built and deployed — package
   manifests, container and deployment descriptors, process definitions,
   pipeline configuration — and list each unit that runs on its own.
2. **Entry points.** For each unit, find where work arrives: program entry
   functions, HTTP or RPC routes, message consumers, scheduled jobs,
   command-line commands, and plugin or extension hooks.
3. **Components.** Group code by responsibility as the code draws it, not
   as the directory names suggest; note where the two disagree.
4. **Dependencies.** Record which component depends on which, following
   imports and calls, and note cycles and calls that skip an apparent layer
   as observations, not violations.
5. **Boundaries and external systems.** List every place where a call
   leaves the process: network clients, SDKs, shared databases, queues,
   files, and the configuration keys that name their hosts.
6. **Build, run, test.** Record the commands the repository documents or
   its pipeline runs, and whether each was verified by running it.

## Depth

| Depth | Covers |
|---|---|
| ORIENT | runtime units, top-level components with one-line responsibilities, main entry points, external systems |
| ESTABLISH | plus dependency direction and every boundary in scope |
| EXHAUSTIVE | plus every entry point and outbound call in scope, with the configuration that selects each |

## Output

Per component: name, responsibility, location, entry points, runtime unit,
inbound and outbound dependencies, data it owns. Per boundary: the two
sides, the protocol, and where it is configured. Label an architectural
reading ("a layered design") as inferred unless a document or a person
states it.

## Gotchas

- In a repository with several deployables, directories and runtime units
  rarely line up; the deployment descriptors decide.
- Modules that are still built can be unreachable at runtime, and code
  outside the repository (infrastructure, other services) can call in. Mark
  what the repository cannot show as UNKNOWN.
