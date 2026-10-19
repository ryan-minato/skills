## Why

The specification layer of the library — `sdd/spec-driven-development` for the practice and `meta/meta-spec-workflow` for the deposited contract — says what a good requirement is and where the two gates sit, but not what a specification must answer for an agent to know it is done: the done criterion the agent can execute itself, the non-goals, the deliverable, and the irreversible or outward actions the change pre-authorizes. Nor does it tell an agent which artifact a given fact belongs in, so proposals restate specs and designs carry acceptance. A written specification for agent harnesses states those four answers and four quality rules, and two repositories whose contract the builder deposited (`hoshiori-dev/devcontainer-features`, `hoshiori-dev/devcontainer-templates`) show, after five weeks and 63 archived changes, what the deposit grew that the builder never offered: an artifact-roles table with an invariant test and an information test, a two-list acceptance (what becomes true, what stays true) that points to scenarios instead of restating them, a recorded approval kept by automation and withdrawn when the package changes, a check that keeps a machine-readable surface equal to its spec, dated observations labelled as observations — and bootstrap-time narrative left in the durable contract that nothing told the builder to keep out.

## What Changes

- `sdd/spec-driven-development`: the specification quality rules gain the four answers every specification carries — a done criterion the implementing agent can execute itself (task-level closing conditions included), the non-goals, the deliverable, and the pre-authorized irreversible or outward actions — and the agent refuses a specification that lacks any of them; the loop's specify step asks for them. A new artifact-roles rule assigns each record artifact one role (outcome and acceptance; behavior contract; approach bounds and change-local information; steps with verification) with two placement tests — an invariant that would hold under any approach belongs with the acceptance, one that holds only because of the chosen approach belongs in the design; information needed the next time the behavior changes belongs in the spec, information needed only for this change belongs in the design — and a two-list acceptance shape (becomes true, stays true) that points to scenarios for behavior and never restates them.
- `meta/meta-spec-workflow`: the questioning round settles, and the deposited contract carries, the artifact roles with the two tests, the acceptance shape, the pre-authorized-actions line of the record, and, for a surface the specification is the source of truth for (option metadata, a schema, a command surface), the comparison check that keeps the surface equal to the spec; the contract design offers two further recorded-approval implementations with their costs beside the platform's review approval — a label kept by automation that recomputes it from the package and withdraws it when the package changes, and a digest-bound approval artifact — and says why neither replaces the conversational default; the durable-output rules require dated, labelled observations and forbid bootstrap-time narrative in the contract, with a reclamation trigger in its "Update this file when" list; the contract asset gains the matching sections.

## Skills touched

- `sdd/spec-driven-development` (modified): the four answers a specification carries; artifact roles with the invariant and information tests and the two-list acceptance.
- `meta/meta-spec-workflow` (modified): the deposited contract's artifact roles, acceptance shape, pre-authorized actions, and spec-as-source check; recorded-approval implementations offered with costs; observations dated and bootstrap narrative excluded.

## Installed behavior

An agent that loads `spec-driven-development` writes and reviews a specification that states how its completion is judged by a command or scenario the agent itself can run, what it does not do, what it delivers, and which irreversible or outward actions are pre-authorized, and refuses to treat a specification missing one of them as complete; it places each fact in the artifact that owns it by the two tests, writes acceptance as what becomes true and what stays true, and points to scenarios instead of restating them. An agent that loads `meta-spec-workflow` asks for those facts in its round, deposits a contract that carries the roles, the tests, the acceptance shape, the pre-authorized-actions line, and the spec-as-source check where one applies, offers recorded-approval implementations with their costs rather than one mechanism, dates every observation it records, and leaves no bootstrap narrative in the durable file. This is a `feat`: both skills gain capabilities; nothing they said before becomes wrong.

## Impact

- `skills/sdd/README.md` and `README.zh.md`: the `spec-driven-development` row names the four answers and the artifact roles.
- `skills/meta/README.md` and `README.zh.md`: the `meta-spec-workflow` row names the artifact roles and the recorded-approval implementations.
- `.claude-plugin/marketplace.json`: unchanged unless a description changes; regenerated by `just gen-marketplace` if it does.
- Mirrored files: none.
- This repository's harness: `.agents/knowledge/spec-workflow.md` and the project schema's templates are not changed by this change; adopting the artifact roles and the acceptance shape here is a separate repository change.

## Non-goals

- `sdd/openspec-workflow` and `sdd/spec-kit-workflow` stay as they are; a tool-specific rule such as stopping a propose flow before its task list, or the documented hand-edit of a domain's purpose line, belongs to the framework skill and is already recorded there or in the project contract.
- The rule that the project's own harness is spec-less stays; the reference repository that specified its harness governance as domains is recorded as a cost (one rule restated in five files), not adopted.
- The library's treatment of a tool-generated task list (pushed as a draft, marked after-approval) stays; the reference repository's rule to delete it is a project choice.
- The harness methodology (`core/meta-harness`, the `meta` harness builders) is the separate change `harness-methodology-alignment`, which points here for the specification layer.
- No change to `meta-agent-authority`; the pre-authorized-actions line in the record is where the authority policy's execution-level pre-authorization is written, and the authority builder is not reopened for it.

## Tracked work

No issue.
