# Deciding Whether, at Which Level, and With Which Approach

Read when the user asks whether, at which level, or with which approach
family to adopt spec-driven development, what the rejected alternatives
cost, or before recommending a tool. The choice, and the rules that go
with it, are recorded by the project's harness builder, not by this skill;
say so when asked directly.

## Three levels

| Level | The spec is... | Obligation after the change ships |
|---|---|---|
| spec-first | written before the change and used to build it | none — it may be archived or discarded |
| spec-anchored | kept as the living description of the feature | every behavior change updates the spec first |
| spec-as-source | the only file humans edit; code is regenerated | humans never patch code by hand |

Spec-anchored fits anything maintained beyond one release and spec-first a
bounded delivery nobody will evolve; spec-as-source is experimental and
only worth it when the toolchain regenerates code reliably. The level fixes
what must be maintained, so it is chosen before the tool.

## When it pays

Use SDD when requirements can be stated before the code exists, the change
outlives one session, several people or agents touch the same behavior, or
acceptance keeps being argued after the fact. Skip it for throwaway
exploration, a spike whose purpose is to discover the requirements, and a
prototype still in its validation window — there, code is the cheapest
spec. Default: adopt SDD the moment a prototype gets its first user who is
not its author.

## Approach families

Each family fits a situation; a spec tool the project already runs is the
answer unless the user asks to change it. Give the fitting family with its
reason; the framework skill for the tool carries its records, commands,
and request automation.

- **A spec-first kit with a project constitution and per-feature spec,
  plan, and task files** (GitHub Spec-Kit; `spec-kit-workflow`) fits a new
  application delivered feature by feature: the heavier, whole-process
  shape a new codebase needs to form habits. The kit does not oblige
  anyone to keep a spec current after the feature ships; at spec-anchored
  that obligation is a written project rule. Its approval package is the
  specification and the plan.
- **A spec-anchored change workflow with specs organized by capability**
  (OpenSpec; `openspec-workflow`) fits a library, framework, or
  infrastructure with no code yet — a library's contract is its
  capabilities, not a sequence of features — and any existing code: it is
  lighter, built for existing systems, and never asks for specs of code
  that is not changing. Its approval package is the proposal, the delta
  specs, and the design.
- **An IDE's native requirements, design, and tasks files** (Kiro) fit a
  team that lives in that IDE; whether other agents honor them is
  unverified, so a tool-agnostic family fits better when several agents
  work the repository. The approval package is the requirements and the
  design; task ticks are status. No framework skill exists for it: run the
  loop by hand with the IDE's own documentation.
- **Committed specification documents** under one directory, linked from
  tracked work, fit a team that refuses tooling: the same discipline with
  a hand-run loop, the approval package being the specification with its
  approach section, the delta merged into the domain document by hand
  inside the request before ready, and a required-headings lint as the
  only validator. No framework skill exists for it.
- **A custom layout** fits only a stated constraint none of the above
  meets, and costs every future agent the tool's validation and
  conventions.

Tool commands and file layouts change between releases: verify them from
the tool's own `--help` and current documentation before running or
describing one. This skill deliberately lists none.

## Rejected alternatives

- **Approving the specification alone and treating the design as an
  implementation draft.** It keeps the gate pure but hands the reviewer no
  say over the approach; implementations diverge in ways the specification
  cannot constrain. A bounded design in the package trades a little
  implementation diversity for controllability, and still excludes steps.
- **Archiving after the merge by an automation that pushes to the
  integration branch.** It keeps the request's record editable until the
  end but needs a push identity with a protected-branch bypass — a
  user-owned repository on some platforms cannot grant one — and leaves
  the integration branch holding an unarchived record between the merge
  and the run. Archiving inside the request, by hand or by a bot that
  pushes to the request's own branch, needs neither.
- **Always requiring a design.** A wording change would carry a design
  that says nothing; the warranted rule keeps the package proportional.
- **Labels or issue types as the specification's lifecycle.** A spec is a
  document in the repository; its lifecycle lives in the tool's layout.
  Status labels a workflow derives from the record are facts about it,
  never its source of truth.
