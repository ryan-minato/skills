# Planning, Labels, and Goals

Read when the selected harness uses labels, milestones, boards, Scrum/Kanban,
iterations, epics, objectives, or work-item hierarchy.

Before design, resolve current capabilities through `llms.txt` with terms such
as `work items`, `labels`, `milestones`, `issue boards`, `Scrum`, `Kanban`,
`iterations`, and `epics`; verify the target version and tier.

## Choose the planning system

When the target carries a workflow file (`.agents/knowledge/gitlab-workflow.md`),
the planning system is not chosen here: the file already names the GitLab
objects in use and the ones deliberately not used — a file without a board
gets no board, one without `priority::*` labels gets none, one without an
iteration cadence gets none. Add the label semantics, board conventions,
and grooming owner this builder settles to that file. The choices below
apply where no workflow file settles the question, and the result is
written as that file.

- **Kanban:** default for continuous flow. Start with one maintained board and
  the fewest lists that reveal backlog, in-progress work, review, blocking, and
  completion. Define WIP handling and a grooming owner.
- **Scrum:** use only when the team commits to backlog refinement, sprint or
  iteration planning, review, and retrospective. Record cadence, definition of
  ready/done, rollover policy, and whether the instance supports the chosen
  iteration features.
- **Other board:** record its authoritative method, list semantics, grooming
  cadence, and mapping to GitLab fields.
- **No board:** use work-item queries, labels, assignees, and milestones; do not
  create a board merely because GitLab offers one.

Boards arrange work items; they do not replace them. Every list or swimlane
must have one meaning, one owner, and a keep-current rule.

## Design the taxonomy

Inspect project and inherited group labels first. Extend them rather than
creating synonyms. Recommend scoped labels when the live instance supports the
needed behavior.

Minimum taxonomy unless equivalent existing labels cover it:

- Type: bug, feature, incident, refactor, task. `bug` is the label for the
  Issue content contract in [work-items-and-mrs.md](work-items-and-mrs.md);
  pick one name per concept and use it in labels, templates, and prose alike.
- Priority: semantic high/medium/low by default; let the user choose P0-P5 or a
  documented project vocabulary instead.
- Status: needs-triage and blocked. Add workflow labels only when status fields
  or board lists do not already own the state.
- Area: derive `area::*`, `module::*`, `catalog::*`, or the project's equivalent
  from real ownership and architecture boundaries. Never ship fictional areas.

Every label needs a unique name, distinct color, concise meaning, allowed
combinations, creator/removal trigger, and owner. Rework
`assets/labels.json` into the target-project taxonomy
file and keep its human/agent meanings in reachable guidance. Run
`sync_labels.py` against that file without `--apply`, review
creates/updates/deletes, and do not prune existing labels without separate
explicit approval.

## Milestones and hierarchy

Use milestones for a coherent future state or long-range goal. Use iterations
for recurring timeboxes when supported and actually practiced. Use epics or
parent work items only when they clarify cross-milestone or cross-project
structure. The Free-tier fallback is labels, linked/child work items where
available, and milestones.

Record how objects relate, who creates/closes them, how unfinished work moves,
and whether releases share milestone names. Closing a milestone does not imply
closing every open child; define the policy explicitly.

Done when: every planning object answers a real team question, has an owner and
maintenance rhythm, and no two fields are competing sources of truth.
