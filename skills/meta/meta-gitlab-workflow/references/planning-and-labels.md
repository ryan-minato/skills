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

Commit the file as `.gitlab/labels.json`. Project labels never go into tf;
group labels do, in the group's configuration repository
([config-repo.md](config-repo.md)). Deliver `assets/sync_labels.py` as the
target's `scripts/sync_labels.py`: it calls the REST API with the token in
`GITLAB_TOKEN` and edits only the project's own labels. How the file
reaches GitLab is a harness option:

- **CI (the default).** Deliver `assets/gitlab-ci-labels-sync.yml`.
  `CI_JOB_TOKEN` cannot call the labels API, so the job uses a project
  access token with the least role that manages labels: Planner from GitLab
  17.7, Reporter before. It is stored as a masked, protected variable of
  that project. On GitLab.com, project access tokens need Premium or
  Ultimate (checked 2026-10-14).
- **Manual.** The only mode on GitLab.com Free. A maintainer runs
  `GITLAB_TOKEN=... python3 scripts/sync_labels.py --file .gitlab/labels.json --api-url https://HOST/api/v4 --project PATH --apply`,
  reading the token from their glab login as in
  [config-as-code.md](config-as-code.md); record the command in knowledge.

In CI mode the label file never ships without its job. Pruning labels
absent from the file is a separate option, off by default. A user who asks
to delete the labels the file does not list has chosen pruning: say once
that deleting a label strips it from every work item and merge request
that carries it, add `--prune`, and record the choice — do not ask again.
Record both choices in `platform-settings.md`.

## Milestones and hierarchy

Use milestones for a coherent future state or long-range goal; a goal
milestone's description carries `## Vision`, `## Observable completion`,
and `## Non-goals`, so completion is linked from there rather than
restated elsewhere. Use iterations
for recurring timeboxes when supported and actually practiced. Use epics or
parent work items only when they clarify cross-milestone or cross-project
structure. The Free-tier fallback is labels, linked/child work items where
available, and milestones.

Record how objects relate, who creates/closes them, how unfinished work moves,
and whether releases share milestone names. Closing a milestone does not imply
closing every open child; define the policy explicitly.

Done when: every planning object answers a real team question, has an owner and
maintenance rhythm, and no two fields are competing sources of truth.
