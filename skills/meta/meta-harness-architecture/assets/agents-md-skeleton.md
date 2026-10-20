<!-- Maximal AGENTS.md skeleton. Fill only the sections backed by current
facts or explicit user decisions; delete every section that does not serve
the project. Target about 100 lines when finished. Approach 200 only when
this file is the complete harness for a lightweight project. -->

# [Project Name]

[One-sentence project purpose.]

## Always

<!-- The rules whose breach costs the same whatever the task: safety,
publishing, commits, language, the validation entry command, what to do
when blocked. They come first because attention decays with position. -->

- [Rule that applies to every task.]
- [Rule that applies to every task.]
- **Blocked work**: finish everything that does not depend on the decision;
  record each pending decision with where it is stuck, the options, the
  recommended option and why, and the completed work each option affects.
- **Mismatches**: a document, script, or command found to disagree with the
  repository is corrected in the same change, queued as a pending decision,
  or filed as [tracked item and label] — never left in the conversation.
- **Authorization**: proceed without asking on [reversible internal
  actions]; push only to the branch named in [the plan or specification
  that records pre-authorized actions]; never [force push, hard reset,
  delete the main branch] — the deny list is fool-proofing, [the sandbox]
  is the boundary, and an agent never relaxes this policy.
- **Checkpoints**: commits, never stashes; a task's drafts squashed into its
  one commit before the task is ticked; drafts never pushed; secret scan on
  every commit.
- **Pending decisions** live in [the description of the change request,
  a tracked file, or an issue] — a place a person sees, never only a
  scratch note.

## Project Map

<!-- Delete if a discoverable architecture document already provides this.
Keep it architecture-level: a map that changes on routine file additions is
too detailed. -->

```text
[path]/    <- [purpose]
```

## When To Read What

<!-- Knowledge and reference files are invisible without these pointers,
and a process-critical skill (publish gate, compliance review) is routed
here by situation, named; only auxiliary skills and registered tools may
rely on self-announcement. Scripts only need their usage documented
elsewhere in this file. -->

| Situation | Read |
|---|---|
| [Trigger condition] | [File, section, skill, or reachable endpoint] |

## Development Environment

- [How to enter and verify the environment.]
- [Isolation, approval, or sensitive-data boundary agents must respect.]
- [Non-interactive settings in force: pagers off, prompts disabled.]

## Validation

| Check | Command |
|---|---|
| [What this validates, full] | `[command]` |
| [Scoped, seconds-scale check for one file or module] | `[command]` |

## Workflow

- [Team-decided rule for commits, reviews, merges, or task state.]

## Keep In Sync

<!-- Delete this section if a dedicated sync skill owns synchronization,
and point to that skill from When To Read What instead. -->

| When this changes | Update |
|---|---|
| [Source-of-truth change] | [Dependent harness artifact] |
