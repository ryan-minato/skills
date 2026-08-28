# Depositing the Contract

Read on every build. This is what makes the agreed conventions survive the
removal of this builder.

## Place the file where the project already keeps knowledge

Probe before creating. Look for an existing agent knowledge location — a
knowledge directory beside the agent entrypoint, a docs tree the entrypoint
already points into, or an existing contribution document — and put the
contract there under a name that matches its neighbours. Create a new location
only when none exists.

Do not split the contract across files. Branch naming, tag naming, protected
refs, merge method, and the hotfix path are consulted together, at the same
moments, by the same reader. One file, one lookup.

If a contribution document for humans already states part of the contract,
keep that document as the human-facing summary and make the agent-facing file
the source of truth. Note in each which one it is; two files stating the same
rule with no stated precedence is the drift that follows every harness build.

## Write the pointer as an event trigger

Branching rules are not needed on every task, so the entrypoint must not
instruct the agent to read them on every task. It must name the operations that
trigger the read:

```markdown
Read `<path>` before creating a branch, opening a pull or merge request,
merging, or creating a tag.
```

Adjust the operation list to the model — an environment chain adds promoting a
change between environments; release branches add cherry-picking a fix. What
never changes is that the pointer names operations. "See `<path>` for our git
conventions" fires unreliably, and a file nothing reaches is a file that does
not exist.

Keep the entrypoint's own copy to that one line. Restating a branch name or the
merge method there creates a second source of truth that will disagree with the
first within a release.

## What the file must answer

Adapt the skill's `assets/git-workflow.md` rather than writing from scratch,
and delete every section the selected model does not use. The finished file
answers, without the reader needing anything else:

- Which model, and the one-sentence reason it was selected.
- Every long-lived branch by exact name, with what it means and what may merge
  into it.
- The short-lived branch naming rule, and who or what generates the name.
- Tag format and the version scheme.
- The merge method per contribution origin.
- The hotfix path as an ordered sequence of refs.
- Which rules are platform-enforced, which are advisory, and which are
  convention, with the readback path for the enforced ones.
- Branch deletion and the support window for any maintained version.

State rules as instructions the next agent can follow, not as background. "Cut
stable branches from `main` as late as possible" is background; "cut
`<major>-<minor>-stable` from `main` at release candidate time, then cherry-pick
fixes into it after they land on `main`" is the contract.

## Register the update triggers

Name the events that make the file wrong, inside the file, so the next agent
knows when to revisit rather than trust:

- A new long-lived branch or environment is added or retired.
- The platform's protection or merge-method settings change.
- A deploy job's branch trigger changes — the CI configuration and this file
  name the same branches and must agree.
- A maintained version reaches end of support.

## Disposal test

Before recommending cleanup, search the target project for this skill's name,
its paths, and its disposable marker; none may appear in a deposited file. Then
verify the reverse direction: from the entrypoint alone, the pointer resolves,
and from the contract alone, an agent can name the model, create a correctly
named branch, choose the merge method, and route a hotfix. If any of those needs
the conversation, the deposit is incomplete.
