# History

Read when the current evidence cannot explain why code is the way it is, or
when a behavior's age or origin bears on a decision. History explains where
a behavior came from; it does not say what anyone wants today.

## Procedure

1. **Find the introducing change.** From the per-line history of the code
   in question, follow moves, renames, and reformatting back to the change
   that introduced the behavior, not the last one that touched it.
2. **Read what the change says.** Its message, the review or pull request
   it belongs to, the issues it references, and any decision record or
   changelog entry from the same time. Quote the stated reason with its
   location.
3. **Date the behavior.** How long it has existed tells how long consumers
   could have learned to rely on it.
4. **Look for coupling.** Files that repeatedly change together reveal
   dependencies the code does not show.

## Depth

| Depth | Use |
|---|---|
| ORIENT | only when the question itself is about origin |
| ESTABLISH | for PENDING_DECISION items whose age or origin affects the ruling |
| EXHAUSTIVE | for every pending item in scope |

## Output

An origin finding: when the behavior was introduced, by which change, the
stated reason quoted with its location or UNKNOWN when none was stated,
related later changes, and its age. Its evidence kind is documented (for
stated reasons) or observed (for dates and diffs); the intent it suggests
is the intent at that time, and current intent still needs a human.

## Gotchas

- Squashed merges and imported histories (one giant initial change) erase
  origins; record UNKNOWN rather than reading intent into the squash.
- Issue trackers outside the repository may be unreachable; record the
  reference so a person can follow it.
- Name people only as the role or team to ask; do not copy personal data
  from history into records that may be committed.
