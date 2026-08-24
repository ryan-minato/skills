# Platform-Enforced Merge and Tag Rules

Read when the user wants merges or tags enforced by the platform rather
than by convention.

## Procedure

1. Fetch the current protection mechanisms from
   <https://docs.github.com/en/repositories>. GitHub has both rulesets
   and the older branch-protection rules; the docs state which is
   current, what each can enforce, and what the repository's plan and
   visibility allow — decide from the fetched page, not from memory.
2. Agree the rule set with the user, smallest first: what blocks a
   direct push to the default branch, whether approvals are required and
   how many, which status checks must pass, and whether tags matching
   the release pattern are protected.
3. Required status checks name real check names — use the ones collected
   from the repository's workflows, verbatim. A name with no producer
   blocks every merge forever.
4. Apply the settings with `gh` (the manual is
   <https://cli.github.com/manual/>) when authenticated; otherwise
   record each setting — exact rule, exact value — in the AGENTS.md
   deposit as manual steps for the user.
5. Verify the result by reading the settings back, not by assuming the
   apply succeeded.
