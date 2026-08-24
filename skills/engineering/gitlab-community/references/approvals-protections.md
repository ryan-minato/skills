# Platform-Enforced Merge and Tag Rules

Read when the user wants merges or tags enforced by the platform —
protected branches, approval rules, or merge-gate settings.

## Procedure

1. Locate the current protection mechanisms through the llms.txt index:
   protected branches
   (<https://docs.gitlab.com/user/project/repository/branches/protected/>
   as of this writing), protected tags, merge request approvals, and the
   merge-when-checks-pass settings. Read every tier badge — approval
   rules beyond the basics are Premium.
2. Agree the rule set with the user, smallest first: who may push and
   merge to the default branch, whether approvals are required and from
   whom, whether the pipeline must succeed to merge, and whether tags
   matching the release pattern are protected.
3. Merge gates that require a passing pipeline key on the project's real
   CI — verify the pipeline exists and passes before enabling, or every
   merge blocks.
4. Apply the settings with `glab` or the API when authenticated;
   otherwise record each setting — exact rule, exact value — in the
   AGENTS.md deposit as manual steps for the user.
5. Verify the result by reading the settings back, not by assuming the
   apply succeeded; on the free tier, record which of the wanted rules
   remain convention-only.
