# Commits, Branches, and Contribution Flow

Read when the harness defines commit format, branch naming, merge/squash
strategy, contribution flow, or commit enforcement.

Run `analyze_history.py` before proposing a convention. Inspect merge settings,
default branch, recent branches/MRs, current CONTRIBUTING guidance, changelog
trailers, and release tags. Preserve a coherent working style unless the user
approves migration.

## Commit contract

When no convention exists, recommend Conventional Commits only if release,
changelog, or automation needs structured intent; otherwise choose the smallest
documented format the team will enforce. Fetch the chosen specification from
its authoritative source before writing policy.

Record allowed types, scope meanings, subject rules, breaking-change syntax,
trailers, merge/revert/fixup exemptions, and examples based on real project
domains. If `Changelog:` trailers feed releases, match their values and case to
the changelog configuration; commits without the expected trailer can disappear
from generated notes silently.

Rework [`assets/check_commits.py`](assets/check_commits.py) only when
mechanical enforcement is selected. Copy it into the target, edit its CONFIG,
and verify message, file, and MR-range modes. CI needs complete MR history and
must validate only the intended range. In squash workflows, validate the MR
title because individual commits may not reach the default branch.

## Branch and merge contract

Agree branch source, naming, lifetime, early draft-MR timing, target branch,
merge method, squash policy, source-branch cleanup, MR size guidance, and review
expectations. Use GitLab's draft/ready operations rather than manually editing a
title prefix.

Do not impose git-flow or GitLab Flow merely because the host is GitLab. The
branch model follows release/deployment needs and existing practice.

## Contribution enforcement

Keep local hooks, documented commands, CI, merge settings, and public
CONTRIBUTING guidance aligned. A policy without feedback is convention-only;
state that plainly. Premium push rules or approvals are optional platform
enforcement verified against the target tier, not the default.

When an MR checklist is checked by CI, keep the required heading and checklist
items synchronized with the template and fail closed if the description is
missing or truncated. MR-event rules that establish MR pipelines must be in the
effective top-level configuration supported by the target instance.

Done when: commit/branch/merge policy matches history or an approved migration,
the same contract is visible to contributors and agents, and each selected
enforcement mechanism rejects a deliberately invalid example.
