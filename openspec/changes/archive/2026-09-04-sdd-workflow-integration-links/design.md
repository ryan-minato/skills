## Context

See proposal.md. `scripts/check_pr_policy.py` runs from the base branch and requires a `Spec:` line on ready pull requests; the pull request that lands this change is checked by the base's older regex, which accepts only the bare path, so that pull request keeps bare `Spec:` lines and adds the links beside them.

## Placement

| What Changes bullet | File(s) | Check that proves it |
|---|---|---|
| Link form on the template and the form | `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/task.yml` | readback |
| Policy accepts link, URL, and path | `scripts/check_pr_policy.py` (`SPEC_PATH`, `SPEC_URL`, `SPEC_RE`) | regex cases below |
| Rule stated | `.agents/knowledge/spec-workflow.md`, `.agents/skills/change-workflow/SKILL.md`, `.agents/knowledge/github-checks.md` | readback |

## Decisions

- **Link to the directory on the branch (`tree/<branch>/...`)**, not to a commit: the link stays valid as the record moves from in-flight to archived only if the body is updated at archive time, which the archive step now does; a commit link would freeze the record at one state. Alternative rejected: linking `main`, which shows nothing until merge.
- **Bare path still accepted** so older pull requests and the base-branch check keep passing; the template prescribes the link.

## Risks / Trade-offs

- [The link breaks when the record is archived] → the `Spec:` line is updated with the archive path when the change is archived in-request; under automated archiving the archive commit lands after merge and the link's branch is deleted, so the archived path on `main` is the durable reference — the archive commit's message names the changes.

## Verification plan

- `scripts/check_pr_policy.py`'s `SPEC_RE`: accepts the link form (in-flight and archive paths), a bare GitHub URL, a bare path, and `none — <reason>`; rejects a link to a non-change path and an empty line.
- `just check` passes; a clean-context readback of `spec-workflow.md` and the template states the link rule.

Skipped: none.
