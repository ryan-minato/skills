from __future__ import annotations

import unittest

from scripts.check_pr_policy import (
    REQUIRED_CHECKLIST_ITEMS,
    validate_pull_request,
)


def body(*, ready: bool = True, related: str = "N/A — direct request") -> str:
    checks = "\n".join(f"- [x] {item}" for item in REQUIRED_CHECKLIST_ITEMS)
    validation = "- `just check` — passed" if ready else "-"
    return f"""## Summary

Summary.

## Related issue

{related}

## Changes

- Change.

## Validation

{validation}

## Checklist

{checks}
"""


def pull_request(
    *,
    draft: bool = False,
    title: str = "feat: add policy checks",
    text: str | None = None,
    same_repository: bool = True,
) -> dict:
    base_repo = "ryan-minato/skills"
    head_repo = base_repo if same_repository else "contributor/skills"
    return {
        "title": title,
        "body": text if text is not None else body(ready=not draft),
        "draft": draft,
        "base": {"sha": "base", "repo": {"full_name": base_repo}},
        "head": {"sha": "head", "repo": {"full_name": head_repo}},
    }


class PullRequestPolicyTests(unittest.TestCase):
    def test_ready_same_repository_pr_passes(self) -> None:
        errors = validate_pull_request(
            pull_request(), ["feat: add policy checks", "test: cover policy"]
        )
        self.assertEqual(errors, [])

    def test_draft_does_not_require_ready_evidence(self) -> None:
        errors = validate_pull_request(
            pull_request(draft=True), ["feat: begin policy checks"]
        )
        self.assertEqual(errors, [])

    def test_missing_heading_is_actionable(self) -> None:
        text = body().replace("## Changes", "## Work")
        errors = validate_pull_request(pull_request(text=text), ["feat: update policy"])
        self.assertTrue(any("## Changes" in error for error in errors))

    def test_invalid_title_fails(self) -> None:
        errors = validate_pull_request(
            pull_request(title="Add policy checks"), ["feat: add policy checks"]
        )
        self.assertTrue(any("PR title" in error for error in errors))

    def test_issue_closing_form_passes(self) -> None:
        errors = validate_pull_request(
            pull_request(text=body(related="Closes #123")),
            ["feat: add policy checks"],
        )
        self.assertEqual(errors, [])

    def test_missing_issue_declaration_fails(self) -> None:
        errors = validate_pull_request(
            pull_request(text=body(related="See discussion")),
            ["feat: add policy checks"],
        )
        self.assertTrue(any("Related issue" in error for error in errors))

    def test_ready_pr_requires_completed_checklist(self) -> None:
        text = body().replace(
            "- [x] `just check` passes locally",
            "- [ ] `just check` passes locally",
        )
        errors = validate_pull_request(
            pull_request(text=text), ["feat: add policy checks"]
        )
        self.assertTrue(any("checklist is incomplete" in error for error in errors))

    def test_same_repository_invalid_commit_fails(self) -> None:
        errors = validate_pull_request(pull_request(), ["temporary work"])
        self.assertTrue(any("rebase into main" in error for error in errors))

    def test_same_repository_merge_and_revert_commits_are_exempt(self) -> None:
        errors = validate_pull_request(
            pull_request(), ["Merge branch 'main'", "Revert accidental change"]
        )
        self.assertEqual(errors, [])

    def test_fork_commit_subjects_are_not_enforced(self) -> None:
        errors = validate_pull_request(
            pull_request(same_repository=False), ["temporary work"]
        )
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
