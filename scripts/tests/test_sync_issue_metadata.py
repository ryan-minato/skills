from __future__ import annotations

import unittest

from scripts.sync_issue_metadata import plan_labels


class IssueMetadataTests(unittest.TestCase):
    def test_adds_priority_and_catalog(self) -> None:
        plan = plan_labels("### Priority\nHigh\n\n### Catalog\nCore", [])
        self.assertEqual(plan["add"], ["catalog/core", "priority/high"])
        self.assertEqual(plan["remove"], [])

    def test_replaces_managed_values(self) -> None:
        plan = plan_labels(
            "### Priority\nLow\n\n### Catalog\nMeta",
            ["priority/high", "catalog/core", "bug"],
        )
        self.assertEqual(plan["add"], ["catalog/meta", "priority/low"])
        self.assertEqual(plan["remove"], ["catalog/core", "priority/high"])

    def test_preserves_unmanaged_labels(self) -> None:
        plan = plan_labels(
            "### Priority\nMedium\n\n### Catalog\nWriting",
            ["bug", "priority/medium", "catalog/writing"],
        )
        self.assertFalse(plan["changed"])

    def test_repeated_event_is_idempotent(self) -> None:
        first = plan_labels("### Priority\nHigh\n\n### Catalog\nCore", [])
        current = first["add"]
        second = plan_labels("### Priority\nHigh\n\n### Catalog\nCore", current)
        self.assertFalse(second["changed"])
        self.assertEqual(second["add"], [])
        self.assertEqual(second["remove"], [])

    def test_invalid_priority_leaves_labels_unchanged(self) -> None:
        plan = plan_labels(
            "### Priority\nUrgent\n\n### Catalog\nCore",
            ["priority/high", "catalog/meta"],
        )
        self.assertFalse(plan["changed"])
        self.assertIn("Priority", plan["warning"])

    def test_absent_fields_leave_labels_unchanged(self) -> None:
        plan = plan_labels("### Problem\nSomething broke", ["priority/high"])
        self.assertFalse(plan["changed"])
        self.assertIn("No managed", plan["warning"])


if __name__ == "__main__":
    unittest.main()
