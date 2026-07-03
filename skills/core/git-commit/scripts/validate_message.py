#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///
"""Validate a git commit message against the 50/72 rule and structural conventions.

Rules enforced:
  title-not-empty              Title line must not be empty
  title-max-length             Title <= 50 characters (override with --max-title)
  title-no-period              Title must not end with a period
  title-no-wip                 Title must not be a bare WIP marker
  blank-line-after-title       Line 2 must be blank when more lines follow
  body-max-line-length         Body lines <= 72 characters (override with --max-body)
  breaking-change-case         BREAKING CHANGE footer token must be uppercase

Exit codes:
  0  message is valid
  1  one or more validation errors found
  2  invalid arguments or unreadable input

Output (stdout): JSON object
  {"valid": bool, "errors": [{"line": int, "rule": str, "message": str}]}
"""

import argparse
import json
import re
import sys
from pathlib import Path

LOWERCASE_BREAKING = re.compile(r"^(breaking[ -]change)\s*:", re.IGNORECASE)


def validate(message: str, max_title: int, max_body: int) -> list[dict]:
    errors: list[dict] = []
    lines = message.rstrip("\n").splitlines()

    if not lines or not lines[0].strip():
        errors.append(
            {
                "line": 1,
                "rule": "title-not-empty",
                "message": "Title line must not be empty",
            }
        )
        return errors

    title = lines[0].rstrip()

    if len(title) > max_title:
        errors.append(
            {
                "line": 1,
                "rule": "title-max-length",
                "message": (
                    f"Title is {len(title)} characters; must be <= {max_title}. "
                    f'Current title: "{title}"'
                ),
            }
        )

    if title.endswith("."):
        errors.append(
            {
                "line": 1,
                "rule": "title-no-period",
                "message": "Title must not end with a period",
            }
        )

    if re.fullmatch(r"(wip|WIP|Wip)[.!]*", title.strip()):
        errors.append(
            {
                "line": 1,
                "rule": "title-no-wip",
                "message": (
                    "Title is a bare WIP marker; describe the change instead "
                    "(or use the project's WIP convention deliberately)"
                ),
            }
        )

    if len(lines) >= 2 and lines[1].strip():
        errors.append(
            {
                "line": 2,
                "rule": "blank-line-after-title",
                "message": (
                    "Line 2 must be blank to separate the title from the body; "
                    f'got: "{lines[1]}"'
                ),
            }
        )

    for i, line in enumerate(lines[2:], start=3):
        if len(line) > max_body:
            errors.append(
                {
                    "line": i,
                    "rule": "body-max-line-length",
                    "message": (
                        f"Line {i} is {len(line)} characters; body lines must "
                        f"be <= {max_body}. Wrap the line"
                    ),
                }
            )
        match = LOWERCASE_BREAKING.match(line)
        if match and match.group(1) not in ("BREAKING CHANGE", "BREAKING-CHANGE"):
            errors.append(
                {
                    "line": i,
                    "rule": "breaking-change-case",
                    "message": (
                        f'Footer token "{match.group(1)}" must be uppercase: '
                        '"BREAKING CHANGE:" (tooling matches it case-sensitively)'
                    ),
                }
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a git commit message against the 50/72 rule.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  message is valid
  1  one or more validation errors found
  2  invalid arguments or unreadable input

Output (stdout): JSON
  {"valid": bool, "errors": [{"line": int, "rule": str, "message": str}]}

Rules checked:
  title-not-empty          Title must not be empty
  title-max-length         Title <= 50 characters (--max-title overrides)
  title-no-period          Title does not end with a period
  title-no-wip             Title is not a bare WIP marker
  blank-line-after-title   Line 2 must be blank when more lines follow
  body-max-line-length     Body lines <= 72 characters (--max-body overrides)
  breaking-change-case     BREAKING CHANGE footer token is uppercase

Examples:
  uv run validate_message.py --message "fix: reject expired tokens"
  uv run validate_message.py --file .git/COMMIT_EDITMSG
  uv run validate_message.py --max-title 72 --file msg.txt
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--message",
        "-m",
        metavar="TEXT",
        help="Commit message text (literal \\n sequences are treated as newlines)",
    )
    group.add_argument(
        "--file",
        "-f",
        metavar="PATH",
        help="Path to a file containing the commit message",
    )
    parser.add_argument(
        "--max-title",
        type=int,
        default=50,
        metavar="N",
        help="Maximum title length (default: 50)",
    )
    parser.add_argument(
        "--max-body",
        type=int,
        default=72,
        metavar="N",
        help="Maximum body line length (default: 72)",
    )
    args = parser.parse_args()

    if args.max_title < 1 or args.max_body < 1:
        print("--max-title and --max-body must be positive", file=sys.stderr)
        sys.exit(2)

    if args.file:
        try:
            message = Path(args.file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Cannot read {args.file}: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        message = args.message.replace("\\n", "\n")

    errors = validate(message, args.max_title, args.max_body)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
