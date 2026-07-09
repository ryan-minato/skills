#!/usr/bin/env python3
"""Pre-commit safety gate for this repository.

The gate complements the git-commit skill and pre-commit hooks. It checks
that a commit has staged content, that the configured author email is an
anonymous GitHub/GitLab address unless explicitly overridden, and that added
lines do not contain obvious secrets or personal email addresses. It also
prints the harness layers touched by the staged files so the committer can
review whether the commit is atomic and belongs in the expected layer.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ANONYMOUS_EMAILS = (
    re.compile(r"^[^@\s]+@users\.noreply\.github\.com$", re.I),
    re.compile(r"^[^@\s]+@noreply\.gitlab\.com$", re.I),
    re.compile(r"^[^@\s]+@users\.noreply\.gitlab\.com$", re.I),
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
CREDENTIAL_RE = re.compile(
    r"(?i)\b(api[_-]?key|apikey|secret|token|passw(?:or)?d|passwd|credential)"  # pragma: allowlist secret
    r"\b\s*[:=]\s*[\"']?([^\"'\s,}]+)"  # pragma: allowlist secret
)
SUPPRESSION_RE = re.compile(
    r"pragma: allowlist secret|gitleaks:allow|detect-secrets:disable|#\s*nosec\b"
)

ALLOWED_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "users.noreply.github.com",
    "noreply.gitlab.com",
    "users.noreply.gitlab.com",
}


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(result.stderr.strip() or f"git {' '.join(args)} failed", file=sys.stderr)
        sys.exit(2)
    return result.stdout


def staged_files() -> list[Path]:
    output = git("diff", "--cached", "--name-only", "-z")
    return [Path(item) for item in output.split("\0") if item]


def classify(path: Path) -> str:
    parts = path.parts
    if parts[:1] == (".devcontainer",) or path.name in {".mcp.json"}:
        return "Environment"
    if parts[:2] == (".agents", "skills") or path.name == "AGENTS.md":
        return "Workflow constraints"
    if parts[:2] == (".agents", "knowledge") or path.name == "ARCHITECTURE.md":
        return "Information tools"
    if parts[:1] == ("scripts",) or path.name in {
        "justfile",
        ".pre-commit-config.yaml",
        ".gitleaks.toml",
        ".gitmessage",
    }:
        return "Quality/repository safety"
    if parts[:1] == (".github",):
        return "Workflow tools"
    if parts[:1] == ("skills",):
        return "Target/implementation constraints"
    return "Unclassified"


def is_anonymous_email(email: str) -> bool:
    return any(pattern.match(email) for pattern in ANONYMOUS_EMAILS)


def check_author_email(allow_private_email: bool) -> list[str]:
    email = git("config", "user.email").strip()
    if not email:
        return ["git config user.email is empty. Configure an anonymous email."]
    if is_anonymous_email(email):
        print(f"Author email: {email} (anonymous)")
        return []
    if allow_private_email:
        print(f"Author email: {email} (private email explicitly allowed)")
        return []
    return [
        "git config user.email is not a GitHub/GitLab anonymous address. "
        "Use a noreply address, or rerun with --allow-private-email only "
        "after the user explicitly approves private email use."
    ]


def looks_like_placeholder(value: str) -> bool:
    value = value.strip().strip("\"'")
    return (
        not value
        or value.startswith("${")
        or value.startswith("$")
        or value.startswith("<")
        or value in {"...", "REDACTED", "redacted", "changeme", "example"}
    )


def check_added_lines() -> list[str]:
    findings: list[str] = []
    current_file = ""
    for line in git("diff", "--cached", "--unified=0", "--no-ext-diff").splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        content = line[1:]
        if SUPPRESSION_RE.search(content):
            continue

        if PRIVATE_KEY_RE.search(content):
            findings.append(f"{current_file}: added private key marker")

        credential = CREDENTIAL_RE.search(content)  # pragma: allowlist secret
        if credential and not looks_like_placeholder(credential.group(2)):
            findings.append(
                f"{current_file}: added credential-like assignment "
                f"for `{credential.group(1)}`"
            )

        for email in EMAIL_RE.findall(content):
            domain = email.rsplit("@", 1)[1].lower()
            if domain not in ALLOWED_EMAIL_DOMAINS:
                findings.append(f"{current_file}: added personal email `{email}`")
    return findings


def print_layers(files: list[Path]) -> None:
    by_layer: dict[str, list[str]] = defaultdict(list)
    for path in files:
        by_layer[classify(path)].append(path.as_posix())

    print("Staged harness layers:")
    for layer in sorted(by_layer):
        print(f"- {layer}:")
        for path in sorted(by_layer[layer]):
            print(f"  - {path}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-private-email",
        action="store_true",
        help="allow the configured non-anonymous author email after user approval",
    )
    args = parser.parse_args(argv)

    files = staged_files()
    errors: list[str] = []
    if not files:
        errors.append(
            "No staged changes. Stage one atomic logical change before committing."
        )
    else:
        print_layers(files)

    errors.extend(check_author_email(args.allow_private_email))
    errors.extend(check_added_lines())

    if errors:
        print("\nCommit safety gate failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Commit safety gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
