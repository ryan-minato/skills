# SECURITY.md

## Secrets and Credentials

- No credentials, API keys, tokens, or passwords anywhere in the repository.
- Pre-commit hooks (detect-secrets, gitleaks) run on every commit. Do not bypass or
  disable them.

**Alert handling — follow this sequence every time a hook fires:**

1. **Determine if it is genuine.** If the flagged value is a real credential or PII,
   fix the content before committing. Do not suppress a real finding under any
   circumstances.
2. **Confirm it is a false positive** with explicit justification before touching any
   suppression config.
3. **Choose the narrowest suppression scope:**
   - *Inline* (preferred — isolated single occurrence):
     - `detect-secrets`: `# pragma: allowlist secret` on the offending line.
     - `gitleaks`: a file-scoped `[allowlist]` entry targeting only that file or commit.
   - *Allowlist rule* (only when an entire pattern class is provably safe):
     - `gitleaks`: add a targeted regex with a path restriction to `.gitleaks.toml`.
     - `detect-secrets`: `detect-secrets scan --update .secrets.baseline` after review.
4. **Never bulk-add** patterns or bulk-update the baseline to silence multiple alerts
   at once. When the correct scope is unclear, ask the user rather than assuming.

## Personally Identifiable Information

- Skills must not include real PII in examples, templates, or assets.
- Use synthetic data in all samples: `user@example.com`, `555-0100`, `Jane Doe`.

`.gitleaks.toml` enforces these PII rules at every commit (extends built-in secret
detection via `[extend] useDefault = true`):

| Rule | Blocked | Allowed |
|---|---|---|
| Emails | Any real address | `*@users.noreply.github.com`, `*@noreply.gitlab.com`, `*@example.{com,org,net}` |
| IPv4 | Public routable addresses | RFC 1918 private, RFC 5737 documentation, loopback, well-known DNS resolvers |
| IPv6 | Public addresses | RFC 3849 prefix (`2001:db8::`), loopback (`::1`), well-known DNS resolvers |

## External Content

- Do not copy third-party code or documentation verbatim unless the license explicitly
  permits it.
- When a skill references external APIs, libraries, components, or services, record
  names, version constraints, and access requirements in the skill's
  `docs/specs/<name>.md` file. Add general tool links to `docs/REFERENCES.md`.
- This repository is intended for public release as a general-purpose skill library.
  Skills must not contain internal business logic, proprietary specs, or confidential
  runbooks. If a skill author believes internal content is genuinely necessary, they
  must obtain explicit user consent and document the full scope and rationale in the
  spec file before committing.

## Script Execution Safety

Scripts in `scripts/` and `skills/<name>/scripts/` must:

- **Non-interactive:** No TTY prompts, no `input()`, no `readline`. Agents run in
  non-interactive shells; any blocking prompt causes an indefinite hang.
- **Flags only:** Read all input from CLI flags or environment variables.
- **Output separation:** Structured output (JSON preferred) to stdout; diagnostics and
  errors to stderr.
- **Exit codes:** 0 = success, 1 = general error, 2 = bad arguments.

## Dependency Hygiene

- Python scripts declare dependencies via uv inline metadata (`# /// script` block).
  Do not rely on globally installed packages.
- Deno scripts use locked import specifiers.
- Do not introduce additional runtimes without updating `ARCHITECTURE.md` and
  `docs/QUALITY.md`.
