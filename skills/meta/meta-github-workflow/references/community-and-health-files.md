# Community, Discussions, and Health Files

Read when the harness includes Discussions routing, CONTRIBUTING,
SECURITY, SUPPORT, CODE_OF_CONDUCT, GOVERNANCE, FUNDING, or the account
`.github` repository.

## Discussions are half of intake

Issues are actionable work; Discussions are questions, ideas, and support.
When Discussions are enabled, design the categories (Q&A with answer
marking, Announcements, Ideas) alongside the issue forms, route non-work
traffic there through `config.yml` contact links, and make
issue-to-discussion conversion a normal triage move. Note the tooling
seam: `gh` has no discussion command group and MCP discussion toolsets are
often off by default — operations go through `gh api` (GraphQL), which the
durable skill must say.

## Health files

GitHub recognizes CODE_OF_CONDUCT, CONTRIBUTING, GOVERNANCE, SECURITY,
SUPPORT, FUNDING.yml, issue templates and their `config.yml`, PR
templates, and discussion category forms, resolved per file from
`.github/` → root → `docs/`, then the account's `.github` repository —
which must be **public** to serve defaults, and whose template defaults
are all-or-nothing against local files. LICENSE cannot be defaulted; it
lives in every repository. The community profile checklist is the
completeness gauge for public repositories.

Per-file guidance:

- **CONTRIBUTING** encodes the real flow this harness built — setup,
  checks, intake, branching, early draft PRs, review, merge method — not
  aspirations. It is the one place multiple-PR-template links can live.
- **SECURITY** points at private vulnerability reporting on public
  repositories, a monitored private channel on private ones (see
  [security-and-ownership.md](security-and-ownership.md)).
- **CODE_OF_CONDUCT** adopts an established text with attribution and a
  real enforcement contact.
- **GOVERNANCE** only when decision rights genuinely need a public
  contract; **SUPPORT** routes to Discussions or the real channel;
  **FUNDING.yml** only with real accounts.
- `good first issue` and `help wanted` feed the repository's `/contribute`
  onboarding page — label them deliberately as part of community design,
  not as leftovers.

Public statements and the internal workflow are one system: when the
harness changes the flow, CONTRIBUTING changes in the same pull request
(a registered synchronization edge).

Done when: every published health file matches the built harness, routing
sends questions to Discussions and vulnerabilities off the tracker, and
nothing promises a process the repository does not run.
