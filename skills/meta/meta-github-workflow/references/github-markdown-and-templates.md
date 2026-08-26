# GitHub Markdown, Forms, and Templates

Read when creating or editing issue forms, pull-request templates,
GitHub-rendered prose, or wiki content.

## Issue forms and templates

Issue forms (`.github/ISSUE_TEMPLATE/*.yml`) are the intake enforcement
GitHub offers: `required:` fields actually block submission, and the
top-level `name`, `description`, `labels:`, `assignees:`, and `type:` keys
apply metadata deterministically — but a label or type that does not exist
is **dropped silently**, which is why the taxonomy check exists. Forms
render each element as a `### <label>` heading in the final body; any
body-scanning automation depends on those headings, so register that edge.
Resolve the current form schema from first-party docs when authoring —
forms are still marked preview and the schema moves.

Templates are listed alphanumerically, YAML before Markdown: prefix
filenames (`01-bug.yml`) to control order. `config.yml` sets
`blank_issues_enabled` and `contact_links` — the Discussions and security
routing — and even with blank issues disabled, write-access users still
see a maintainers-only blank option. An organization `.github` repository
supplies default templates only when the target repository has **none of
its own**: one local file disables the entire org default folder.

**Non-interactive creation ignores templates entirely** — `gh issue
create`, the REST API, and MCP capabilities all skip them. The durable
project skill must carry the body-construction procedure: read the form,
mirror its `### <label>` headings per answered element, apply the form's
labels and type explicitly in the same call.

## Pull-request templates

One default template (`.github/pull_request_template.md` or under
`.github/PULL_REQUEST_TEMPLATE/`). Multiple PR templates have **no chooser
UI** — they are reachable only through `?template=NAME.md` query
parameters, so ship one default unless the project will genuinely
distribute those links (in CONTRIBUTING or bot comments). The checklist
workflow parses the template's own `## ` headings; editing the template and
the workflow is one synchronized change.

## GitHub-flavored markdown

Resolve current syntax from first-party docs rather than memory. Stable
facts worth designing around: five alert types (`> [!NOTE]`, `[!TIP]`,
`[!IMPORTANT]`, `[!WARNING]`, `[!CAUTION]`) that cannot nest; mermaid,
geoJSON/topoJSON, and STL diagrams render in issues, PRs, discussions,
wikis, and files; math via `$...$`/`$$...$$`; footnotes work in most
surfaces but **not in wikis**; collapsible `<details>` sections; task
lists render checkboxes, and issue references in them unfurl — but
tasklist *blocks* are retired and must never be emitted. `#N` autolinks
share one number space across issues and PRs. Custom autolinks to external
trackers are a repository setting worth recording when present.

## Wiki and docs placement

Documentation defaults to the repository — `docs/` plus optional Pages —
because wikis have no page hierarchy (a hand-maintained `_Sidebar` fakes
one), a 5,000-file soft limit, no footnotes, no PR review, and are
plan-gated on private repositories. A wiki is an opt-in for
low-ceremony collaborator notes only; agent-critical rules always live in
the repository, where a checkout can see them.

Done when: every shipped form validates against the current schema, every
template's metadata references only labels and types that exist, the
body-construction procedure is in the durable skill, and no agent-critical
rule lives only in a wiki.
