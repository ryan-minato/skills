# Wiki, Templates, and GitLab Flavored Markdown

Read when creating or editing wiki pages, work-item/MR templates, comments,
release notes, or any content whose GitLab rendering or quick actions matter.

Resolve `GitLab Flavored Markdown`, `description templates`, `quick actions`,
`wikis`, and `wiki API` through `llms.txt` for the target instance.

## Author rendered content

- Use semantic headings, short paragraphs, accessible link text, alt text, and
  task lists whose completion semantics are intentional.
- Verify current support before using GitLab-specific references, alerts,
  description lists, collapsible sections, diagrams, math, media, or front
  matter. Markdown support differs among titles, descriptions, repository
  files, comments, and wikis.
- Preview the exact raw Markdown in GitLab or through a supported render API.
  A local CommonMark renderer is not evidence of GitLab rendering.
- Keep relative links correct for their surface. Wiki page and attachment
  paths do not necessarily behave like repository Markdown paths.

## Description templates

Templates are `.md` files in the target version's documented `.gitlab/`
directories and become selectable only from the default branch. Projects can
inherit group or instance templates; inspect those before adding duplicates.
Rework the provided task, issue, incident, and MR assets for the agreed project
semantics and delete all fill guidance that should not reach users.

Quick actions deliberately embedded in a template are executable metadata.
Verify their current syntax, permissions, and target labels/types; document
each one and test it after the template reaches the default branch. Unknown
labels or unsupported actions may fail silently.

## Wiki architecture

Decide whether knowledge belongs in the repository or wiki. Agent-critical
rules default to the repository because they must be available in a checkout.
Use the wiki for maintained human-facing operational or product knowledge only
when it has an owner, navigation, review workflow, and durable pointer from the
repository harness.

For bulk wiki work, use its Git repository when supported; preserve page
history, redirects, sidebars, attachments, and naming rules. Review the full
diff and outgoing commit metadata before push. After publication, read every
changed page back and inspect rendered links and navigation.
