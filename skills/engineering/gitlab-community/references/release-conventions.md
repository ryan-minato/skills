# Release conventions: versioning policy, changelog config, tag CI

Loaded when the task standardizes releases. Inputs from "Assess the
project first": existing tags and their shape, existing releases and
notes style, an existing `.gitlab/changelog_config.yml`, whether commits
carry `Changelog:` trailers, milestone usage, and existing CI job names.

## Versioning and tag policy

Copy [assets/versioning-policy.md](assets/versioning-policy.md) to
`docs/versioning-policy.md` (or the project's docs location) and settle
every `{{...}}` placeholder **with the user**: SemVer is the default;
when the commit-conventions domain is in place, keep the bump table
keyed to commit types (breaking → major, feat → minor, else patch),
otherwise replace it with the manual bump rule; fix the tag format
(default `vMAJOR.MINOR.PATCH`, regex
`^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z]+\.[0-9]+)?$`) and record whether
tags must be annotated (`--tag-message` on create) or signed (created
locally and pushed).

Done when: the policy doc has no `{{...}}` left and matches the tag
scheme the assessment found (or the user approved the change).

## Changelog configuration

Copy [assets/changelog-config.yml](assets/changelog-config.yml) to
`.gitlab/changelog_config.yml`. Its `categories:` keys are `Changelog:`
trailer **values** (case-sensitive); the mapped text is the rendered
section heading. Commits without the trailer are excluded from generated
changelogs entirely — when the project does not yet have the trailer
habit, run the commit-conventions domain first, or record the
manual-notes path as the project default instead. Also copy
[assets/release-notes-template.md](assets/release-notes-template.md)
next to the policy doc for the hand-written-notes path.

Read [changelog-config-schema.md](changelog-config-schema.md) when
editing the config beyond the shipped categories and a key is uncertain.

Done when: the config's categories match the trailer values the
convention actually uses (or the manual path is recorded).

## Milestone policy

Record in the policy doc whether each release gets a milestone titled
exactly like the version. When it does: `glab release create
--milestone "vX.Y.Z"` associates it and **closes it by default** — the
policy records whether that auto-close is wanted or the generated skill
must pass `--no-close-milestone`. Milestone lifecycle itself belongs to
the `gitlab-ops` planning domain.

## Tag check in CI

Copy the job in [assets/tag-check-job.yml](assets/tag-check-job.yml)
into `.gitlab-ci.yml` and align both its `rules:` tag pattern and its
`TAG_REGEX` with the policy (the rule polices release tags only, so
convenience tags like `latest` are not failed). The job runs in tag
pipelines, is tokenless, validates the tag name, and verifies the tagged
commit is reachable from the default branch. Validate the edited
`.gitlab-ci.yml` with `glab ci lint` when glab is authenticated.

Read [release-automation.md](release-automation.md) when the user opts
into more than the tag check — creating releases from tag pipelines
(release-cli) or attaching build artifacts. Note the hard constraint
there: GitLab has no draft releases, so CI-created releases publish
immediately and are appropriate only for fully mechanical notes.

Done when: the job is in `.gitlab-ci.yml`, parses, and its regex equals
the policy doc's.

## Generate the project-level skill

For the default deliverable, copy
[assets/project-skill-releases.md](assets/project-skill-releases.md) to
`<skills-dir>/<project-name>-releases/SKILL.md` and fill every
`{{PLACEHOLDER}}`:

| Placeholder | Fill with |
|---|---|
| `{{PROJECT_NAME}}` | Project name, lowercase, hyphens only |
| `{{PROJECT_PATH}}` / `{{GITLAB_HOST}}` | From the origin remote |
| `{{POLICY_DOC_PATH}}` | Where the versioning policy doc was installed |
| `{{TAG_FORMAT}}` / `{{TAG_REGEX}}` | From the policy doc |
| `{{BUMP_RULE}}` | The policy doc's bump rule, one sentence |
| `{{NOTES_RULE}}` | The notes rule (glab changelog generate, or the template path) |
| `{{NOTES_RULE_SHORT}}` | The same as one imperative clause |
| `{{MILESTONE_RULE}}` | The milestone policy, incl. the auto-close decision |
| `{{EXTRA_CREATE_FLAGS}}` | Extra `glab release create` flags the policy implies (`--tag-message` for annotated tags, `--no-close-milestone`), or empty |

For the AGENTS.md fallback, copy
[assets/agents-md-releases-section.md](assets/agents-md-releases-section.md)
into the project's `AGENTS.md` and fill the same placeholders (it uses a
subset).

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Gotchas

- `changelog_config.yml` affects only changelog **generation** (`glab
  changelog generate` and the REST changelog endpoint) — the release
  description is still whatever gets passed to `glab release create`.
- Trailer values match `categories:` keys case-sensitively — a commit
  with `Changelog: Added` lands nowhere when the key is `added`.
- GitLab has no draft releases: nothing this domain configures changes
  the fact that `glab release create` publishes immediately — the
  generated skill's gate runs before create, and CI must never create
  releases with unreviewed notes.
- The tag check validates tags after they are pushed; it cannot prevent
  the push. Pair it with the generated skill (which picks compliant
  names up front) rather than relying on CI alone.
