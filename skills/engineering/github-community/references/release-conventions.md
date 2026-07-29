# Release conventions: versioning policy, release.yml, tag CI

Loaded when the task standardizes releases. Inputs from "Assess the
project first": existing tags and their shape, existing releases and
notes style, an existing `.github/release.yml`, whether the label
taxonomy exists, and whether commits follow Conventional Commits
(type-mapped bump rules only work then).

## Versioning and tag policy

Copy [assets/versioning-policy.md](assets/versioning-policy.md) to
`docs/versioning-policy.md` (or the project's docs location) and settle
every `{{...}}` placeholder **with the user**: SemVer is the default;
when the commit-conventions domain is in place, keep the bump table keyed
to commit types (breaking → major, feat → minor, else patch), otherwise
replace it with the manual bump rule; fix the tag format (default
`vMAJOR.MINOR.PATCH`, regex `^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z]+\.[0-9]+)?$`)
and record whether tags must be annotated or signed.

Done when: the policy doc has no `{{...}}` left and matches the tag
scheme the assessment found (or the user approved the change).

## Notes configuration

Copy [assets/release-config.yml](assets/release-config.yml) to
`.github/release.yml` and map its `categories:` labels to the labels the
repository actually has — every referenced label must exist, or PRs fall
into the catch-all silently (the same silent-drop hazard as issue
forms). When the label taxonomy is missing, run the issue-conventions
domain first — this file keys on it. Also copy
[assets/release-notes-template.md](assets/release-notes-template.md)
next to the policy doc for the hand-written-notes path.

Read [release-yml-schema.md](release-yml-schema.md) when editing the
config beyond the shipped categories and a key or matching rule is
uncertain.

Done when: every label in `.github/release.yml` exists in the
repository.

## Tag check in CI

Copy [assets/workflow-tag-check.yml](assets/workflow-tag-check.yml) to
`.github/workflows/tag-check.yml` and set both its trigger pattern and
its regex to the policy's tag format (the trigger polices release tags
only, so convenience tags like `latest` are not failed). It runs on tag
pushes, validates the tag name, and verifies the tagged commit is
reachable from the default branch — a wrong-format or off-branch tag
fails the run with a fix-it message before anyone releases from it.
First-party actions only; the checks are plain `run:` steps.

Done when: the workflow parses and its regex equals the policy doc's.

## Automation beyond the check

Read [release-automation.md](release-automation.md) when the user opts
into release automation (auto-draft on tag push, artifact attachment).

## Generate the project-level skill

For the default deliverable, copy
[assets/project-skill-releases.md](assets/project-skill-releases.md) to
`<skills-dir>/<repo-name>-releases/SKILL.md` and fill every
`{{PLACEHOLDER}}`:

| Placeholder | Fill with |
|---|---|
| `{{REPO_NAME}}` / `{{OWNER_REPO}}` | From the origin remote |
| `{{POLICY_DOC_PATH}}` | Where the versioning policy doc was installed |
| `{{TAG_FORMAT}}` / `{{TAG_REGEX}}` | From the policy doc |
| `{{BUMP_RULE}}` | The policy doc's bump rule, one sentence |
| `{{NOTES_RULE}}` | The notes rule (generated via release.yml, or the template path) |
| `{{NOTES_RULE_SHORT}}` | The same as one imperative clause |
| `{{EXTRA_CREATE_FLAGS}}` | Extra `gh release create` flags the policy implies (for example `--verify-tag` for annotated tags), or empty |

The template pre-wires the draft-first flow, the repository's tag rule,
and the condensed pre-publish gate. For the AGENTS.md fallback, copy
[assets/agents-md-releases-section.md](assets/agents-md-releases-section.md)
into the project's `AGENTS.md` and fill the same placeholders (it uses a
subset).

Done when: the generated deliverable contains no `{{...}}` placeholder
and (for a skill) its frontmatter `name` matches its directory name.

## Gotchas

- `.github/release.yml` affects only `--generate-notes` and web-generated
  notes — it does not validate tags, gate releases, or touch manual
  notes.
- Categories match by PR **labels**, never by commit messages — an
  unlabeled PR lands in the `*` catch-all category regardless of its
  commit types.
- The order of `categories:` entries is the order sections render in;
  the `*` exclude catch-all must come last or it swallows later
  categories.
- The tag-check workflow validates tags after they are pushed; it cannot
  prevent the push. Pair it with the generated skill (which picks
  compliant names up front) rather than relying on CI alone.
