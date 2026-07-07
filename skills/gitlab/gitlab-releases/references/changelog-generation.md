# Generating release notes from Changelog trailers

Load condition: the notes decision table chose generated notes, or you
need to control what the generator includes.

## What `glab changelog generate` does

GitLab builds the changelog from **commits carrying a Git trailer** —
`Changelog: <category>` by default (for example `Changelog: added`,
`Changelog: fixed`). Commits without the trailer are silently excluded.
The range defaults to "since the last tag matching the version scheme up
to HEAD".

```bash
glab changelog generate --version X.Y.Z > NOTES.md
```

Useful controls:

```bash
glab changelog generate --version X.Y.Z \
  --from PREV_SHA --to HEAD_SHA \      # explicit range (--from is excluded)
  --config-file PATH \                  # non-default config location
  --trailer Changelog \                 # non-default trailer key
  --date 2026-08-01T00:00:00Z           # release date stamped in the heading
```

## Categories: `.gitlab/changelog_config.yml`

The config maps trailer **values** to section headings and sets template
and ordering. Trailer values are matched case-sensitively against the
`categories:` keys — a commit with `Changelog: Added` lands in the
default bucket if the key is `added`. Authoring or changing that file is
`gitlab-release-conventions` work — do not edit it as a side effect of
cutting a release.

## The REST endpoint variant

The same generator is exposed over REST, with one extra ability: it can
**commit** the generated section into a changelog file in the repository
instead of (or besides) returning it.

```bash
# Return the notes as data (like the CLI):
glab api "projects/:fullpath/repository/changelog?version=X.Y.Z"

# Commit the section into CHANGELOG.md on BRANCH (a write — gate the
# generated text first by fetching it as data above):
glab api --method POST projects/:fullpath/repository/changelog \
  -f version=X.Y.Z -f branch=BRANCH -f message="Add changelog for vX.Y.Z"
```

## Mixing generated and hand-written content

Generate into `NOTES.md`, then edit: add a summary paragraph on top, trim
noise, keep the commit list. The pre-publish gate reviews the final file —
generated text gets no exemption, since commit subjects can carry
anything their authors wrote.
