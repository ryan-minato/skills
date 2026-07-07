# Release recipes — long tail

Operations beyond the SKILL.md tables. All rows are glab/REST (the Duo
MCP server has no release tools). The pre-publish gate applies wherever
text or files go public.

## Create a tag without a release

glab has no `tag` command group; the repository tags endpoint covers it.
An annotated tag carries `message`.

```bash
glab api --method POST projects/:fullpath/repository/tags \
  -f tag_name=TAG -f ref=BRANCH_OR_SHA [-f message="MSG"]
glab api --method DELETE projects/:fullpath/repository/tags/TAG
```

Signed tags cannot be created over the API — create them locally
(`git tag -s TAG -m "MSG"`) and `git push origin TAG`.

## Upload assets into the generic package registry

Instead of attaching raw files to the release, store them in the
project's generic package registry and link them — versioned, listable,
and independent of the release object:

```bash
glab release upload TAG -R G/P --use-package-registry \
  [--package-name NAME] FILE1 FILE2
```

The same switch works on `release create`. Default package name:
`release-assets`.

## Publish a CI/CD component release to the catalog

For projects that are CI/CD catalog components (experimental flag):

```bash
glab release create TAG -R G/P --name "NAME" --notes-file NOTES.md \
  --publish-to-catalog
```

Requires the project to be set as a catalog resource; otherwise the flag
errors — tell the user instead of retrying.

## Schedule an upcoming release

```bash
glab release create TAG -R G/P --name "NAME" --notes-file NOTES.md \
  --released-at 2026-09-01T00:00:00Z
```

Until the timestamp passes, the release shows an "Upcoming" badge and is
not the latest release. The content is visible immediately — the gate
applies now, not at the release date.

## Release evidence (Premium badge for compliance)

GitLab collects release evidence automatically at create time on tiers
that support it; it is immutable afterwards. There is nothing to invoke —
but it is one more reason the gate runs before create, not after.

## Download a release's assets for verification

```bash
glab release download TAG -R G/P [-D SCRATCH_DIR]
```

Read-only; useful to confirm what an existing release actually ships
before editing it.
