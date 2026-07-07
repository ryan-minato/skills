# Release recipes — long tail

Operations beyond the SKILL.md tables. All rows are gh-only (MCP covers
release reads, nothing here). The pre-publish gate applies wherever text
or files go public.

## Open a discussion with the release

```bash
gh release create TAG -R O/R --draft --notes-file NOTES.md \
  --discussion-category "Announcements"
```

The category must already exist in the repository's Discussions settings;
the discussion is created when the release is published, not while it is
a draft.

## Release from a non-default branch or exact commit

```bash
gh release create TAG -R O/R --draft --notes-file NOTES.md \
  --target release/1.x        # branch name or full commit SHA
```

`--target` is ignored when the tag already exists — the tag wins.

## Checksums for assets

Generate and attach a checksum file so consumers can verify downloads:

```bash
(cd DIST_DIR && sha256sum *) > checksums.txt
gh release upload TAG checksums.txt -R O/R
```

## Republish or fix a botched release

- Wrong notes/title: `gh release edit TAG --notes-file FIXED.md` (gate
  first) — no need to delete.
- Wrong artifacts: `gh release upload TAG FILE --clobber` replaces by
  asset name; `gh release delete-asset TAG NAME --yes` removes.
- Wrong commit entirely: delete release **and** tag, then redo —
  `gh release delete TAG --cleanup-tag --yes`, recreate the tag on the
  right commit, create the release again. Consumers who already fetched
  the old tag keep it; announce the move in the notes.

## Skip releasing when nothing changed

`--fail-on-no-commits` on the create call aborts (exit non-zero) when the
target has no commits since the last release — useful in scheduled
release automation.

## Download assets for verification

```bash
gh release download TAG -R O/R -p 'PATTERN' -D SCRATCH_DIR
```

Read-only; useful to confirm what an existing release actually ships
before editing it.
