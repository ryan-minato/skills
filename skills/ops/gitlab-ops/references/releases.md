# Releases and tags

Loaded when the task cuts, edits, reads, or deletes releases or tags. The
GitLab Duo MCP server has no release tools — glab (with `glab api` for
gaps) is the only path.

Surface exposure: GitLab has no draft releases — creating a release
publishes the tag, name, notes, and asset links the moment the call
succeeds, so the pre-publish gate runs on the complete assembled release
**before** create: tag name, release name, `NOTES.md`, and the asset
file/link list. Long or generated notes and assets are reviewed from
disk. Reads and deletes carry no new text.

## Create a release

1. **Pick the version.** Follow the project's policy; mechanical bumps
   with [scripts/next_version.py](scripts/next_version.py):
   `python3 scripts/next_version.py --bump patch` (or `minor`/`major`;
   `--pre rc` for a prerelease; `--latest vX.Y.Z` when local tags are not
   the source of truth). The output is `TAG` below.
2. **Write the notes** to `NOTES.md` per the decision table below.
3. **Run the gate** over tag name, release name, `NOTES.md`, and the
   asset list. GitLab publishes on create — there is no draft to fix up.
4. **Create** (the tag is created at `--ref` if it does not exist;
   `--tag-message` makes it an annotated tag):

       glab release create TAG -R G/P --name "NAME" \
         --notes-file NOTES.md [--ref BRANCH_OR_SHA] \
         [--tag-message "MSG"] [--milestone "M"] [--no-close-milestone] \
         [ASSET_FILES...]

5. Report the release URL from the output.

Done when: the URL is reported and `glab release view TAG -R G/P` shows
the expected notes and assets.

## Notes

| Situation | Path |
|---|---|
| `.gitlab/changelog_config.yml` exists and commits carry `Changelog:` trailers | `glab changelog generate --version X.Y.Z > NOTES.md`, then curate before the gate |
| Hand-written notes | Compose NOTES.md from `git log PREV_TAG..HEAD --oneline` grouped by kind, matching the prior notes style (`glab release view -R G/P` for the latest) |

Read [changelog-generation.md](changelog-generation.md) when generating
notes (trailer semantics, category config, ranges, the REST changelog
endpoint that can also commit a CHANGELOG file — hazard marked there).

## Assets

| Task | Command |
|---|---|
| Upload files to an existing release | `glab release upload TAG -R G/P FILE1 'FILE2#Display label'` |
| Link an external URL as an asset | `glab release upload TAG -R G/P -a '[{"name":"NAME","url":"https://...","link_type":"package"}]'` |
| Delete an asset link | `glab release delete-asset TAG ASSET_NAME -R G/P -y` |

`link_type` is one of `other`, `runbook`, `image`, `package`. The same
`-a`/`--assets-links` JSON works on `release create`.

## Read, list, edit, delete

| Task | Command |
|---|---|
| List releases | `glab release list -R G/P` |
| View one / latest | `glab release view TAG -R G/P` (omit `TAG` for the latest) |
| List tags | `glab api "projects/:fullpath/repository/tags?per_page=20"` — tag objects embed full release notes; never fetch unbounded |
| Edit name/notes | `glab api --method PUT projects/:fullpath/releases/TAG -f name="N" -F "description=@NOTES.md"` — glab has no release-edit subcommand; gate on text changes |
| Delete a release | `glab release delete TAG -R G/P -y` — the git tag survives; confirm with the user first |
| Delete the tag too | after the delete: `glab api --method DELETE projects/:fullpath/repository/tags/TAG` |

On the REST fallback tier, the reads map to `rest_read.py releases`
(`--tag TAG` / `--latest`) and `tags` — see
[rest-fallback.md](rest-fallback.md).

Read [release-recipes.md](release-recipes.md) when the release needs more
than the standard flow above (standalone tag creation, generic package
registry uploads, publishing to the CI/CD catalog, release evidence,
released-at scheduling).

## Gotchas

- `glab release create` on an existing release **updates** it instead of
  failing, unless `--no-update` — convenient for fixing notes, dangerous
  when the tag name was a typo. Check `glab release list` first.
- Associating a milestone **closes it by default** when the release is
  created — pass `--no-close-milestone` unless closing is intended.
- `glab changelog generate` includes only commits carrying the
  `Changelog:` trailer (or the trailer set in the config) — commits
  without it are silently excluded; an empty changelog usually means the
  trailer convention is not in use, not that nothing changed.
- `glab release delete` keeps the git tag; recreating the release later
  reuses that tag's commit. Delete the tag explicitly when the whole
  version was wrong.
- Without `--tag-message`, a tag created by `release create` is
  lightweight; projects requiring annotated/signed tags should pass
  `--tag-message` or create the tag with git first and push it.
- A 404 on release endpoints usually means insufficient role (Developer+
  required) or a private project without access — not a tier gate;
  releases are a Free feature.
- `--released-at` in the future creates an "Upcoming" release — it is not
  shown as the latest until the timestamp passes; check the
  `upcoming_release` field before calling something "released".
