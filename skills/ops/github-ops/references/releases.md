# Releases and tags

Loaded when the task cuts, edits, reads, or deletes releases or tags.

Surface exposure: publishing a release publishes the notes, the tag, and
every asset, and notifies watchers — the draft step exists so the
pre-publish gate always runs before the release goes live. The gate runs
once on the assembled draft — title, notes file, tag name, and the asset
file list — before `--draft=false`. Draft creation itself is low-exposure
(drafts are visible to collaborators only), but genuinely sensitive
content must not even enter a draft. Long or generated notes, assets, and
screenshots are reviewed from disk; read
[publish-review.md](publish-review.md) when the assembled release is too
bulky to inspect inline.

## Create a release (draft-first)

1. **Pick the version.** Follow the project's policy; mechanical bumps:
   `python3 scripts/next_version.py --bump patch` (or `minor`/`major`;
   `--pre rc` for a prerelease; `--latest vX.Y.Z` when local tags are not
   the source of truth) — script:
   [scripts/next_version.py](scripts/next_version.py). The output is
   `TAG` below.
2. **Decide tag creation.** Default: let GitHub create the tag —
   `--target BRANCH` pins what it points at. If the project requires
   annotated or signed tags: `git tag -a TAG -m "MSG"` (or `-s`), `git
   push origin TAG`, then add `--verify-tag` to the create call.
3. **Write the notes** to `NOTES.md` per the decision table below.
4. **Create the draft** (assets optional, `path#Display label` form):

       gh release create TAG -R O/R --draft --title "TITLE" \
         --notes-file NOTES.md [--target BRANCH] [--prerelease] \
         [--verify-tag] [ASSET_FILES...]

5. **Run the gate** over title, NOTES.md, tag name, and the asset list.
6. **Publish:** `gh release edit TAG -R O/R --draft=false [--latest]`.
7. Close the matching milestone if one exists (see
   [planning.md](planning.md) for the how); report the release URL.

Done when: the release URL is reported and `gh release view TAG -R O/R
--json isDraft` shows `false`.

## Notes

| Situation | Path |
|---|---|
| `.github/release.yml` exists, or the user wants generated notes | Add `--generate-notes` to the create call (with `--notes-start-tag PREV` for a non-adjacent range); to curate before the gate, use the preview endpoint below, edit, then pass `--notes-file NOTES.md` |
| Hand-written notes | Compose NOTES.md from `git log PREV..HEAD --oneline` grouped by kind, matching the prior notes style (`gh release view -R O/R --json tagName,body -q .body`) |
| Notes from an annotated tag message | `--notes-from-tag` on the create call |

What `--generate-notes` does: GitHub builds the notes from **merged pull
requests** between the previous release and the new tag — one line per PR
(title, author, PR link), a first-time-contributors section, and a
full-changelog compare link. It keys on PRs, not commit messages — direct
pushes without a PR do not appear.

When the repository has `.github/release.yml`, the generator groups PR
lines into its `changelog.categories` by PR **label** and hides anything
matched by `changelog.exclude`. The label taxonomy must actually be
applied to PRs, or everything lands in the `*` catch-all category.
Authoring or changing that file is `github-community` work — do not edit
it as a side effect of cutting a release.

The generator diffs from the previous release tag by default. When
releasing from a maintenance branch or after deleting releases, pin the
start explicitly with `--notes-start-tag PREV_TAG`.

Preview without creating anything — the REST endpoint returns the
generated name and body as JSON, useful to curate into `NOTES.md` before
any draft exists (the tag does not need to exist yet):

```bash
gh api -X POST repos/O/R/releases/generate-notes \
  -f tag_name=TAG [-f previous_tag_name=PREV] -q .body > NOTES.md
```

Mixing generated and hand-written content: generate first (preview
endpoint), then edit `NOTES.md` — add a summary paragraph on top, trim
noise, keep the PR list. The gate reviews the final file; generated text
gets no exemption, since PR titles can carry anything their authors wrote.

## Read, list, edit, delete

| Task | MCP capability | gh command |
|---|---|---|
| List releases | list releases | `gh release list -R O/R --limit 20` |
| Read one / latest | read a release by tag / the latest release | `gh release view TAG -R O/R` (omit `TAG` for latest) |
| List tags | list a repository's tags | `gh api repos/O/R/tags --jq '.[].name'` |
| Edit title/notes/flags | — | `gh release edit TAG -R O/R [--title "T"] [--notes-file NOTES.md] [--prerelease] [--latest]` (gate on text changes) |
| Upload / replace assets | — | `gh release upload TAG FILE... -R O/R [--clobber]` |
| Delete an asset | — | `gh release delete-asset TAG ASSET_NAME -R O/R --yes` |
| Delete a release | — | `gh release delete TAG -R O/R --yes [--cleanup-tag]` — confirm with the user first |

Every release **write** is a `—` row: MCP covers reads only. On the REST
fallback tier, the reads map to `rest_read.py releases` (`--tag TAG` /
`--latest`) and `tags` — see [rest-fallback.md](rest-fallback.md).

Read [release-recipes.md](release-recipes.md) when the release needs more
than the standard flow above (discussion category, releasing from a
non-default branch, checksums, republishing, downloads).

## Gotchas

- `gh release create` without an existing tag creates a **lightweight**
  tag at `--target` (default branch tip by default). Projects requiring
  annotated/signed tags must create the tag with git first (step 2) —
  `--verify-tag` makes gh abort if it is missing.
- Deleting a release does **not** delete its tag unless `--cleanup-tag`;
  a leftover tag makes a later re-create reuse the old commit.
- `--latest` is not automatic for the highest semver: GitHub marks the
  most recently *published* non-prerelease as latest unless `--latest` is
  set explicitly.
- A draft release's URL becomes invalid after publish (the final URL uses
  the tag) — report the post-publish URL, not the draft's.
- Prereleases (`--prerelease`) never become "latest" and are hidden from
  the latest-release API — users on `releases/latest` will not see them.
- Draft releases are invisible to anyone without push access — a release
  the user mentions but the tools cannot see is usually still a draft.
