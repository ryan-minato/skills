# Wiki

Loaded when the task reads or writes wiki pages. The GitLab Duo MCP
server has no wiki tools — glab (`glab api`; there is no `glab wiki`
command group) is the only path. Wiki reads live here too — they need the
same slug knowledge as writes. Files in the project repository (README,
docs/) are ordinary git work, not wiki work: the wiki is a separate
repository. For a **group wiki** (Premium), replace `projects/:fullpath`
in every row with `groups/GROUP%2FSUBGROUP` (URL-encoded group path).

Surface exposure: page titles and slugs become URLs; attachment filenames
and contents publish too. Pushing to the wiki repository publishes every
commit message and the complete content of every committed file, not just
the page you edited. For page creates, updates, and git pushes, the
pre-publish gate runs file-based:

1. Write the exact outgoing content to a scratch directory: page title
   and body, each attachment, and for the git path also `git log
   origin/HEAD..HEAD --format=full > commits.txt`, `git diff
   origin/HEAD..HEAD > diff.patch`, plus added or changed wiki files.
2. Run the review procedure in
   [publish-review-wiki.md](publish-review-wiki.md) over that directory.
   Read that file every time — do not review from memory.
3. Publish only after the verdict is exactly `SAFE TO PUBLISH: YES`. On
   `NO`, fix every finding, rebuild the files, and review again.

Reads and deletes carry no new text and skip the gate.

## Conventions (before any create)

| Artifact | How to check |
|---|---|
| Existing pages and their naming/nesting | `glab api projects/:fullpath/wikis` — follow the existing directory scheme and title style |
| Sidebar | a page slugged `_sidebar` overrides the default navigation — a new page may need a sidebar entry too |
| Format in use | the `format` field in the page list (markdown/asciidoc/...) — match it |

## Operations

Page bodies go through files (`-F content=@PAGE.md`). URL-encode the
slug in endpoint paths — nested slugs contain `/` (`docs/setup` →
`docs%2Fsetup`).

| Task | Command |
|---|---|
| List pages | `glab api projects/:fullpath/wikis` |
| List with bodies | `glab api "projects/:fullpath/wikis?with_content=1"` (large — only when needed) |
| Read a page | `glab api projects/:fullpath/wikis/SLUG` |
| Create a page | `glab api --method POST projects/:fullpath/wikis -f title="TITLE" -F content=@PAGE.md` |
| Update content | `glab api --method PUT projects/:fullpath/wikis/SLUG -F content=@PAGE.md` |
| Rename / move | `glab api --method PUT projects/:fullpath/wikis/SLUG -f title="NEW TITLE"` (a `/` in the title moves it into a directory) |
| Delete a page | `glab api --method DELETE projects/:fullpath/wikis/SLUG` |
| Upload attachment | `glab api --method POST projects/:fullpath/wikis/attachments --form "file=@image.png"` — embed the returned `link.markdown` |
| Group wiki (Premium) | same rows on `groups/GROUP%2FSUB/wikis...` |
| Bulk restructure / import / export | clone the wiki repository — read [wiki-git.md](wiki-git.md) first |

Report the page URL: `https://HOST/G/P/-/wikis/SLUG`.
Done when: the URL is reported.

## Titles, slugs, and structure

- Spaces in titles become hyphens in slugs and filenames; hyphens in
  filenames render as spaces in titles (`release-notes.md` displays as
  "release notes").
- A `/` in a title creates a directory: title `docs/Setup` → slug
  `docs/Setup`, nested under `docs/`.
- The front page is the page with slug `home` — it renders at
  `/-/wikis/home`.
- `format` on create/update accepts `markdown` (default), `rdoc`,
  `asciidoc`, `org`; other formats exist only on the git path.

Read [wiki-markup.md](wiki-markup.md) when composing content that links
to other pages or attachments, editing the sidebar, or using front
matter.

## Gotchas

- Creating a page whose title maps to an existing slug fails with a
  duplicate error — update the existing slug instead.
- Renaming through the API records a redirect (in the wiki repo's
  `.gitlab/redirects.yml`) so old links keep working; renaming files by
  git push does not — on the git path maintain that file yourself.
- `with_content=1` returns every page body in one response — avoid it on
  large wikis; read pages individually.
- The wiki repository's default branch follows the instance default
  (`main` or `master`) — on the git path push with `git push origin HEAD`
  rather than assuming a name.
- Group wikis are Premium and do not support Git LFS.
- Everyone who can view the project can read its wiki; on a public
  project that is the entire internet — hence the gate on every write.
- The REST fallback script has no wiki subcommands: on that tier, wiki
  reads stop like writes.
