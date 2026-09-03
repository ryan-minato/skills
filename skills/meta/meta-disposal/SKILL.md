---
name: meta-disposal
description: >-
  Disposable builder skill (delete after the harness is built): removes the
  temporary builder skills installed for this project — every skill of the
  `meta` and `scaffold` catalogs, found by the shared marker their
  descriptions open with. Shows a dry-run listing, deletes only after the
  user has decided to delete and confirmed the exact listing, deletes itself
  last, and leaves the repository ready for review. Use when a builder's
  closing step hands off here after the user decided to remove the builders,
  or when the user asks to remove, clean up, or dispose of the meta or
  scaffold skills, the temporary builders, or the harness-building skills.
  Not for durable project skills or any other files, and not before the user
  has decided that the builders should go.
---

# Builder Disposal

Removes the disposable builders of both catalogs once the harness is built
and verified and the user has decided to delete them. Identification is by
description, never by name or directory — installers rename skills, so the
only stable key is the marker every disposable builder's description starts
with:

```text disposable-marker
Disposable builder skill (delete after the harness is built):
```

This skill is loaded after the decision, not to obtain it. If the
conversation holds no explicit decision to delete the builders, stop and ask
the user first; a request to build the harness was never consent to delete.

## Workflow

1. Confirm nothing is still pending: every builder handed off to has
   finished or was declined, the harness is verified, and the work has not
   yet been presented for review.
2. Dry run: `python3 scripts/dispose.py`. It reads the marker from this
   skill's own description, enumerates `<skill-root>/<name>/SKILL.md` one
   level below each root (default: the root this skill is installed in;
   add `--root <dir>` for every other skill directory the project's
   frameworks use), and lists the path, name, and first description line
   of every skill whose resolved description starts with the marker.
   Unparsable frontmatter is skipped and reported, never guessed;
   symlinked entries are reported, not followed. Record the confirmation
   token printed after the exact listing.
3. Show the listing to the user verbatim and ask them to confirm it. If
   anything in the listing looks wrong — a durable skill matched, a builder
   missing — stop and investigate before deleting anything.
4. On confirmation, run the same command plus
   `--delete --confirm <token>`. The script rescans and refuses deletion if
   the roots, marker, matches, or skipped entries differ from the confirmed
   dry run. It deletes every matched directory and this skill's own
   directory last. If another deletion fails, it retains itself so the
   recovery procedure remains available.
5. Clean up what the build left behind: remove the builders' entries from
   `$(git rev-parse --git-path info/exclude)`, delete any dangling symlinks
   the deletion leaves in mirrored skill directories, and if a deleted
   directory was tracked, stage its removal so the deletion lands as one
   clean commit.
6. Relay the report to the user, including every skipped or unparsable
   entry, so nothing disappears silently and nothing lingers unnoticed.
   Then confirm the entrypoint still reaches every deposited rule with no
   builder present, and hand the work to the project's review step.

Done when: the user decided and confirmed the exact dry-run listing, every
matched directory including this one is gone from every scanned root, the
exclude entries and dangling links are gone, and the report was shown.

## Gotchas

- Never match by name, directory, or "looks like scaffolding" — the
  description prefix is the only key, and widening it deletes durable
  skills.
- If the script cannot run, do the same steps manually: enumerate
  `SKILL.md` files one level below each skill root, parse each
  frontmatter, list those whose resolved description starts with the
  marker shown above, get confirmation on that listing, delete the
  matches, this skill last, and report — still skipping anything
  unparsable.
- Skills installed through a plugin manager (a plugin marketplace, a
  package manager) live in manager-owned caches: uninstall them through
  that manager instead of deleting files, or the manager will restore or
  mis-track them. File deletion is only for skills copied into the
  project's own skill directories.
- The normal listing is every `meta` builder plus at most one `scaffold`
  topic builder; a single scaffold entry is the expected shape, not a sign
  the scan missed something. This skill ships with the `meta` catalog, so
  a project that installed only a scaffold builder gets it when that
  builder's closing step installs the whole `meta` catalog.
