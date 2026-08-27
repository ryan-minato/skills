---
name: scaffold-disposal
description: >-
  Disposable scaffold skill (delete after the harness is built): removes
  the disposable scaffold skills installed in this project — the
  temporary builders whose descriptions share this catalog's marker.
  Finds them by that marker, shows a dry-run listing, deletes only after
  fresh explicit user confirmation, and deletes itself last. Use when the
  project is scaffolded and verified and the user asks to remove, clean
  up, or dispose of the scaffold skills, or of the temporary builder
  skills installed for the project. A disposable builder from another
  catalog carries a different marker and is left to that catalog's own
  disposal skill. Not for durable project skills or any other files.
---

# Scaffold-Skill Disposal

This skill removes the disposable scaffold skills from the project once
the project is scaffolded and verified. Identification is by description,
never by name or directory — installers rename skills, so the only stable
key is the marker every disposable scaffold skill's description starts
with:

```text scaffold-skill-marker
Disposable scaffold skill (delete after the harness is built):
```

Disposable builders published under another catalog reserve a different
marker and are deliberately out of reach here; see the last gotcha.

## Workflow

1. Dry run: `python3 scripts/dispose.py`. It reads the marker from this
   skill's own description, enumerates `<skill-root>/<name>/SKILL.md` one
   level below each root (default: the root this skill is installed in;
   add `--root <dir>` for every other skill directory the project's
   frameworks use), and lists the path, name, and first description line
   of every skill whose resolved description starts with the marker.
   Unparsable frontmatter is skipped and reported, never guessed;
   symlinked entries are reported, not followed. Record the confirmation token
   printed after the exact listing.
2. Show the listing to the user verbatim and ask for fresh, explicit
   confirmation. An earlier "scaffold the project" request is not consent
   to delete. If anything in the listing looks wrong — a durable skill
   matched, a scaffold skill missing — stop and investigate before
   deleting anything.
3. On confirmation, run the same command plus
   `--delete --confirm <token>`. The script rescans and refuses deletion if the
   roots, marker, matches, or skipped entries differ from the confirmed dry run.
   It deletes every matched directory and this skill's own directory last. If
   another deletion fails, it retains itself so the recovery procedure remains
   available.
4. Relay the report to the user, including every skipped or unparsable
   entry, so nothing disappears silently and nothing lingers unnoticed.

Done when: the user confirmed the exact dry-run listing, every matched
directory including this one is gone from every root that was scanned,
and the report was shown.

## Gotchas

- Never match by name, directory, or "looks like scaffolding" — the
  description prefix is the only key, and widening it deletes durable
  skills.
- Only one topic scaffold is normally installed, so a listing with a
  single builder beside this skill is the expected shape, not a sign the
  scan missed something.
- If the script cannot run, do the same steps manually: enumerate
  `SKILL.md` files one level below each skill root, parse each
  frontmatter, list those whose resolved description starts with the
  marker shown above, get fresh confirmation on that listing, delete the
  matches, this skill last, and report — still skipping anything
  unparsable.
- Frameworks sometimes mirror skills through symlinks: delete the real
  directory, then remove any dangling links the deletion leaves behind.
- Skills installed through a plugin manager (a plugin marketplace, a
  package manager) live in manager-owned caches: uninstall them through
  that manager instead of deleting files, or the manager will restore or
  mis-track them. File deletion is only for skills copied into the
  project's own skill directories.
- A project can install more than one disposable catalog, each reserving
  its own marker. This skill matches only the marker above; a skill
  carrying a different disposable marker is neither deleted nor listed as
  skipped, so it leaves no trace in the report. After this run, check the
  scanned roots for skills whose description still opens with a
  `Disposable ... (delete after the harness is built):` sentence and run
  the disposal skill shipped with that catalog.
