# Retrofitting An Existing Skill

Read when an existing project skill misfires, went stale, or triggers too
broadly. Diagnose in this order; fix the first failing layer before
touching the next.

## 1. Trigger Surface

- Loads for unrelated work → the description promises too much. Rewrite it
  as concrete user requests plus the project condition; add a "Not for"
  clause naming what it must ignore.
- Never loads when needed → the description names internals instead of
  intent. Rewrite it in the words a user would actually say.

## 2. Body Currency

Check every command, path, and claim against the project as it is now.
Fix or delete stale lines — a stale instruction followed with confidence
is the worst outcome a skill can produce. If the whole procedure changed,
rewrite from the matching skeleton rather than patching line by line.

## 3. Shape

- Two unrelated triggers in one skill → split it; each half gets its own
  tight description.
- Two skills covering one procedure → merge them; the seam between them is
  where agents fall through.
- A body grown past what every invocation needs → move the rarely needed
  branches into the skill's own references with load conditions.

## 4. Update Trigger

If the skill decayed once, its update trigger is missing or unenforced.
Add a project-visible rule naming what change obliges revising the skill,
and hand it to the harness's sync mechanism.
