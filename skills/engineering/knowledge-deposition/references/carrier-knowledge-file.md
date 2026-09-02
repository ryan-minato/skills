# Carrier: Knowledge-Base File

Load this when the chosen carrier is a knowledge-base file.

## Place the file

Put it in the directory the probe found — where the project's other
agent-facing knowledge already lives — named in the same style as its
neighbors. Extend an existing file instead of creating a new one when the
new knowledge is consulted at the same moment as that file's content: one
lookup, one file. Create a new location only when the project has none;
then choose the lightest spot beside the entrypoint (a single `docs/` or
knowledge folder) rather than inventing a tree.

## Shape the file

- Open with the load condition — one line saying when an agent should be
  reading this file.
- Front-load the facts; an agent skims top-down and may stop early.
- One concern per file. A second, unrelated lesson gets its own file (or
  its own section of a genuinely shared concern), not a paragraph appended
  here.
- Keep the reason with each rule as one clause wherever the rule leaves
  judgment room.

## Record update triggers inside the file

Name, inside the file, the events that would make it wrong — "revisit if
the deploy command changes", "stale once the API v3 migration lands", "the
team renegotiates this each quarter". A future agent then knows when to
verify instead of trust.

## Keep one source of truth

If a human-facing document (README, contributing guide, wiki) already
states part of this knowledge, do not create a silent second copy. Pick
one document as the source of truth and have the other defer to it
explicitly — a sentence naming which file wins. Two unranked statements of
one rule is guaranteed drift.

## Register the pointer

A file nothing reaches does not exist. Add one line to the entrypoint (or
to the existing document agents pass through on the way to this topic):

- Event-triggered wording: `Read <path> before <operation>.` or
  `Read <path> when <condition>.` — the trigger names the moment from the
  finding, a real event an agent recognizes mid-task.
- Never "see `<path>` for details": without a firing condition the pointer
  either always loads or never fires.
- The entrypoint keeps only that one line. Restating the rule beside the
  pointer creates the second source of truth the previous section forbids.
