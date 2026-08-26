# Reorganizing Existing Knowledge

Read when project knowledge already exists scattered — mixed structures,
stray notes, agent docs in odd places — before seeding anything new.

## Sequence

1. **Inventory.** Find every agent-facing knowledge artifact: notes files,
   doc folders, comments-turned-docs, wiki exports. Record path and topic.
2. **Deduplicate.** Where two artifacts state the same fact, keep the one
   that is current and delete the other. Two copies of a fact is a future
   contradiction.
3. **Converge on the chosen structure.** Move content into the single
   structure fixed in the workflow — flat files per concern or per-topic
   folders — renaming to match its conventions. Convert prose to
   agent-first style as it moves: load condition at top, facts first, one
   concern per file.
4. **Keep reachability during the move.** Update the entrypoint's
   when-to-read table in the same change as each move; a moved-but-
   unregistered document is lost.
5. **Leave public files alone.** README-class and other public-convention
   files are not knowledge documents: never absorb them, never restyle
   them, at most point to them.

## What Not To Migrate

Drop, rather than migrate, anything stale (contradicts the code), any
transcript-like session notes with no durable fact, and anything whose only
content is restating what the code shows.
