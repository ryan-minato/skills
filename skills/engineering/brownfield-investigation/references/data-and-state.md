# Data and State

Read when the question involves persisted data, a state lifecycle, data
ownership, or state shared across components.

## Procedure

1. **Stores.** List every place state lives: databases, caches, queues,
   files, object storage, search indexes, and external systems of record.
2. **Structure.** For each table or collection in scope, take the structure
   from the live schema when it is reachable and reading it is permitted;
   otherwise from migrations and models that agree, noting that the live
   schema was not checked. Report every disagreement between them as
   CONTRADICTED, with the live schema winning when it was read.
3. **Writers and readers.** Record which components write and which read
   each table or field, with the code locations.
4. **Ownership.** One writer found in the repository is a lead, not
   ownership: another system may write the store or be its system of
   record, and a search of this repository cannot rule that out. Record
   the ownership as inferred until a person, a decision record, or the
   store's access grants confirm it; with nothing to go on, record it as
   UNKNOWN. A store written by several components, or by a system outside
   the repository, is a contract boundary: its shape is something another
   party may depend on.
5. **Lifecycle.** How records are created, updated, and deleted (soft or
   hard), how long they are kept, and which invariants hold — and where
   each invariant is enforced: a database constraint, code, or both.
6. **Derived and cached state.** What is computed from what, and how caches
   are invalidated.

## Depth

| Depth | Covers |
|---|---|
| ORIENT | the main stores and each one's likely owner, marked inferred unless confirmed |
| ESTABLISH | tables and fields in scope, with writers, readers, and constraints |
| EXHAUSTIVE | every field in scope, including null versus missing semantics, retention, and the migrations that shaped it |

## Gotchas

- Migrations and the live schema drift apart; manual fixes in production
  never appear in migrations.
- Models omit tables they do not use, and legacy tables with no model are
  still read by someone. List them as found in the schema.
- An invariant enforced only in code is broken by any writer that bypasses
  that code, including other systems sharing the database.
- Null, empty, and absent can mean three different things to consumers;
  record which one each writer produces.
- Time zones and encodings stored implicitly are part of the data's
  meaning.
