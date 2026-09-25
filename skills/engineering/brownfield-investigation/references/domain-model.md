# Domain Model

Read when the question turns on domain terms, entities, states, or their
relationships, or when two parts of the code use one term differently.

## Procedure

1. **Glossary.** For each term a reader meets — in types, tables, API
   names, user-facing text, and documents — record its meaning in this
   codebase, where that meaning is defined, and its synonyms.
2. **Entities and relationships.** From types, schemas, and constraints,
   record the entities, their identifiers, and how they relate
   (cardinality, ownership, lifecycle dependence).
3. **States and transitions.** For each entity with a lifecycle, list its
   states and every transition: the trigger, the guard, the component that
   performs it, and where it is enforced (code, database constraint, both).
   Look beyond explicit status fields: a timestamp that becomes non-null or
   a row that moves tables is a state too.
4. **Conflicts.** Record every term used with two meanings, and every
   concept with two names, as a finding with both locations.

## Depth

| Depth | Covers |
|---|---|
| ORIENT | the core entities and the terms a newcomer meets first |
| ESTABLISH | all states and transitions of the entities in scope |
| EXHAUSTIVE | plus error, compensation, and administrative transitions, and who may trigger each |

## Output

A glossary, an entity list with relationships, and one state table per
entity with a lifecycle, each row sourced. A transition the code allows but
no document or test mentions is a finding with status PENDING_DECISION: it
may be intended, or a gap nobody noticed.

## Gotchas

- Transitions of one entity can be spread across services; a state table
  built from one component is partial until the others are checked.
- Soft deletion keeps rows that some queries still return; record which
  paths filter them.
- Enumeration values that nothing sets, or that only old data carries, are
  still states for consumers reading old records.
