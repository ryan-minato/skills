# Study Report

The deliverable of a harness study. Written for whoever will decide what
to do with the findings, so every item carries its evidence, its cost,
and its assessment, and nothing decides what another project does.

## Shape

1. **Scope and repositories** — the scope (all six layers, or the aspect
   named and the layers it maps to), then each repository with its location, pinned revision
   (and the uncommitted state, for a working tree), the frameworks and
   platforms its harness targets, and the facts asked of the user that
   remain unanswered.
2. **One section per layer in scope**, in the fixed order: environment and sandbox;
   knowledge and routing; hard gates; authorization boundary;
   specification and completion criteria; entropy management. Each
   section, per repository:
   - Grade (omitted, light, medium, thick) and the visible variables that
     explain it, or "not visible".
   - Pattern cards: mechanism, problem it solves, enforcement kind,
     liveness evidence with location and last-touched date, cost per
     task, and the assessment — strength, weakness, or neutral — with
     its reason.
   - Unverified rules (prose with no wiring) and broken items (pointers to
     missing files, commands with no recipe), each with its path.
   - With several repositories: a side-by-side table of the layer — one
     row per problem solved, one column per repository, each cell the
     mechanism, its cost, and its assessment, or "absent".
3. **Open questions** — what the repository could not show, contradictions
   between its documents and its wiring, and what the user intends to do
   with the findings, stated as a question for the application step.

## Rules

- A layer with nothing found is marked omitted and gets no description;
  do not infer a layer from a repository's general tidiness.
- A comparison ranks mechanisms by evidence and cost, never repositories
  by preference. "A does X, B does Y, X costs more context and is
  machine-enforced" is a comparison; "A's harness is better" is not.
- The assessment judges the studied repository on its own terms; it never
  decides what another project adopts and proposes no grade for one.
  "Worth learning from" and "to avoid" are findings about this
  repository; when the user's motive is borrowing, the last open question
  names it and hands it to the application step.
- Praise from the user, the repository's reputation, or its README is not
  evidence; the same probes and criteria apply to every repository.
- Quotations from a repository appear only as evidence for a finding, kept
  short, never as content to reuse. Secrets appear by name and location
  only.
- A document that contradicts the repository's wiring is reported with
  both locations, with no verdict on which side is intended.
- Last-touched dates come from history, not from file timestamps.
