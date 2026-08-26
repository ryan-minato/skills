# External Knowledge Backend

Read when the team already maintains truth outside the repository (an
issue tracker's documents, a wiki, a shared drive) or the user chooses an
external location.

## Precondition

Agents must be able to reach the backend through a documented access path
— a tool, an API, or a fetchable URL available in their environment. If no
such path exists for the agents this project runs, the backend is invisible
in practice; keep the knowledge base in-repo instead and say why.

## What To Set Up

1. **Access path in the entrypoint.** Record in AGENTS.md how agents read
   and write the backend: which tool or endpoint, and any scope limits.
2. **An in-repo index.** Keep a small index document in the repository
   (seed it from the index asset) listing each external document with a
   one-line hook and its locator. The index is what the entrypoint's
   when-to-read table points at.
3. **Same authoring rules.** External documents follow the same agent-first
   style and the same one-structure rule as in-repo ones.

## The Drift Cost

External knowledge does not version with the code, so it drifts on every
code change that would have touched a co-versioned document. Compensate
with stricter reverse triggers: whenever an external document is updated,
the mechanism names the repository artifacts to inspect first. Hand this
obligation to whatever sync mechanism the harness installs.
