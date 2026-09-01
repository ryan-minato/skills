---
name: agentic-writing
description: >
  Agent-first documents — writes, reviews, improves, and prunes the files an
  agent reads before it works: AGENTS.md and other entrypoints, CLAUDE.md,
  knowledge-base entries, agent-facing spec formats such as DESIGN.md, skill
  instructions, and prompt files. Use when creating or restructuring one; when
  one has grown long or costs too much context and needs trimming; when the
  agent ignores a rule the document states, follows a pointer unreliably, or
  the instructions have drifted from the code; or when a file of this kind is
  the material at hand. Applies as the shared baseline when a specialized
  skill (skill authoring, harness architecture) is also active — the
  specialized skill's rules win on conflict. Not for prose written for human
  readers, for README, CONTRIBUTING, or SECURITY, or for application code.
license: Apache-2.0
---

# Agentic Writing

An agent-first document is read by an agent first and a person second. Every
line it loads spends a scarce resource — not only tokens, but attention:
attending to everything is attending to nothing, and answer quality decays as
context grows. A line earns its place by changing what the agent does.

## Scope and precedence

This covers agent entrypoints (AGENTS.md, CLAUDE.md), knowledge-base entries,
agent-facing spec formats such as DESIGN.md, skill instructions, and prompt
files. README, CONTRIBUTING, and SECURITY answer to a broader human audience
and follow human writing conventions instead.

Apply this as the baseline whenever a specialized skill for the artifact is
also loaded (skill authoring, harness architecture); the specialized rules win
on conflict.

Write for the weakest model expected to read the file, and tie the document to
no single model or framework: a strong model reconstructs an unstated rule
from its priors where a small one breaks without it, and priors differ across
model families.

## Route by task

Writing or restructuring a whole document: apply every rule below, then read
[references/document-types.md](references/document-types.md) for the structure
its type requires.

Reviewing, improving, or diagnosing an existing one: locate the symptom, apply
the section it points to, then prune.

- A stated rule the agent ignores — it is buried in bulk, or phrased as a
  prohibition (Prompt the positive).
- A pointer the agent rarely follows — its trigger condition is vague
  (Context pointers).
- Instructions drifted from the code — the document cached what the
  environment already answers (One source of truth per meaning).
- A document too long to hold — repeated meaning, or a split never made
  (Prune, Split by loading).

## Spend context on behaviour

Courtesy, restatements of the request, and commentary about the document
itself carry no information and impose no constraint; delete them. Keep the
reason behind a rule wherever the rule leaves the agent room to judge — an
agent that knows why applies it correctly in a case you never enumerated.

## Split by loading, not by topic

Ask how a file loads, not what it is about: is it ever loaded alone? is it
loaded on every task? is it always and only loaded beside another file? Merge
what always loads together, and split off only what most runs skip. A person
opens a second file for free; an agent pays a tool call for it, so the bar for
splitting an agent-first document sits higher than for a human one. A split
that does not shrink what a single task loads only adds navigation cost.

## Context pointers

A context pointer names material outside the current context and states the
condition for reaching it — a skill description is one, and so is "read X when
Y" inside a document. Its trigger wording, not its destination, decides
whether the agent reaches: "Read `references/api-errors.md` when a request
returns a non-200 status", never "see references/ for details". Pointers
target project paths and external URLs alike; for a URL, say what that page is
authoritative for.

Pointers cost as well: the trigger occupies context permanently and following
one spends a round trip. That cost is what a split has to beat, so granularity
finer than the material justifies loses more than it saves. When a must-read
pointer fires unreliably, sharpen its condition first, and inline the material
only if that fails.

## Leading words

A leading word is a compact concept already in the model's pretraining that
the agent thinks with while the document is loaded (_lesson_, _fog of war_,
_tracer bullets_). Repeat the word as a token, never the meaning as a
sentence: the repetitions accumulate a distributed definition and anchor a
whole region of behaviour for the price of one word. Hunt for restatements to
collapse — "fast, deterministic, low-overhead" becomes a _tight_ loop; "a loop
you believe in" becomes the loop going _red_. Prefer widely attested words: a
niche or self-coined term recruits nothing on the models that lack the prior,
while charging you the tokens to define it. A word too weak to beat the
default is dead weight (_be thorough_); reach for a stronger word
(_relentless_) rather than another sentence.

## Prompt the positive

A prohibition drags the forbidden behaviour into context and makes it more
available, not less — _don't think of an elephant_. State the target behaviour
so the unwanted one is never named: "write one-line comments" rather than
"don't write long comments". Keep a prohibition only as a hard guardrail that
resists positive phrasing, and pair it with the positive target.

## One source of truth per meaning

Each meaning lives in exactly one authoritative place, so changing the
behaviour stays a one-place edit. A meaning repeated across the document costs
maintenance and tokens, and weights it past its real rank. (A leading word is
the deliberate inverse: it repeats the token, never the meaning.)

The environment is a source of truth too — package.json scripts, config files,
the directory layout, `--help` output. A document restating it is a cache, and
a cache pays off only where the lookup is expensive. Cache what lookup cannot
reveal: unwritten conventions, the reason behind a choice, the trap an
undocumented setting hides. Leave whatever one file or one command answers to
the environment, which never goes stale.

## Prune

Three passes over anything you wrote or touched:

1. **Deduplicate** — collapse every copy of a meaning into its one source of
   truth, or replace the copy with a pointer.
2. **Relevance** — delete lines that no longer bear on what the document is
   for: mere exposition, a branch that belongs behind a pointer, or a fact
   that went stale as the code moved. Short documents stay relevant more
   easily; with no pruning pass a document ends as sediment, and its live
   rules have to be dug out from under the dead ones.
3. **No-op hunt** — sentence by sentence: does this change behaviour against
   the default of the weakest model expected to read the file? Delete the
   whole sentence when it does not, since trimming its words keeps the no-op.
   The verdict is model-relative, not reader-relative: two people who disagree
   about a no-op disagree about the default, and settle it by running the
   document rather than by argument.

Done when: one more deletion would change what the weakest model in scope does.
