# The Entrypoint

The entrypoint is the file coding agents always read, so it is the discovery
root: every harness rule must be stated in it or reachable from it.

## Choose the file

Default to AGENTS.md. When some of the team uses Claude Code, add a
CLAUDE.md whose only content is a reference to AGENTS.md, so both stay
consistent. Write rules directly in CLAUDE.md only when the team explicitly
uses Claude Code exclusively. Frameworks differ in how they treat the
entrypoint, but AGENTS.md is the widely recognized shared convention.

## Two parts, in this order

First the always-applicable constraints: global safety rules, commit and
publishing rules, language conventions, the validation entry command, the
pending-decisions rule. The test for this part is that breaking the rule
costs the same whatever the task. It is short, and it comes first because
attention decays with position. Then the routing table, keyed by situation
rather than by topic: "when you are doing this kind of task, or meet this
condition, read that file's section" — a declaration of where each kind of
fact lives, organized by source of truth so the sync rules fall out of it.

Before moving any content behind a pointer, answer four questions: how will
the agent know, at the right moment, to read it; what does it gain from
the read; how much unrelated content does it ingest on the way; and does
the tool jump cost less than inlining? A negative answer to the first or
the last keeps the content inline. The budget below exists to force these
four answers on every section, not for the number itself.

Skills that are process-critical — a publish gate, a compliance review
before commit — are routed from the table by situation, naming the skill.
Description matching is probabilistic and is enough only for an auxiliary
skill; a miss on a critical one is the same as not having it.

## What belongs in it

- The project's purpose, in a sentence or two.
- An architecture-level project map — unless another discoverable
  architecture document already provides one. Keep it at the level of
  directories and modules; a map that changes on routine file additions is
  too detailed.
- Core conventions that apply to most tasks.
- When-to-read-what pointers for every knowledge and reference file, since
  nothing else reveals those files exist.
- Validation commands, and the approval or safety boundaries that apply to
  most tasks: which actions proceed, which are pre-authorized by the
  specification (or, without one, the approved plan or the policy file),
  which are queued as pending decisions.
- The keep-current rule, or a pointer to whatever mechanism owns it.

## Subdirectory files and framework carriers

Frameworks differ in when a subdirectory entry file loads and which copy
wins when files nest: some load it when the agent touches the directory,
some only read the chain from the current directory to the root at
startup, some define "nearest wins" without saying whether the parent
still applies. Path-matched rules and hook-injected text are framework
features too. Rely on these carriers only when the team's framework is
known and supports them; with mixed or undecided frameworks use them for
local supplements only. Either way, one rule is framework-independent: a
subdirectory file or framework carrier holds no rule that must hold on
every task and never contradicts the root, because it may not load and
because the precedence is unpredictable. Safety rules and commit
discipline live in the root only.

## Length budget

The entrypoint is a map, not a manual. Aim for about 100 lines; when it
grows past that, move detail into knowledge files, references, or skills and
leave a pointer. It may approach about 200 lines only when it is itself the
complete harness for a lightweight project and separate files would add more
navigation cost than value.

Two limits on that compression: rules every agent must always see stay in
the entrypoint no matter what, since content behind a pointer is only loaded
when the pointer's condition fires; and an oversized entrypoint is not a
harmless archive — it is loaded into every session and crowds out working
context.

## Use the skeleton

Copy `assets/agents-md-skeleton.md` (path relative to this skill's root),
fill only the sections backed by current facts or explicit user decisions, and
delete every section that does not serve the project. An empty or
speculative section teaches agents to distrust the file.
