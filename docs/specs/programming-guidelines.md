# Spec: programming-guidelines

## Purpose

Defines four universal coding standards for all programming tasks: think before coding,
simplicity-first, surgical changes, and goal-driven execution. Anchors agent behavior
to a disciplined engineering philosophy, preventing speculative output, scope creep,
and vague execution.

## Trigger Conditions

Load for any programming task. Also trigger on indirect phrasings:

- "Keep it simple" / "don't over-engineer" / "minimal changes only"
- "Only change what's needed" / "surgical edits" / "don't touch unrelated code"
- "What are the tradeoffs?" / "should I refactor this?"
- "Write a plan first" / "define success criteria before starting"
- Any code writing, editing, reviewing, debugging, or refactoring request

## What the Agent Lacks Without This Skill

Without this skill, agents tend to:

- Add unrequested features, abstractions, or configuration flexibility
- Refactor adjacent code while fixing a targeted bug
- Skip surfacing tradeoffs or alternative interpretations of a request
- Produce vague success criteria ("make it work") that require constant clarification

## Constraints and Scope

- Apply proportionally: trivial one-off tasks do not require a formal written plan,
  but the underlying principles still apply.
- Language- and framework-agnostic. No specific runtime or toolchain is assumed.
- These are meta-guidelines, not language-specific style rules.

## External References

- Andrej Karpathy (source): https://x.com/karpathy/status/2015883857489522876
- Community synthesis: https://github.com/forrestchang/andrej-karpathy-skills

No external APIs, libraries, or services are required.
