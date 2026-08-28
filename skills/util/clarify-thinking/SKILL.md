---
name: clarify-thinking
description: >
  Interrogates a plan, an idea, or a decision in rounds until the user and the
  agent hold one understanding of it, with nothing left silently assumed. Use
  whenever the user hands over thinking rather than a task: a plan, an idea, or
  a decision they have already half-made, offered for a reaction or a look-over
  before they commit to it — "帮我看看还有哪些没想清楚", "grill me on this",
  "poke holes in this", "what am I not considering", "help me think this
  through", "帮我把这个想法理清楚"; when the alternatives behind a decision were
  never enumerated; or when a request is too under-specified to act on and
  guessing would waste the work. Not for defining what something should achieve
  from scratch and recording it as a goal document, and not for the
  implementation plan, task breakdown, or code that follows once the
  understanding is shared.
license: MIT
metadata:
  references: >
    https://github.com/mattpocock/skills/blob/main/skills/productivity/grilling/SKILL.md
---

# Clarify Thinking

Interrogate the user relentlessly until the two of you hold one understanding
of the plan, idea, or decision on the table. Build nothing and decide nothing
on your own authority until they confirm you have reached it.

## Work the design tree in rounds

Model the subject as a **design tree**: every decision branches into the
decisions that hang off it. Restate the subject in a sentence first so the
tree has a root the user recognizes.

The **frontier** is every decision whose prerequisites are already settled —
the questions you can ask *now* without guessing at answers you have not heard
yet. Ask the whole frontier in one round. The answers reshape the tree: settled
decisions push the frontier outward and unblock what depended on them.
Recompute it and ask the next round. A question whose answer depends on another
question still open in this round belongs to a *later* round, not this one.

Attach your recommended answer, and the reasoning it rests on, to every
question you can reason an answer for: a question with no proposal moves the
whole burden of thought onto the user. When only the user can know the answer —
an unstated intent, a budget, a deadline — ask it bare, because an invented
recommendation only anchors them to your guess.

Done when: the frontier is empty — every branch visited, nothing left silently
assumed — and the user confirms the shared understanding.

## Ask through the host's own question tool

When the harness offers a tool for putting structured questions to the user —
selectable options, a multiple-choice prompt, an interactive form — ask through
it, and pack each round into as few calls as its limits allow. Prefer it over
writing the questions out: answering by choosing beats answering by retyping,
and each question comes back answered on its own instead of collapsing into one
reply that silently drops half the round.

- Lead each question's options with your recommended answer and mark it as the
  recommendation.
- Respect the tool's own limits on questions per call and options per question.
  When a round does not fit, ask the questions that unblock the most branches
  first and hold the rest for the next round; never fuse two decisions into one
  compound question to save a slot.
- Open questions still go through the tool: offer the candidate answers you can
  enumerate as options and let its free-form entry carry anything else.
- Fall back to plain text only when the harness has no such tool. Then number
  the round, format it as below, and stop and wait for the user's reply:

  ```
  ❓ **Q1** — **<question title>**: <question body, may run several
  paragraphs and may itself offer choices>

  ➡️ <your recommended answer>
  ```

Keep a round whole: put it entirely through the tool or entirely in text, never
half in each.

## Find the facts yourself

Facts are your job; the decisions are the user's. When a frontier question turns
on something the environment already knows — a file, a config value, a command's
output, a version — go and find it instead of asking. Dispatch a subagent when
the harness has them, otherwise look it up inline. Do not stall the round on it:
a running lookup is one more unsettled prerequisite, so only the questions
downstream of it wait while the rest of the frontier goes out now.

## Leave no files behind

Hold the tree, the drafts, and the conclusions in the conversation. A scratch
file dropped into the user's project gets swept into their next commit, and
everything produced here is half-formed by design.

When an artifact is genuinely needed — the user asks for one, or the outcome is
too large to carry in conversation — write it outside the workspace and hand
back its absolute path: the harness's scratchpad or temporary directory when one
is announced, otherwise `$TMPDIR` (`/tmp` when unset) on Linux and macOS, or
`%TEMP%` on Windows. Write inside the workspace only where the user names a
location there.

## Language

Ask in the language the user is speaking. They answer a round of questions under
time pressure, and translating first is friction that buys nothing.
