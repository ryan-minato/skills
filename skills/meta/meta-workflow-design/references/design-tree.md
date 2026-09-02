# Design Tree and Questioning Discipline

Read before the first question of any design or redesign run.

## The tree

Maintain four collections: **Facts** (evidenced), **Decisions** (user-owned
choices with their current answer), **Dependencies** (edges from
prerequisites to the decisions they unlock), and the **Frontier** (every
unanswered decision whose prerequisites are all resolved).

1. Resolve every frontier item that is a discoverable fact yourself.
   Dispatch clean-context subagents for independent read-only research when
   the harness has them; never ask the user for anything the repository or
   its platform can prove.
2. Remove resolved facts and recompute the frontier.
3. Ask the entire remaining frontier in one round. A question whose answer
   depends on another question still open in the same round belongs to the
   next round — never ask a downstream question early. Asking which columns
   a board needs before the user has decided whether a planning surface
   exists at all is the canonical violation.
4. Apply the answers, recompute, and repeat until the frontier is empty or
   the user says the information is sufficient.

Keep the working tree in the conversation; before consensus it is a working
model, not an agreed source of truth, and a scratch file dropped into the
project gets swept into a commit.

## Ask through the harness's question tool

When the harness offers a structured question tool — selectable options, a
multiple-choice prompt, an interactive form — put every round through it
rather than free text: choosing beats retyping, and each question comes back
answered on its own instead of collapsing into one reply that drops half the
round.

- Single-select for exclusive decisions. Lead with the recommended option,
  mark it as recommended, and give the one-sentence reason drawn from the
  evidenced facts ("12 open items, one maintainer — a filtered list already
  answers every planning question").
- Multi-select for composable characteristics, such as overlays.
- Free-form entry only where the answer is genuinely open semantics — an
  objective's name, a constraint no option list can anticipate. Never turn
  an enumerable choice into an open question.
- Respect the tool's limits per call; when a round does not fit, send the
  questions that unblock the most branches first. Never fuse two decisions
  into one compound question to save a slot.
- Fall back to a numbered text round, one recommendation per question, only
  when the harness has no such tool — and then stop and wait for the reply.

## Recommendation is not decision

Attach a reasoned recommendation to every question you can reason about;
a bare question moves the whole burden onto the user. Ask bare only what no
reasoning can reach — unstated intent, budget, appetite. The user may
overrule any recommendation. When their choice carries a management cost
they may not see, state the cost and the trade-off exactly once, offer the
alternative, and accept the confirmed answer. Do not relitigate.

## Settled decisions are constraints

Once the user decides, the decision joins the tree as a prerequisite, not a
topic. Later rounds, later phases, and later builders must neither re-ask it
nor construct an outcome that quietly contradicts it. If new evidence
genuinely invalidates a settled decision, present the evidence and ask for
an explicit revision — never silently substitute your own answer.
