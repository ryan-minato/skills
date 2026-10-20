# The Authorization Boundary

Read when the authorization layer is rated, or when a permission list, a
pre-authorization line, or a pending-decisions rule is written. The sandbox
and the tokens decide what an agent *can* do; this layer decides, inside
that, what it *may* do on its own, what waits for a human, and what stays
human. The two axes are independent: a strict sandbox may carry a wide
authorization, and the reverse.

The policy this reference designs governs execution — commands, files,
pushes, comments. Review admission and integration are management gates
with their own policy, deposited by the authority builder where one ran;
this policy points to it and never restates it.

## Approval is not a security mechanism

The value of an agent is a stretch of unattended time. Per-command approval
destroys it, and the safety it suggests is false: a human asked to
understand each command fatigues within the hour and approves everything,
which is no review at all while the risk the agent declined to own has
been handed to a person. Security is the sandbox and the token. This layer
spends human attention only where judgment is needed.

What this does not reject: the single review of irreversible outward
results before they leave (push, publication). It reviews few, readable
outcomes rather than many scattered steps; the difference is frequency and
object, not the presence of a human.

## Move approval to both ends

Predictable decisions are approved in the plan; unpredictable ones are
collected at the end.

- **The plan approves results and bounds, never a step list.** The human
  approves what is done, what is not, and where the approach is limited.
  Approving steps is approving a method: it costs attention and absorbs no
  risk, because the steps are not where the risk is. This is what lets
  pre-authorization happen once instead of per command.
- **A blocked agent continues.** Finish everything that does not depend on
  the decision; many frameworks stop and ask by default, so the harness
  states the opposite. Queue the decision in a shape a person can act on
  hours later, from `assets/pending-decisions.md` (path relative to this
  skill's root): where it is stuck, the options, the recommended option
  with its reason, and the completed work each option affects. Without
  that shape, batched decisions are per-command fatigue delivered in one
  lump.

## Reversibility is the key

Whether an action may proceed, be pre-authorized, or be queued follows
from whether it is reversible and whether it leaves the machine.

| Action class | Examples | Rule |
|---|---|---|
| Reversible, internal | editing files, creating a branch, running tests, a local commit | proceed |
| Irreversible or outward | push, a comment, deleting data, a release | pre-authorized in the specification's pre-authorized-actions line, or queued as a pending decision |

A local commit and a push are not the same class: local history is
rewritten freely; after a push the common hosting platforms keep
unreachable commits addressable by hash and purge them only on request
(verify the behaviour of the project's platform). The specification — or,
where the project has none, the approved plan or the policy file itself —
therefore carries a line naming the branch it pushes and the outward
writes it allows, and the harness says which of these holds the line.

The pending-decisions list lives where a person will see it: the
description of the change request, a tracked file, or an issue, chosen by the
project and named in the policy. It is not a scratch note; only a
credential or log excerpt inside an item goes to an untracked place.

## A refusal is a decision

A refused action means one of two things, and each has one response;
routing around it is never a third:

- The approach is in the wrong place — a refused write to a file may mean
  the logic belongs elsewhere. Return to the plan and choose differently.
- It is a policy boundary. Record it and queue it as a pending decision.
- Never: patch around it with an equivalent command, a script that bypasses
  the check, or a retry. Giving up at the first refusal and routing around
  it are the same laziness in two directions.

## Writing permission lists

Allow explicitly the commands known to be safe and the ones the plan
authorized; deny explicitly the commands known to be dangerous; hand the
rest to the framework's default, with the sandbox absorbing a wrong call.
Add an ask entry only for a clear, necessary, irreplaceable reason, and
write the reason beside it. The same rule applies to tools and tool
servers: read-only tools are allowed, tools with outward side effects are
classified by reversibility, and no server is set to ask wholesale.

Say what the deny list is for: the mistakes of a careless colleague —
accidental deletion, a force push, a hard reset — not an attacker. An agent
with code execution can get around it, so it is fool-proofing, and the
boundary stays in the sandbox and the token. For a framework with a
classifier, "the rest goes to the default" means relying on a
non-deterministic component that changes between versions, acceptable with
a sandbox underneath; for a framework that asks by default, it means
either widening the allow list or accepting fatigue, and the project
chooses.

## Pairing mode

Step-by-step approval has two legitimate sources, and outside them the
project returns to approval at both ends: a phase (bootstrap and
rule-setting, where the human is present and the approval is collaboration
rather than interruption) and a project kind (strict security or
confidentiality, or a named developer who must answer for every line).
Fatigue applies in pairing mode too; pairing is not a license to ask
without limit.

## Where the policy lives and how it is kept

Write the policy where agents will read it: the entrypoint's
always-applicable part, or a knowledge file the entrypoint points to for
the situation. The policy protects itself: relaxing it is outside the
agent's authority, because an agent that is stuck will otherwise treat
"change the rule in the way" as the reasonable next step — the same risk
as relaxing a lint rule to pass a test, with the policy as the object.

Allow lists accumulate entropy. When the harness is next improved, do a
one-time review: an ask entry approved nearly every time is pure cost and
moves to allow; an allowed command that keeps causing rollbacks is
tightened. This is a harness-improvement action, not a standing metric.
