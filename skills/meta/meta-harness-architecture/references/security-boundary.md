# The Security Boundary

Read when the environment layer is rated, when an agent role reads external
input, or when deciding which harness files agents may change. The boundary
lives outside the framework: in the sandbox and in the scope of the tokens.
Framework permission lists are fool-proofing for a careless agent; an agent
with code execution can write a script or edit a build file around them.

## Three conditions decide isolation

An agent role needs a strictly isolated, credential-stripped instance only
when three conditions hold together. Drop any one and ordinary protection
suffices.

1. **Exposure.** The role holds sensitive assets or high privileges, reads
   input it does not control, and can act outward or change state. Nearly
   every networked coding agent meets this: it holds credentials, edits
   files, and reads external content. Missing one of the three removes the
   exposure: a read-only analysis agent with no credentials may read
   anything.
2. **Source reachability.** How cheaply can an attacker put content where
   the agent will read it? The trusted repository: unreachable.
   Reputation-guarded sources (official documentation, a major package's
   registry page, a well-known project's README): expensive, and the
   compromise is visible. Open-write sources (the project's own issues and
   comments, search results, arbitrary pages, an obscure dependency's
   README): free. Issues and tickets are the worst case, because they sit
   in a task-instruction context where a planted "reasonable next step" is
   read as a requirement.
3. **No human in the loop.** When a developer directs the agent and reviews
   the result, treat the agent as a colleague who may err or cut corners,
   not as an attacker: outward actions are queued for the person or
   narrowly pre-authorized (typically one named branch and the few
   outward writes the pre-authorization line names), so a hijacked
   agent can damage the local tree and reach only what the scoped token
   and that narrow pre-authorization allow; everything else leaves only
   through a person. This single review of irreversible
   outward results is not the per-command approval `authorization.md`
   rejects; it reviews few, concentrated, readable outputs, not many
   scattered steps. An unattended agent has no such gate.

All three hold — for example an unattended agent that triages public
issues with a token that can push and comment — and the role runs in a
separate instance without the sensitive credential and without direct
write to the main branch or to configuration. Any condition missing —
reading official documentation with a developer in the loop — and the role
gets ordinary protection.

A model's resistance to passive-content injection adjusts only the middle
tier: reputation-guarded sources, or open sources with a human in the loop.
It decides how much cleaning and restriction a read-only retrieval task
needs — a resistant model may read search results freely; a weak model, or
a weak model used as the retrieval sub-agent, gets its sources narrowed at
the network or tool-input layer. Resistance never changes the isolated
tier's verdict: it lowers the chance of hijack, isolation lowers the damage
after one, and the two do not substitute. Confirm resistance per model,
never by vendor reputation.

## The sandbox audit

Before granting autonomy, record an explicit accept-or-cut decision for
each of these:

- **Assets in reach.** Which credentials, environment variables, and private
  files can the agent and its scripts read or write? Never mount the host's
  whole credential directories; use scoped tokens with the minimum
  permission.
- **Host privilege leaks.** Is a container socket, a host root mount, or
  another high-privilege interface exposed inside the sandbox?
- **Local and intranet exposure.** Can the container reach the host's local
  services or the corporate network through bridged networking?
- **Session logs.** The framework's transcript records every file read and
  every command output, so it carries credentials and personal data more
  easily than a scratch note. Record where it is written — the framework's
  default directory when the project sets nothing — and whether it is
  collected centrally; both are explicit privacy decisions, never left as
  an open question.
- **The decision.** For each finding, the project owner accepts the risk or
  cuts the permission, in writing.

## Control-plane mutability

During bootstrap a human is present and an agent editing configuration is
no different from the human doing it. Once the harness is in daily use,
place each control-plane file on a spectrum and say so in the harness:

| Protection | Applies to | Mechanism |
|---|---|---|
| Hard read-only | rules that intercept privileged commands, sandbox configuration, secret-scan hooks | read-only at the mount or OS layer; only the framework executes the rejection |
| Soft read-only, guarded invariant | lint rules, compiler and test configuration, build scripts | an instruction: never relax a rule or delete an assertion to make failing code pass; adjust within the invariant, or propose the change for human approval |

A project may make a security rule unreadable to agents so they cannot read
the logic and route around it. Then the logic may be hidden, but the
rejection message must be readable and must route: it says this is a policy
boundary, not a transient fault, and names the alternative or the owner who
can change the policy. Otherwise the agent cannot tell policy from outage
and will retry or search for a bypass, which is what hiding the rule was
meant to prevent. Hiding raises the cost of a bypass; the boundary is still
the sandbox and the token.

## Fallback for what no check covers

Some concerns — a privacy review, a business-rule consistency check — have
no deterministic check. Two soft controls stand in, and both are
supplements, not boundaries: an explicit checkpoint in the workflow
("before committing, verify these three items") that turns an implicit
step into a visible one with a fixed place, and a verifier sub-agent with a
fresh context that reviews the result before submission, which avoids the
long-context dilution and self-justification of the agent that wrote the
code. Compliance with either is below 100 percent; they cover the gap
between hard checks, never replace one.
