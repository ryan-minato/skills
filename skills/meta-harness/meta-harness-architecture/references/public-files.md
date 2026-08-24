# Public-Convention Files

Read when the plan touches README, LICENSE, SECURITY, CONTRIBUTING, or an
architecture document. These files follow public conventions and serve
humans as well as agents — agent-doc style rules do not apply to them.

## Classification

| File | Audience | Rule |
|---|---|---|
| `README.md` | humans and agents | Public front door. Follow common README conventions; keep it human-readable prose. |
| `LICENSE` / `LICENSE.md` | humans, tools | Verbatim license text only. Never edit, annotate, or restyle. |
| `SECURITY.md` | humans | Follow the security-policy conventions (reporting channel, scope, disclosure expectations). |
| `CONTRIBUTING.md` | humans | Follow contributing-guide conventions; write for a human contributor. |
| `ARCHITECTURE.md` | humans and agents | Human-readable architecture description. In an agent harness it also serves as the offload target for long architecture and stack material from the entrypoint. |
| `AGENTS.md` (and framework entry files) | agents only | Agent-first: terse, no pleasantries, no human-readability duty. |
| `DESIGN.md` | agents (per its public spec) | Reserved name — only the visual-design description format may use it. |
| knowledge documents | agents first | Terse, facts first, load condition at top, no flattery or filler. |

## Planning Rules

- Never rewrite a public-convention file into agent-first style, and never
  count it as a knowledge document.
- If a public-convention file is missing and the project needs it, plan it
  as a human-authored or human-reviewed item — its content is a team
  statement, not an agent inference.
- Improvements to these files follow their own public conventions, not the
  harness style rules in the rest of the plan.
