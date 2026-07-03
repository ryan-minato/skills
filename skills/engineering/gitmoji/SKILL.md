---
name: gitmoji
description: >
  Drafts git commit messages following the gitmoji convention: picks the one emoji
  that matches the change intent and composes a structured message. Use when writing
  emoji-prefixed commit messages, when the project uses gitmoji, when the user says
  "use gitmoji", "gitmoji format", or "what emoji for this commit", or when git
  history shows titles beginning with an emoji or an emoji code like ":bug:".
---

# Gitmoji

Produce one message whose single leading emoji states the change intent. Resolve
the project's variant before drafting.

## Rule precedence

Resolve each dimension at the highest level that states it; lower levels only fill
what remains open.

1. **Explicit documentation** — gitmoji rules in `AGENTS.md`, `CONTRIBUTING.md`, or
   `README.md`.
2. **Git history** — `git log --oneline -20` decides three dimensions:
   - **Grammar**: standalone `<emoji> <description>` (the gitmoji.dev default) vs
     combined `<type>: <emoji> <description>` layered on Conventional Commits.
   - **Emoji form**: unicode character (`🐛`) vs text code (`:bug:`).
   - **Scope**: present only if history consistently uses one.
3. **Defaults in this skill** — standalone grammar, unicode form, no scope.

## Message grammar

Standalone (default):

```
<emoji> <description>

[body]

[footer(s)]
```

Combined (only when history shows it): `<type>[(<scope>)][!]: <emoji> <description>`
— the emoji sits after the colon and everything else follows the project's
Conventional Commits rules.

Hard rules for every message:

- Exactly one emoji, first thing in the title, expressing the dominant intent. A
  change that honestly needs two emojis is two commits.
- Title ≤ 50 characters (the emoji counts as one); description in imperative mood,
  lowercase first letter, no trailing period.
- Title only is the default; add a body (blank line after the title, lines ≤ 72
  characters, explains why not how) when the reason is not obvious from the diff or
  the user asks.
- Breaking changes use 💥 as the emoji and a `BREAKING CHANGE: <what breaks and how
  to migrate>` footer.

## Emoji selection

Walk this list top-down and stop at the first match:

1. Reverts a prior commit → ⏪️
2. Introduces a breaking change → 💥
3. Fixes a security or privacy issue → 🔒️
4. Fixes a bug → 🚑️ if it is a critical production hotfix, 🩹 if trivial and
   non-critical, otherwise 🐛
5. Adds new user-visible behavior → ✨
6. Improves measured performance → ⚡️
7. Removes code or files → 🔥
8. Restructures code without behavior change → ♻️; if purely cosmetic
   (formatting, whitespace) → 🎨
9. Touches only tests → ✅ (🧪 when deliberately committing a failing test)
10. Touches only documentation → 📝 (✏️ for pure typo fixes)
11. Changes dependencies → ⬆️ upgrade, ⬇️ downgrade, ➕ add, ➖ remove,
    📌 pin
12. Changes CI → 👷 (💚 when fixing a broken CI build)
13. Changes configuration files → 🔧; development scripts → 🔨
14. Moves or renames files, paths, or routes → 🚚
15. Tags a release → 🔖; begins a project → 🎉

No match, or two rules feel equally right for different reasons? Read
[references/gitmoji-list.md](references/gitmoji-list.md) — the full official list
covers rarer intents (UI, i18n, logs, types, infra, DX) — and prefer the more
specific emoji over the more generic one.

## Validate before handing over

- [ ] Exactly one emoji, first in the title, matching the dominant intent.
- [ ] Grammar, emoji form, and scope match the resolved project variant.
- [ ] Title ≤ 50 characters, imperative, lowercase start, no trailing period.
- [ ] Blank line between title and body; body lines ≤ 72 characters.
- [ ] Breaking change uses 💥 plus a `BREAKING CHANGE:` footer.

## Gotchas

- 🔥 deletes, 🗑️ deprecates, ⚰️ removes already-dead code — pick by what the diff
  actually does, not by how final the removal feels.
- ⚡️ claims a measured improvement; a feature that happens to be fast is ✨.
- 💥 is reserved for consumer-breaking changes — a dramatic internal refactor is
  still ♻️.
- 🚧 marks deliberate work-in-progress that a later commit continues; it is not a
  license to commit broken or speculative code.
- Several emojis carry a trailing variation selector (`U+FE0F`) — 🚑️, ⚡️, 🔒️,
  ⏪️, 🩹 among them. Copy the emoji from history or the reference list rather than
  retyping it, or code-form and unicode-form messages will mismatch in tooling that
  compares titles byte-wise.
- Text codes only render on platforms that expand them (GitHub does; plain
  `git log` does not) — which is why matching the project's existing form matters
  more than personal preference.
