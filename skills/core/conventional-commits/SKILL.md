---
name: conventional-commits
description: >
  Drafts git commit messages in the Conventional Commits 1.0.0 format. Use when
  writing a commit message with a type prefix; when the project uses commitlint or
  semantic-release, or its docs mandate conventional commits; when the user says "use
  conventional commits" or "CC format"; when asked which type, scope, or
  breaking-change marker a change needs; or when git history shows type-prefixed
  titles like "feat:", "fix:", "chore:". Not for executing the commit workflow itself
  or for emoji (gitmoji) message conventions.
---

# Conventional Commits

Produce one message that passes the project's own tooling first and the 1.0.0
specification second. Draft nothing before resolving the rules below.

## Rule precedence

Resolve each formatting dimension (allowed types, scope policy, casing, footers) at
the highest level that states it; lower levels only fill dimensions the higher
levels leave open. Never let a lower level override a higher one.

1. **Explicit documentation** — commit rules stated in `AGENTS.md`,
   `CONTRIBUTING.md`, or `README.md` (follow pointers between them).
2. **Tool configuration** — `commitlint.config.*`, `.commitlintrc*`, or the
   `commitlint` key in `package.json`. `rules['type-enum']` replaces the default
   type list; `rules['scope-empty']` / `rules['scope-enum']` decide scope policy.
3. **Git history** — `git log --oneline -20`: type casing, scope usage and
   granularity, footer habits.
4. **Defaults in this skill** — everything still unresolved.

## Message grammar

```
<type>[(<scope>)][!]: <description>

[body]

[footer(s)]
```

Hard rules for every message:

- Title ≤ 50 characters total, counting the type/scope prefix.
- `<description>`: imperative mood, lowercase first letter, no trailing period,
  exactly one space after the colon.
- Title only (no body) is the default; add a body when the why is not obvious from
  the diff, when trade-offs or side effects need recording, or when the user asks.
- Body: blank line after the title, lines ≤ 72 characters, explains why — the diff
  already shows how.
- Footers: one per line after a blank line; tokens hyphenate spaces
  (`Co-authored-by:`, `Reviewed-by:`); `BREAKING CHANGE` is the only token spelled
  with a space and is always uppercase.

## Type selection

Walk this list top-down and stop at the first match (with a config `type-enum`,
map the same intents onto the configured names instead):

1. Reverts a prior commit → `revert` (title repeats the original title; body names
   the reverted SHA).
2. Changes user-visible behavior → `fix` if it corrects wrong behavior, `feat` if
   the behavior is new.
3. Improves measured performance without changing behavior → `perf`.
4. Restructures code without changing behavior → `refactor`; if the change is
   purely cosmetic (whitespace, formatting, lint autofixes) → `style`.
5. Touches only tests → `test`; only documentation → `docs`.
6. Changes the build system or dependencies → `build`; CI pipeline files → `ci`.
7. Anything left (tooling, `.gitignore`, release chores) → `chore`.

A commit that matches two branches for different files is not atomic — propose
splitting it before bending the type.

## Scope

Omit scope by default. Add one only when config requires it, history uses it
consistently for this kind of change, or the user asks — and then copy the naming
and granularity history already uses rather than inventing a new scheme.

## Breaking changes

A breaking change is any change that forces consumers to update their code or
configuration. Mark it with `!` before the colon. When the migration needs
explanation, also add a `BREAKING CHANGE: <what breaks and how to migrate>` footer;
the `!` alone suffices when the title says it all.

## Validate before handing over

- [ ] Title ≤ 50 characters, imperative, lowercase start, no trailing period.
- [ ] Type is in the project's allowed list; casing matches history.
- [ ] Scope present/absent per the resolved policy.
- [ ] Blank line between title and body; body lines ≤ 72 characters.
- [ ] Breaking change carries `!` (and a `BREAKING CHANGE:` footer if it needs
      explanation).
- [ ] Footer tokens hyphenated; `BREAKING CHANGE` uppercase.

If the project has commitlint or a commit-msg hook, run it on the draft instead of
trusting the checklist alone.

## Gotchas

- `style` never changes logic; a lint fix that alters behavior is `fix`.
- `chore` is the last resort, not a catch-all — build- or CI-affecting maintenance
  belongs to `build`/`ci`.
- Semantic-release only bumps versions on `feat`, `fix`, `perf`, and breaking
  changes; typing a feature as `chore` silently skips its release.
- `breaking change:` in lowercase is ignored by semantic-release and commitlint —
  the footer token is case-sensitive in practice even though types are not.
- The prefix eats the budget: `feat(authentication-service): ` already spends 30 of
  the 50 title characters. Prefer a shorter scope over a truncated description.

## References and assets

- Read [references/cc-spec.md](references/cc-spec.md) when the type choice stays
  ambiguous after the list above, a footer edge case appears (multi-line values,
  `token #value` form), or the user asks about the specification itself.
- Copy [assets/gitmessage](assets/gitmessage) when the user asks to create or
  update a commit template, then enable it with
  `git config commit.template .gitmessage`.
