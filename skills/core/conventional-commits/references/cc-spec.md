# Conventional Commits 1.0.0 — Specification Reference

Source: https://www.conventionalcommits.org/en/v1.0.0/

## Message structure

```
<type>[optional scope][optional !]: <description>

[optional body]

[optional footer(s)]
```

## Normative rules

1. Commits MUST be prefixed with a type: a noun (`feat`, `fix`, …) followed by an
   optional scope, an optional `!`, and a required colon and space.
2. `feat` MUST be used when the commit adds a new feature.
3. `fix` MUST be used when the commit fixes a bug.
4. A scope MAY follow the type: a noun in parentheses describing a section of the
   codebase, e.g. `fix(parser):`.
5. A description MUST immediately follow the colon and space.
6. A longer body MAY follow the description, beginning one blank line after it.
7. The body is free-form and MAY span multiple newline-separated paragraphs.
8. One or more footers MAY follow one blank line after the body. Each footer is a
   token, then `: ` or ` #` as separator, then a value.
9. Footer tokens MUST replace inner whitespace with `-` (e.g. `Acked-by`);
   `BREAKING CHANGE` is the sole exception.
10. A footer value MAY contain spaces and newlines; parsing stops at the next
    valid token/separator pair.
11. Breaking changes MUST be indicated by `!` before the colon and/or a
    `BREAKING CHANGE:` footer.
12. A `BREAKING CHANGE:` footer MUST be the uppercase text `BREAKING CHANGE`,
    a colon, a space, and a description of the break.
13. When `!` is used, the description itself MAY explain the break and the
    footer MAY be omitted.
14. Types other than `feat` and `fix` MAY be used.
15. Implementations MUST NOT treat units as case-sensitive, except
    `BREAKING CHANGE`, which MUST be uppercase.
16. `BREAKING-CHANGE` MUST be treated as synonymous with `BREAKING CHANGE` in
    footer tokens.

## Standard types and semver

| Type | Semver bump | Intent |
|---|---|---|
| `feat` | MINOR | New user-visible behavior |
| `fix` | PATCH | Correct wrong behavior |
| `perf` | PATCH | Measured performance improvement, behavior unchanged |
| `refactor` | — | Restructure code, behavior unchanged |
| `style` | — | Cosmetic only: whitespace, formatting, lint autofixes |
| `test` | — | Tests only |
| `docs` | — | Documentation only |
| `build` | — | Build system or dependency changes |
| `ci` | — | CI pipeline configuration |
| `chore` | — | Remaining maintenance (no src/test impact) |
| `revert` | varies | Revert a prior commit |

Any commit marked breaking (`!` or `BREAKING CHANGE:`) → MAJOR bump regardless of
type.

## Grammar (ABNF)

```
commit      = title [ LF LF body ] [ LF LF footers ]
title       = type [scope] ["!"] ":" SP description
type        = 1*wchar
scope       = "(" *wchar ")"
description = 1*(wchar / SP)
body        = paragraph *(LF LF paragraph)
paragraph   = 1*(wchar / SP / LF)
footers     = footer *(LF footer)
footer      = token ": " value
            / token " #" value
token       = "BREAKING CHANGE" / "BREAKING-CHANGE" / 1*(wchar / "-")
value       = 1*(wchar / SP / LF)
wchar       = %x21-7E / %x80-10FFFF   ; any non-space, non-control character
```

## Worked examples

Short bug fix:

```
fix(auth): reject expired JWT tokens
```

Feature with context body:

```
feat(api): add rate limiting to all endpoints

Without rate limiting a single client can exhaust the connection pool
under sustained load. Adds a sliding-window limiter (100 req/min per
IP) backed by Redis; the ceiling is configurable via RATE_LIMIT_MAX.
```

Breaking change with `!` and footer:

```
feat!: replace session cookies with JWT

BREAKING CHANGE: all existing sessions are invalidated on upgrade and
users must re-authenticate. The Set-Cookie header is no longer issued.
```

Breaking change where the title says enough (footer omitted):

```
refactor!: drop support for Node.js 16
```

Revert:

```
revert: feat(auth): add OAuth2 provider support

Reverts commit 3a4b5c6. The OAuth2 integration regressed the password
reset flow and the fix is not ready.
```

Multiple footers:

```
fix(deps): upgrade axios to 1.7.2

Fixes an SSRF vulnerability in redirect handling.

Closes #482
Reviewed-by: Jane Doe
Co-authored-by: John Smith <john@example.com>
```

## Footer edge cases

- `token #value` form: `Refs #133` is a valid footer without a colon.
- Multi-line footer values continue until the next `token: ` / `token #` pair:

  ```
  BREAKING CHANGE: environment variables now take precedence over
  config files, which changes resolution order for existing setups.
  Reviewed-by: Jane Doe
  ```

- Issue references are footers, not body text: `Closes #12`, `Fixes #7, #9`.
- Tools generally match footer tokens case-insensitively except
  `BREAKING CHANGE`/`BREAKING-CHANGE`, which must be uppercase.
