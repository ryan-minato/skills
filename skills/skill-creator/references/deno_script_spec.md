# Deno Script Specification

## Imports

Deno resolves dependencies directly from import specifiers — no install step required.

```typescript
import * as cheerio from "npm:cheerio@1.0.0";
import { z } from "npm:zod@3.22.0";
import { assertEquals } from "jsr:@std/assert@0.226.0";
```

| Prefix | Registry |
|---|---|
| `npm:` | npm registry |
| `jsr:` | JSR (Deno-native packages) |

Always pin exact versions. Unpinned imports (`npm:cheerio`) resolve to latest at
runtime and should be avoided.

## Shebang

```typescript
#!/usr/bin/env -S deno run
```

With embedded permissions:

```typescript
#!/usr/bin/env -S deno run --allow-read --allow-net
```

The `-S` flag splits the remainder as multiple arguments.

## Running

```bash
deno run --allow-read scripts/extract.ts
```

## Permission Flags

Deno is deny-by-default. Every external resource access requires an explicit flag.

| Flag | Grants access to |
|---|---|
| `--allow-read[=path]` | Filesystem reads (optionally scoped) |
| `--allow-write[=path]` | Filesystem writes |
| `--allow-net[=host]` | Network (optionally scoped to hostname) |
| `--allow-env[=var]` | Environment variables |
| `--allow-run[=cmd]` | Subprocess execution |

Prefer scoped permissions over broad ones. Document required flags in `--help`. Avoid
`--allow-all`. Use `--` to separate Deno flags from script flags:

```bash
deno run --allow-read scripts/lint.ts -- --fix .
```

## Design Rules

- **No interactive prompts** — use `Deno.args` for all input
- **`--help` required** — document all flags and include a usage example
- **Data to stdout, diagnostics to stderr** — `console.log` vs `console.error`
- **Structured output** (JSON preferred)
- **Exit codes**: `Deno.exit(0)` success, `Deno.exit(2)` bad arguments,
  `Deno.exit(1)` general error
