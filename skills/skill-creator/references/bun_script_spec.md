# Bun Script Specification

## Self-Contained Dependencies via Version-Pinned Imports

Bun auto-installs packages at runtime when it encounters version specifiers in import
paths — no `package.json` or prior install step needed.

```typescript
import * as cheerio from "cheerio@1.0.0";
import { z } from "zod@3.22.0";
```

Pin the exact version in the import path. Without a version, Bun resolves against
the local `package.json`, which may not be present.

## Auto-Install Behavior

Auto-install applies **only when no `node_modules` directory exists** anywhere in the
directory tree (from the script's location up to the filesystem root).

| Condition | Bun behavior |
|---|---|
| No `node_modules` found in tree | Auto-installs to the global cache |
| `node_modules` found anywhere in tree | Falls back to Node.js resolution — auto-install disabled |

Keep scripts in `scripts/` without a sibling `package.json` or `node_modules` to
reliably trigger auto-install.

## Shebang

```typescript
#!/usr/bin/env bun
```

## Running

```bash
bun run scripts/extract.ts
```

TypeScript is supported natively — no compilation step.

## Design Rules

- **No interactive prompts** — use `process.argv` for all input
- **`--help` required** — document all flags and include a usage example
- **Data to stdout, diagnostics to stderr** — `console.log` vs `console.error`
- **Structured output** (JSON preferred)
- **Exit codes**: `process.exit(0)` success, `process.exit(2)` bad arguments,
  `process.exit(1)` general error
