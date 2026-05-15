#!/usr/bin/env bun

// Import dependencies with version pins for auto-install:
// import { load } from "cheerio@1.0.0";
// import { z } from "zod@3.22.0";

const args = parseArgs(process.argv.slice(2));

if (args.help) {
  console.log(
    `Usage: bun run scripts/[name].ts [options]

Options:
  --help    Show this help message`,
  );
  // Add flag descriptions here.
  process.exit(0);
}

// --- implementation ---

// Data → stdout (JSON preferred):   console.log(JSON.stringify(result, null, 2));
// Diagnostics → stderr:             console.error("message");
// Exit codes: process.exit(0) success, process.exit(1) error, process.exit(2) bad args.

// --- minimal arg parser ---
function parseArgs(raw: string[]): Record<string, string | boolean> {
  const out: Record<string, string | boolean> = {};
  for (let i = 0; i < raw.length; i++) {
    if (raw[i].startsWith("--")) {
      const key = raw[i].slice(2);
      const next = raw[i + 1];
      if (next && !next.startsWith("--")) {
        out[key] = next;
        i++;
      } else {
        out[key] = true;
      }
    }
  }
  return out;
}
