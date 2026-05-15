#!/usr/bin/env -S deno run
// Add --allow-* permission flags to the shebang line as needed, e.g.:
// #!/usr/bin/env -S deno run --allow-read --allow-net

// Import dependencies (pin exact versions):
// import { load } from "npm:cheerio@1.0.0";
// import { z } from "npm:zod@3.22.0";

const args = parseArgs(Deno.args);

if (args.help) {
  console.log(
    `Usage: deno run --allow-... scripts/[name].ts [options]

Options:
  --help    Show this help message`,
  );
  // Add flag descriptions here.
  Deno.exit(0);
}

// --- implementation ---

// Data → stdout (JSON preferred):   console.log(JSON.stringify(result, null, 2));
// Diagnostics → stderr:             console.error("message");
// Exit codes: Deno.exit(0) success, Deno.exit(1) error, Deno.exit(2) bad args.

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
