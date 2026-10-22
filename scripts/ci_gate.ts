#!/usr/bin/env -S deno run --no-prompt --allow-env=NEEDS
// The `checks / gate` aggregator: the one job name the ruleset requires from checks.yml. GitHub reports a
// skipped required job as success, so the gate job runs `if: always()` and this script decides on the
// results of the jobs it needs, read from the NEEDS environment variable (`${{ toJSON(needs) }}`).
//
// Usage: NEEDS='<json>' scripts/ci_gate.ts [--allow-skip <job>]...
//   --allow-skip <job>  a dependency that may be skipped by design (repeatable; must be in NEEDS)
//
// It prints each dependency's result to stdout, names every failed, cancelled, or unexpectedly skipped
// dependency on stderr, and exits 1 when there is one. Exit 0: every dependency succeeded or was an allowed
// skip. Exit 2: NEEDS is missing or not the shape GitHub's `needs` context has, or the arguments are wrong.
// Called only by the `gate` job of .github/workflows/checks.yml.
import { parseArgs } from "./lib/args.ts";
import { eprint, UsageError } from "./lib/repo.ts";

const USAGE = "usage: NEEDS='<json of the needs context>' scripts/ci_gate.ts [--allow-skip <job>]...";

/** The results GitHub documents for `needs.<job_id>.result`. */
const RESULTS = ["success", "failure", "cancelled", "skipped"] as const;
type Result = typeof RESULTS[number];

/** NEEDS is missing or not shaped like the `needs` context; main() reports it and exits 2. */
export class NeedsError extends Error {}

/** Parses the `needs` context JSON into job -> result, refusing anything not shaped like it. */
export function parseNeeds(raw: string | undefined): Map<string, Result> {
    if (raw === undefined || raw.trim() === "") {
        throw new NeedsError("NEEDS is not set; the gate step must set `NEEDS: ${{ toJSON(needs) }}`");
    }
    let data: unknown;
    try {
        data = JSON.parse(raw);
    } catch (error) {
        throw new NeedsError(`NEEDS is not JSON (${(error as Error).message}): ${raw.slice(0, 200)}`);
    }
    if (typeof data !== "object" || data === null || Array.isArray(data)) {
        throw new NeedsError(`NEEDS must be a JSON object of job -> {result}, got: ${raw.slice(0, 200)}`);
    }
    const entries = Object.entries(data as Record<string, unknown>);
    if (entries.length === 0) {
        throw new NeedsError("NEEDS is empty; the gate job must list the jobs it aggregates in `needs:`");
    }
    const results = new Map<string, Result>();
    for (const [job, value] of entries) {
        const result = typeof value === "object" && value !== null
            ? (value as Record<string, unknown>).result
            : undefined;
        if (typeof result !== "string" || !(RESULTS as readonly string[]).includes(result)) {
            throw new NeedsError(
                `NEEDS.${job}.result must be one of ${RESULTS.join(", ")}, got: ${JSON.stringify(value)}`,
            );
        }
        results.set(job, result as Result);
    }
    return results;
}

/** The problems the gate fails on, one message per dependency, in NEEDS order. */
export function decide(results: Map<string, Result>, allowSkip: string[]): string[] {
    const unknown = allowSkip.filter((job) => !results.has(job));
    if (unknown.length > 0) {
        throw new UsageError(`--allow-skip names a job the gate does not need: ${unknown.join(", ")}`);
    }
    const problems: string[] = [];
    for (const [job, result] of results) {
        if (result === "failure" || result === "cancelled") {
            problems.push(`${job}: ${result}`);
        } else if (result === "skipped" && !allowSkip.includes(job)) {
            problems.push(`${job}: skipped, and it is not listed with --allow-skip`);
        }
    }
    return problems;
}

export function main(argv: string[], needs: string | undefined = Deno.env.get("NEEDS")): number {
    try {
        const args = parseArgs(argv, { multiple: ["allow-skip"], boolean: ["help"], alias: { h: "help" } });
        if (args.flags.help) {
            console.log(USAGE);
            return 0;
        }
        if (args.positional.length > 0) throw new UsageError(`unexpected argument ${args.positional[0]}`);
        const results = parseNeeds(needs);
        const problems = decide(results, args.lists["allow-skip"]);
        for (const [job, result] of results) console.log(`${job}: ${result}`);
        if (problems.length > 0) {
            for (const problem of problems) eprint(`ERROR: ${problem}`);
            eprint("A required job did not succeed. Fix that job; never weaken this gate.");
            return 1;
        }
        return 0;
    } catch (error) {
        if (error instanceof UsageError) {
            eprint(`ERROR: ${error.message}`);
            eprint(USAGE);
            return 2;
        }
        if (error instanceof NeedsError) {
            eprint(`ERROR: ${error.message}`);
            return 2;
        }
        throw error;
    }
}

if (import.meta.main) Deno.exit(main(Deno.args));
