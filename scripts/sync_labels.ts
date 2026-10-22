#!/usr/bin/env -S deno run --no-prompt --allow-read=. --allow-run=gh
// Syncs a GitHub repository's labels to a JSON taxonomy file through the gh CLI.
//
// Reads the desired labels from a JSON array of {"name", "color", "description"} objects, compares
// them with the labels currently in the repository, and prints the plan as JSON to stdout. Dry-run by
// default: nothing changes without --apply. Idempotent: re-running after a successful apply yields an
// all-skip plan. --prune (with --apply) deletes repository labels absent from the file.
//
//   scripts/sync_labels.ts --file .github/labels.json --repo OWNER/REPO            # dry run
//   scripts/sync_labels.ts --file .github/labels.json --repo OWNER/REPO --apply [--prune]
//
// Called by the `labels / sync` workflow (.github/workflows/labels-sync.yml) with a write token, and by
// hand as the manual fallback. It runs beside that token, so it imports only local modules.
//
// `gh` is checked where its output enters: a missing gh, a non-zero exit, or output that is not a JSON
// list of labels ends the script naming the command, before any plan is printed.
//
// Exit codes: 0 plan printed or applied cleanly; 1 an invalid labels file; 2 bad arguments or a gh failure.
import { parseArgs } from "./lib/args.ts";
import { CommandError, eprint, type Output, run, UsageError } from "./lib/repo.ts";

const HEX_COLOR = /^[0-9a-f]{6}$/;
const REPO_SLUG = /^[^/\s]+\/[^/\s]+$/;
/**
 * gh paginates label listings; a truncated listing would classify existing labels as "create" and make
 * the apply fail halfway through.
 */
export const LABEL_PAGE_LIMIT = 1000;

const USAGE = `usage: sync_labels.ts --file FILE --repo OWNER/REPO [--apply] [--prune]

Sync a GitHub repository's labels to a JSON taxonomy file. Dry-run by default: prints the
create/update/skip/prune plan as JSON to stdout without changing anything; re-run with --apply to
execute it.

options:
  -h, --help     show this help message and exit
  --file FILE    path to the labels JSON file: an array of {"name", "color", "description"} objects
  --repo REPO    target repository as OWNER/REPO
  --apply        execute the plan (default: dry-run, print the plan only)
  --prune        with --apply, delete repo labels absent from the file

Exit codes: 0 plan printed or applied cleanly, 1 invalid labels file, 2 bad arguments or gh failure.`;

export interface Label {
    name: string;
    color: string;
    description: string;
}

export interface Plan {
    repo: string;
    applied: boolean;
    create: Label[];
    update: Label[];
    skip: string[];
    prune_candidates: string[];
    pruned: string[];
}

/** Runs one gh command and returns its exit status and output; injectable so tests never reach GitHub. */
export type GhRunner = (args: string[]) => Output;

/** The labels file is unusable; main() prints the message and exits 1. */
export class LabelsFileError extends Error {}

/** Runs gh for real; a missing gh is a CommandError naming the fix. */
export function defaultGh(args: string[]): Output {
    try {
        return run("gh", args);
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) {
            throw new CommandError("`gh` is not installed; install the GitHub CLI and run `gh auth login`.");
        }
        throw error;
    }
}

/** Runs one gh command through the runner and returns its stdout; a non-zero exit is a CommandError. */
function ghChecked(gh: GhRunner, args: string[]): string {
    const out = gh(args);
    if (out.code !== 0) {
        throw new CommandError(`\`gh ${args.join(" ")}\` exited ${out.code}: ${out.stderr.trim()}`);
    }
    return out.stdout;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The Python original's color.strip().lstrip("#").lower(). */
function normalizeColor(color: string): string {
    return color.trim().replace(/^#+/, "").toLowerCase();
}

/** The labels file text as normalized label objects; anything else in it is a LabelsFileError naming the file. */
export function parseDesired(text: string, path: string): Label[] {
    let data: unknown;
    try {
        data = JSON.parse(text);
    } catch (error) {
        throw new LabelsFileError(`cannot read ${path} as JSON: ${(error as Error).message}`);
    }
    if (!Array.isArray(data) || data.length === 0) {
        throw new LabelsFileError(`${path} must be a non-empty JSON array of label objects.`);
    }
    const labels: Label[] = [];
    const seen = new Set<string>();
    data.forEach((item: unknown, index) => {
        const where = `${path}[${index}]`;
        if (!isRecord(item) || typeof item.name !== "string" || item.name.trim() === "") {
            throw new LabelsFileError(`${where} must be an object with a non-empty string 'name'.`);
        }
        const name = item.name.trim();
        const color = item.color;
        if (typeof color !== "string" || !HEX_COLOR.test(normalizeColor(color))) {
            throw new LabelsFileError(
                `${where} (${JSON.stringify(name)}): color ${JSON.stringify(color ?? null)} is not a 6-digit hex ` +
                    `string (use e.g. "d73a4a", without "#").`,
            );
        }
        const description = "description" in item ? item.description : "";
        if (typeof description !== "string") {
            throw new LabelsFileError(
                `${where} (${JSON.stringify(name)}): description ${JSON.stringify(description)} is not a string.`,
            );
        }
        if (seen.has(name.toLowerCase())) {
            throw new LabelsFileError(`${where}: duplicate label name ${JSON.stringify(name)}.`);
        }
        seen.add(name.toLowerCase());
        labels.push({ name, color: normalizeColor(color), description: description.trim() });
    });
    return labels;
}

/** Reads the labels file; an unreadable file is a LabelsFileError. */
export function loadDesired(path: string): Label[] {
    let text: string;
    try {
        text = Deno.readTextFileSync(path);
    } catch (error) {
        throw new LabelsFileError(`cannot read ${path} as JSON: ${(error as Error).message}`);
    }
    return parseDesired(text, path);
}

/** The repository's labels by lower-cased name, read through `gh label list`. */
export function fetchCurrent(gh: GhRunner, repo: string): Map<string, Label> {
    const command = "gh label list";
    const out = ghChecked(gh, [
        "label",
        "list",
        "-R",
        repo,
        "--json",
        "name,color,description",
        "--limit",
        String(LABEL_PAGE_LIMIT),
    ]);
    let entries: unknown;
    try {
        entries = JSON.parse(out);
    } catch (error) {
        throw new CommandError(
            `\`${command}\` did not print JSON (${(error as Error).message}): ${JSON.stringify(out.slice(0, 200))}`,
        );
    }
    if (!Array.isArray(entries)) {
        throw new CommandError(
            `\`${command}\` printed ${JSON.stringify(out.slice(0, 200))}, not a JSON list of labels.`,
        );
    }
    if (entries.length >= LABEL_PAGE_LIMIT) {
        throw new CommandError(
            `${repo} returned ${entries.length} labels, at or above the ${LABEL_PAGE_LIMIT} listing limit — the ` +
                "listing may be truncated, which would turn existing labels into creates. " +
                "Raise LABEL_PAGE_LIMIT and re-run.",
        );
    }
    const current = new Map<string, Label>();
    entries.forEach((entry: unknown, index) => {
        if (
            !isRecord(entry) || typeof entry.name !== "string" || typeof entry.color !== "string" ||
            typeof entry.description !== "string"
        ) {
            throw new CommandError(
                `\`${command}\` entry ${index} is ${JSON.stringify(entry)}, not an object with string ` +
                    "name, color, and description; check the gh version's `label list --json` fields.",
            );
        }
        current.set(entry.name.toLowerCase(), {
            name: entry.name,
            color: entry.color.toLowerCase(),
            description: entry.description.trim(),
        });
    });
    return current;
}

/** Orders strings by code point, as Python's sorted() does. */
function byCodePoint(a: string, b: string): number {
    const x = [...a].map((c) => c.codePointAt(0)!);
    const y = [...b].map((c) => c.codePointAt(0)!);
    for (let i = 0; i < Math.min(x.length, y.length); i++) {
        if (x[i] !== y[i]) return x[i] - y[i];
    }
    return x.length - y.length;
}

export function buildPlan(
    desired: Label[],
    current: Map<string, Label>,
): { create: Label[]; update: Label[]; skip: string[]; pruneCandidates: string[] } {
    const create: Label[] = [];
    const update: Label[] = [];
    const skip: string[] = [];
    for (const label of desired) {
        const existing = current.get(label.name.toLowerCase());
        if (existing === undefined) create.push(label);
        else if (existing.color !== label.color || existing.description !== label.description) update.push(label);
        else skip.push(label.name);
    }
    const desiredKeys = new Set(desired.map((label) => label.name.toLowerCase()));
    const pruneCandidates = [...current.keys()].sort(byCodePoint).filter((key) => !desiredKeys.has(key))
        .map((key) => current.get(key)!.name);
    return { create, update, skip, pruneCandidates };
}

export function applyPlan(
    gh: GhRunner,
    repo: string,
    create: Label[],
    update: Label[],
    pruneCandidates: string[],
    doPrune: boolean,
): string[] {
    for (const label of create) {
        eprint(`create: ${label.name}`);
        ghChecked(gh, [
            "label",
            "create",
            label.name,
            "-R",
            repo,
            "--color",
            label.color,
            "--description",
            label.description,
        ]);
    }
    for (const label of update) {
        eprint(`update: ${label.name}`);
        ghChecked(gh, [
            "label",
            "edit",
            label.name,
            "-R",
            repo,
            "--color",
            label.color,
            "--description",
            label.description,
        ]);
    }
    const pruned: string[] = [];
    if (doPrune) {
        for (const name of pruneCandidates) {
            eprint(`delete: ${name}`);
            ghChecked(gh, ["label", "delete", name, "-R", repo, "--yes"]);
            pruned.push(name);
        }
    }
    return pruned;
}

/** JSON as Python's json.dump(indent=2) prints it: every character outside printable ASCII escaped. */
export function pythonJson(value: unknown): string {
    return JSON.stringify(value, null, 2).replace(
        /[\u007f-\uffff]/g,
        (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"),
    );
}

export function main(argv: string[], gh: GhRunner = defaultGh): number {
    try {
        const args = parseArgs(argv, {
            string: ["file", "repo"],
            boolean: ["apply", "prune", "help"],
            alias: { h: "help" },
        });
        if (args.flags.help) {
            console.log(USAGE);
            return 0;
        }
        if (args.positional.length > 0) throw new UsageError(`unrecognized arguments: ${args.positional.join(" ")}`);
        const file = args.values.file;
        const repo = args.values.repo;
        if (file === undefined || repo === undefined) {
            const missing = [file === undefined ? "--file" : "", repo === undefined ? "--repo" : ""].filter(Boolean);
            throw new UsageError(`the following arguments are required: ${missing.join(", ")}`);
        }
        if (!REPO_SLUG.test(repo)) throw new UsageError(`--repo must be OWNER/REPO, got ${JSON.stringify(repo)}`);
        if (args.flags.prune && !args.flags.apply) {
            throw new UsageError("--prune requires --apply; the dry-run plan already reports prune candidates");
        }

        const desired = loadDesired(file);
        const current = fetchCurrent(gh, repo);
        const { create, update, skip, pruneCandidates } = buildPlan(desired, current);
        const pruned = args.flags.apply ? applyPlan(gh, repo, create, update, pruneCandidates, args.flags.prune) : [];
        const plan: Plan = {
            repo,
            applied: args.flags.apply,
            create,
            update,
            skip,
            prune_candidates: pruneCandidates,
            pruned,
        };
        console.log(pythonJson(plan));
        return 0;
    } catch (error) {
        if (error instanceof UsageError) {
            eprint(USAGE.split("\n")[0]);
            eprint(`sync_labels: error: ${error.message}`);
            return 2;
        }
        if (error instanceof LabelsFileError) {
            eprint(`sync_labels: error: ${error.message}`);
            return 1;
        }
        if (error instanceof CommandError) {
            eprint(`sync_labels: error: ${error.message}`);
            return 2;
        }
        throw error;
    }
}

if (import.meta.main) Deno.exit(main(Deno.args));
