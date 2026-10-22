#!/usr/bin/env -S deno run --no-prompt --allow-read --allow-run=gh
// Plans or applies managed issue labels from the issue forms' answers.
//
// The forms ask for Priority and Catalog; this script turns those answers into `priority/<value>` and
// `catalog/<value>` labels. The mapping is derived from .github/labels.json by prefix — there is no second
// list to keep in sync. Only managed labels are ever added or removed: the two prefixed families and
// `status/needs-triage`, which is removed once the issue carries a type label (bug, enhancement, task) and
// exactly one priority label.
//
// Usage:
//     ./scripts/sync_issue_metadata.ts --event "$GITHUB_EVENT_PATH" [--apply]
//     ./scripts/sync_issue_metadata.ts --issue 42 --repo OWNER/REPO [--apply]
//
// Without --apply the plan is printed to stdout as JSON and nothing changes (dry run). With --apply the plan
// is executed through `gh api` (GH_TOKEN or an authenticated gh).
// Called by: the `issues / triage` job of .github/workflows/issue-triage.yml (--event ... --apply), which
// holds an issues write token, so this script imports only local modules; by hand for a dry run.
// Exit codes: 0 plan computed (and applied); 1 a form answer did not map to a label (the rest of the plan
// still prints, and applies with --apply); 2 bad arguments, unreadable input, or a gh/API failure.
import { parseArgs } from "./lib/args.ts";
import { CommandError, eprint, type Output, readText, runAsync, UsageError } from "./lib/repo.ts";

export const LABELS_FILE = ".github/labels.json";
export const MANAGED_PREFIXES = ["priority/", "catalog/"] as const;
export const SECTION_FOR_PREFIX: Record<string, string> = { "priority/": "Priority", "catalog/": "Catalog" };
export const TYPE_LABELS = ["bug", "enhancement", "task"];
export const NEEDS_TRIAGE = "status/needs-triage";

export const USAGE =
    `usage: sync_issue_metadata.ts [-h] (--event PATH | --issue N --repo OWNER/REPO) [--labels PATH] [--apply]

Plan or apply the managed issue labels (priority/*, catalog/*, ${NEEDS_TRIAGE})
from the issue form's Priority and Catalog answers. Without --apply the plan is
printed as JSON and nothing changes.

options:
  -h, --help          show this help message and exit
  --event PATH        path to the issues event payload (GITHUB_EVENT_PATH)
  --issue N           issue number to fetch with gh
  --repo OWNER/REPO   repository of --issue (required with --issue)
  --labels PATH       label taxonomy (default ${LABELS_FILE})
  --apply             execute the plan through gh api

exit codes: 0 plan computed (and applied), 1 a form answer did not map to a
label, 2 bad arguments, unreadable input, or an API failure`;

/** The way this script reaches GitHub; tests pass a stub. */
export interface GhRunner {
    gh(args: string[], stdin?: string): Promise<Output>;
}

export const defaultRunner: GhRunner = {
    gh: (args, stdin) => runAsync("gh", args, stdin === undefined ? {} : { stdin }),
};

/** An input file or API response is not the shape this script relies on; main() reports it and exits 2. */
export class InputError extends Error {}

export interface Issue {
    number: number;
    body: string;
    labels: string[];
}

export interface Plan {
    number: number;
    add: string[];
    remove: string[];
}

// Python's str.isspace() set, which `\s` and str.strip() use in the original; JavaScript's `\s` and trim()
// differ in a few code points (U+001C–U+001F, U+0085, U+FEFF).
const WS = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const STRIP_RE = new RegExp(`^[${WS}]+|[${WS}]+$`, "g");

function pyStrip(text: string): string {
    return text.replace(STRIP_RE, "");
}

function stripChars(text: string, chars: string, start: boolean, end: boolean): string {
    let i = 0;
    let j = text.length;
    while (start && i < j && chars.includes(text[i])) i++;
    while (end && j > i && chars.includes(text[j - 1])) j--;
    return text.slice(i, j);
}

function escapeRegExp(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Code-point order, as Python's sorted() on str. */
function byCodePoint(a: string, b: string): number {
    const x = [...a].map((c) => c.codePointAt(0)!);
    const y = [...b].map((c) => c.codePointAt(0)!);
    for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
    return x.length - y.length;
}

/** Returns the answers under `### <name>`: [] for no answer, null if the heading is absent. */
export function sectionValues(body: string, name: string): string[] | null {
    // Python's MULTILINE ^ and $ (line boundaries at "\n" only) and \Z, spelled out for JavaScript.
    const bol = "(?:(?<![\\s\\S])|(?<=\\n))";
    const eol = "(?=\\n|(?![\\s\\S]))";
    const pattern = new RegExp(
        `${bol}### ${escapeRegExp(name)}[${WS}]*${eol}\\n([\\s\\S]*?)(?=${bol}### [^${WS}]|(?![\\s\\S]))`,
        "iu",
    );
    const match = pattern.exec(body);
    if (!match) return null;
    const raw = pyStrip(match[1]);
    if (!raw || raw === "_No response_") return [];
    const values: string[] = [];
    for (const piece of raw.split(/[\n,]/)) {
        const value = stripChars(pyStrip(stripChars(pyStrip(piece), "-*", true, false)), "`", true, true)
            .toLowerCase();
        if (value) values.push(value);
    }
    return values;
}

/** The label names declared in the taxonomy file. */
export function loadLabelNames(path: string): Set<string> {
    let data: unknown;
    try {
        data = JSON.parse(readText(path));
    } catch (error) {
        throw new InputError(`cannot read ${path}: ${(error as Error).message}`);
    }
    if (!Array.isArray(data)) throw new InputError(`${path} must be a JSON array of label objects`);
    const names = new Set<string>();
    data.forEach((item, index) => {
        if (typeof item !== "object" || item === null || typeof item.name !== "string") {
            throw new InputError(
                `${path} entry ${index} is not a label object with a string "name": ${JSON.stringify(item)}`,
            );
        }
        names.add(pyStrip(item.name));
    });
    return names;
}

/** Checks the fields this script reads from an issue object (the REST issue shape, also in event payloads). */
export function parseIssue(value: unknown, source: string): Issue {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
        throw new InputError(`${source} carries no issue object`);
    }
    const issue = value as Record<string, unknown>;
    if (!Number.isInteger(issue.number) || (issue.number as number) <= 0) {
        throw new InputError(`${source}: issue.number is ${JSON.stringify(issue.number)}, expected a positive integer`);
    }
    if (issue.body !== null && issue.body !== undefined && typeof issue.body !== "string") {
        throw new InputError(`${source}: issue.body is ${typeof issue.body}, expected a string or null`);
    }
    const labels = issue.labels ?? [];
    if (!Array.isArray(labels)) throw new InputError(`${source}: issue.labels is not an array`);
    const names = labels.map((label) => {
        if (typeof label !== "object" || label === null || typeof label.name !== "string") {
            throw new InputError(`${source}: issue label ${JSON.stringify(label)} has no string "name"`);
        }
        return label.name as string;
    });
    return { number: issue.number as number, body: (issue.body as string | null) ?? "", labels: names };
}

/** Reads an issues event payload: the issue and the repository it belongs to. */
export function loadIssueFromEvent(path: string): { issue: Issue; repo: string } {
    let payload: unknown;
    try {
        payload = JSON.parse(readText(path));
    } catch (error) {
        throw new InputError(`cannot read event payload ${path}: ${(error as Error).message}`);
    }
    if (typeof payload !== "object" || payload === null || typeof (payload as { issue?: unknown }).issue !== "object") {
        throw new InputError(`${path} is not an issues event payload`);
    }
    const record = payload as { issue: unknown; repository?: { full_name?: unknown } | null };
    const repo = record.repository?.full_name;
    if (typeof repo !== "string" || !repo) throw new InputError(`${path} carries no repository.full_name`);
    return { issue: parseIssue(record.issue, path), repo };
}

/** Fetches an issue through `gh api`. */
export async function loadIssueViaGh(number: number, repo: string, runner: GhRunner): Promise<Issue> {
    const args = ["api", `repos/${repo}/issues/${number}`];
    const out = await runner.gh(args);
    if (out.code !== 0) throw new CommandError(`\`gh ${args.join(" ")}\` exited ${out.code}: ${out.stderr.trim()}`);
    let data: unknown;
    try {
        data = JSON.parse(out.stdout);
    } catch (error) {
        throw new CommandError(`\`gh ${args.join(" ")}\` printed no JSON (${(error as Error).message})`);
    }
    return parseIssue(data, `gh api repos/${repo}/issues/${number}`);
}

/** Computes the label changes for one issue, and a finding for each answer with no label. */
export function plan(issue: Issue, known: Set<string>): { plan: Plan; findings: string[] } {
    const findings: string[] = [];
    const current = new Set(issue.labels);
    const add = new Set<string>();
    const remove = new Set<string>();
    for (const prefix of MANAGED_PREFIXES) {
        const section = SECTION_FOR_PREFIX[prefix];
        const values = sectionValues(issue.body, section);
        if (values === null) continue; // the form did not ask; leave this axis alone
        const desired = new Set<string>();
        for (const value of values) {
            const label = prefix + value;
            if (known.has(label)) {
                desired.add(label);
            } else {
                findings.push(
                    `'${value}' under '### ${section}' has no label '${label}' in ${LABELS_FILE}. ` +
                        `Fix: add the label to ${LABELS_FILE} and the form, or correct the issue.`,
                );
            }
        }
        const managedNow = [...current].filter((label) => label.startsWith(prefix));
        for (const label of desired) if (!managedNow.includes(label)) add.add(label);
        for (const label of managedNow) if (!desired.has(label)) remove.add(label);
    }
    const after = new Set([...current, ...add].filter((label) => !remove.has(label)));
    const hasType = TYPE_LABELS.some((label) => after.has(label));
    const priorities = [...after].filter((label) => label.startsWith("priority/"));
    if (hasType && priorities.length === 1 && after.has(NEEDS_TRIAGE)) remove.add(NEEDS_TRIAGE);
    return {
        plan: { number: issue.number, add: [...add].sort(byCodePoint), remove: [...remove].sort(byCodePoint) },
        findings,
    };
}

/** Executes a plan: one POST for the additions, one DELETE per removal (a 404 there means already removed). */
export async function apply(repo: string, number: number, add: string[], remove: string[], runner: GhRunner) {
    const base = `repos/${repo}/issues/${number}/labels`;
    if (add.length > 0) {
        const args = ["api", base, "--method", "POST", "--input", "-"];
        const out = await runner.gh(args, JSON.stringify({ labels: add }));
        if (out.code !== 0) throw new CommandError(`\`gh ${args.join(" ")}\` failed: ${out.stderr.trim()}`);
    }
    for (const label of remove) {
        const args = ["api", `${base}/${label}`, "--method", "DELETE"];
        const out = await runner.gh(args);
        if (out.code === 0) continue;
        if (out.stderr.includes("404")) continue; // removed by a concurrent run
        throw new CommandError(`\`gh ${args.join(" ")}\` failed: ${out.stderr.trim()}`);
    }
}

/** JSON as Python's json.dumps(indent=2) prints it (ensure_ascii escapes everything above U+007F). */
export function pythonJson(value: unknown): string {
    return JSON.stringify(value, null, 2).replace(
        /[\u0080-\uffff]/g,
        (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"),
    );
}

export async function main(argv: string[], runner: GhRunner = defaultRunner): Promise<number> {
    let event: string | undefined;
    let issueNumber: number | undefined;
    let repoArg: string | undefined;
    let labelsPath: string;
    let doApply: boolean;
    try {
        const args = parseArgs(argv, {
            string: ["event", "issue", "repo", "labels"],
            boolean: ["apply", "help"],
            alias: { h: "help" },
        });
        if (args.flags.help) {
            console.log(USAGE);
            return 0;
        }
        if (args.positional.length > 0) throw new UsageError(`unexpected argument ${args.positional[0]}`);
        event = args.values.event;
        repoArg = args.values.repo;
        labelsPath = args.values.labels ?? LABELS_FILE;
        doApply = args.flags.apply ?? false;
        if (event !== undefined && args.values.issue !== undefined) {
            throw new UsageError("--event and --issue are mutually exclusive");
        }
        if (event === undefined && args.values.issue === undefined) {
            throw new UsageError("one of --event or --issue is required");
        }
        if (args.values.issue !== undefined) {
            if (!/^[+-]?\d+$/.test(args.values.issue.trim())) {
                throw new UsageError(`--issue: invalid int value: '${args.values.issue}'`);
            }
            issueNumber = Number(args.values.issue.trim());
            if (!repoArg) throw new UsageError("--repo is required with --issue");
        }
    } catch (error) {
        if (!(error instanceof UsageError)) throw error;
        eprint(USAGE);
        eprint(`sync_issue_metadata: error: ${error.message}`);
        return 2;
    }

    try {
        let issue: Issue;
        let repo: string;
        if (event !== undefined) {
            ({ issue, repo } = loadIssueFromEvent(event));
        } else {
            issue = await loadIssueViaGh(issueNumber!, repoArg!, runner);
            repo = repoArg!;
        }
        const known = loadLabelNames(labelsPath);
        const result = plan(issue, known);
        let applied = false;
        if (doApply && (result.plan.add.length > 0 || result.plan.remove.length > 0)) {
            await apply(repo, result.plan.number, result.plan.add, result.plan.remove, runner);
            applied = true;
        }
        console.log(pythonJson({ ...result.plan, repo, applied }));
        for (const finding of result.findings) eprint(finding);
        return result.findings.length > 0 ? 1 : 0;
    } catch (error) {
        if (error instanceof InputError || error instanceof CommandError) {
            eprint(`sync_issue_metadata: error: ${error.message}`);
            return 2;
        }
        if (error instanceof Deno.errors.NotFound && (error as Error).message.includes("gh")) {
            eprint(`sync_issue_metadata: error: gh is not available: ${(error as Error).message}`);
            return 2;
        }
        throw error;
    }
}

if (import.meta.main) Deno.exit(await main(Deno.args));
