#!/usr/bin/env -S deno run --no-prompt --allow-run=gh
// Digests a GitHub Actions run's failures into one compact JSON object, so multi-megabyte logs stay out of
// agent context: it reads only the failed-step logs of unsuccessful jobs and keeps the last N lines per job.
// Called by agents diagnosing a red check (.agents/knowledge/github-checks.md, the change-workflow skill).
// Needs the gh CLI installed and authenticated; every gh call is read-only.
//
// Usage:
//   .agents/skills/change-workflow/scripts/run_log_digest.ts --repo OWNER/REPO --run-id ID [--tail N]
//
//   --repo    Repository as OWNER/REPO (e.g. octocat/hello-world).
//   --run-id  Numeric workflow run id (databaseId from `gh run list`).
//   --tail    Log lines to keep per failed job (default 50).
//
// Output: one JSON object on stdout:
//   {"run_id": ..., "status": ..., "conclusion": ...,
//    "failed_jobs": [{"name": ..., "job_id": ..., "failed_steps": [...], "log_tail": [...]}]}
// A job or step is unsuccessful when its conclusion is failure, timed_out, or startup_failure, or cancelled inside a run that was not itself cancelled.
// Diagnostics go to stderr.
//
// Exit codes: 0 digest produced (a green run digests to an empty failed_jobs list), 1 gh missing, gh
// errored, run not found, or a failed job's log could not be fetched, 2 bad arguments or a run JSON
// without the shape `gh run view --json databaseId,conclusion,status,jobs` documents.
import { parseArgs } from "../../../../scripts/lib/args.ts";
import { eprint, type Output, runAsync, UsageError } from "../../../../scripts/lib/repo.ts";

export const USAGE = `usage: run_log_digest.ts --repo OWNER/REPO --run-id ID [--tail N]

Digest the failed jobs of a GitHub Actions run into one small JSON object
(failed jobs, failed steps, last N log lines each).

  --repo    repository as OWNER/REPO
  --run-id  numeric workflow run id
  --tail    log lines to keep per failed job (default: 50)
  -h, --help  show this help

Exit codes: 0 digest produced (empty failed_jobs on a green run), 1 gh
missing/errored/run not found/job log unfetchable, 2 bad arguments or an
unexpected run JSON shape.`;

/** Conclusions that make a job or step unsuccessful. */
export const UNSUCCESSFUL = new Set(["failure", "timed_out", "startup_failure"]);

/** Runs gh; tests pass a stub. Throws Deno.errors.NotFound when gh is not installed. */
export interface GhRunner {
    gh(args: string[]): Promise<Output>;
}

export const realGh: GhRunner = { gh: (args) => runAsync("gh", args) };

/** The run shape gh guarantees for `--json databaseId,conclusion,status,jobs`, after validation. */
export interface Step {
    name?: string;
    conclusion?: string | null;
}
export interface Job {
    databaseId: number;
    name?: string;
    status?: string | null;
    conclusion?: string | null;
    steps?: Step[] | null;
}
export interface Run {
    databaseId?: number;
    status?: string | null;
    conclusion?: string | null;
    jobs: Job[];
}

export interface FailedJob {
    name: string;
    job_id: number;
    failed_steps: string[];
    log_tail: string[];
}
export interface Digest {
    run_id: number;
    status: string | null;
    conclusion: string | null;
    failed_jobs: FailedJob[];
}

/** A run JSON that does not have the shape gh documents; main() reports it and exits 2. */
export class ShapeError extends Error {}

export interface Options {
    repo: string;
    runId: number;
    tail: number;
}

function parsePositiveInt(name: string, value: string | undefined, fallback?: number): number {
    if (value === undefined) {
        if (fallback === undefined) throw new UsageError(`the following argument is required: --${name}`);
        return fallback;
    }
    if (!/^\s*[+-]?\d+\s*$/.test(value)) throw new UsageError(`--${name}: invalid int value: '${value}'`);
    const n = Number(value.trim());
    if (n <= 0) throw new UsageError(`--${name} must be a positive integer (got ${n})`);
    return n;
}

/** Parses argv; returns null for --help. */
export function parseOptions(argv: string[]): Options | null {
    const args = parseArgs(argv, { string: ["repo", "run-id", "tail"], boolean: ["help"], alias: { h: "help" } });
    if (args.flags.help) return null;
    if (args.positional.length) throw new UsageError(`unrecognized arguments: ${args.positional.join(" ")}`);
    const repo = args.values.repo;
    if (repo === undefined) throw new UsageError("the following argument is required: --repo");
    const runId = parsePositiveInt("run-id", args.values["run-id"]);
    const parts = repo.split("/");
    if (parts.length !== 2 || !parts.every((p) => p !== "")) {
        throw new UsageError(`--repo must be OWNER/REPO (got '${repo}'), e.g. --repo octocat/hello-world`);
    }
    const tail = parsePositiveInt("tail", args.values.tail, 50);
    return { repo, runId, tail };
}

function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Checks the parsed `gh run view --json` output before anything is built on it. */
export function validateRun(value: unknown): Run {
    const iface = "`gh run view --json databaseId,conclusion,status,jobs`";
    if (!isObject(value)) throw new ShapeError(`${iface} returned ${JSON.stringify(value)}, not an object`);
    if (!Array.isArray(value.jobs)) {
        throw new ShapeError(`${iface} returned no jobs array (jobs: ${JSON.stringify(value.jobs)})`);
    }
    value.jobs.forEach((job, i) => {
        if (!isObject(job)) throw new ShapeError(`${iface} jobs[${i}] is ${JSON.stringify(job)}, not an object`);
        if (typeof job.databaseId !== "number") {
            throw new ShapeError(`${iface} jobs[${i}] has databaseId ${JSON.stringify(job.databaseId)}, not a number`);
        }
        if (job.steps != null && !Array.isArray(job.steps)) {
            throw new ShapeError(`${iface} jobs[${i}].steps is ${JSON.stringify(job.steps)}, not an array`);
        }
    });
    return value as unknown as Run;
}

/** The jobs to digest, and a note when the run failed but no job did (then every completed job). */
export function selectJobs(run: Run): { jobs: Job[]; note?: string } {
    // A cancelled job counts unless the whole run was cancelled (superseded): a job past its
    // timeout-minutes can conclude cancelled inside a failed run.
    const failed = run.jobs.filter((j) =>
        UNSUCCESSFUL.has(j.conclusion ?? "") || (j.conclusion === "cancelled" && run.conclusion !== "cancelled")
    );
    if (failed.length || !UNSUCCESSFUL.has(run.conclusion ?? "")) return { jobs: failed };
    const completed = run.jobs.filter((j) => j.status === "completed");
    return {
        jobs: completed,
        note: `note: run concluded ${run.conclusion} but no job is unsuccessful; ` +
            `falling back to all ${completed.length} completed job(s).`,
    };
}

/** Python's str.splitlines(): every line boundary it recognizes, no trailing empty line. */
export function splitLines(text: string): string[] {
    // deno-lint-ignore no-control-regex -- Python ends lines at the file, group, and record separators too.
    const lines = text.split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/);
    if (lines.length && lines[lines.length - 1] === "") lines.pop();
    return lines;
}

/** JSON with two-space indent and non-ASCII escaped, byte-identical to Python's json.dump(indent=2). */
export function toJson(value: unknown): string {
    return JSON.stringify(value, null, 2).replace(
        /[\u0080-\uffff]/g,
        (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"),
    );
}

/** A gh call failed; `message` holds the diagnostic lines and main() exits 1. */
export class GhError extends Error {}

const GH_MISSING = "error: gh not found — install and authenticate the GitHub CLI " +
    "(https://cli.github.com/), or use the MCP column of the skill's Actions table instead.";

async function callGh(runner: GhRunner, args: string[]): Promise<Output> {
    try {
        return await runner.gh(args);
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) throw new GhError(GH_MISSING);
        throw error;
    }
}

/** Builds the digest; throws GhError (exit 1) or ShapeError (exit 2). */
export async function digest(opts: Options, runner: GhRunner, log: (line: string) => void = eprint): Promise<Digest> {
    const { repo, runId, tail } = opts;
    const view = await callGh(runner, [
        "run",
        "view",
        String(runId),
        "-R",
        repo,
        "--json",
        "databaseId,conclusion,status,jobs",
    ]);
    if (view.code !== 0) {
        throw new GhError(
            `error: \`gh run view ${runId}\` failed (exit ${view.code}): ${view.stderr.trim()}\n` +
                `fix: check that run ${runId} exists in ${repo} (\`gh run list -R ${repo} --limit 20\`) ` +
                "and that gh is authenticated (`gh auth status`).",
        );
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(view.stdout);
    } catch (error) {
        throw new GhError(
            `error: could not parse gh JSON output: ${(error as Error).message}\n` +
                "fix: rerun the command; if it persists, update gh.",
        );
    }
    const run = validateRun(parsed);
    const selected = selectJobs(run);
    if (selected.note) log(selected.note);

    const failedJobs: FailedJob[] = [];
    for (const job of selected.jobs) {
        const name = job.name ?? "";
        const failedSteps = (job.steps ?? []).filter((s) => UNSUCCESSFUL.has(s.conclusion ?? "")).map((s) =>
            s.name ?? ""
        );
        const logArgs = ["run", "view", "-R", repo, "--job", String(job.databaseId), "--log-failed"];
        const out = await callGh(runner, logArgs);
        if (out.code !== 0) {
            throw new GhError(
                `error: could not fetch the failed-step log of job ${job.databaseId} (${name}): ` +
                    `\`gh ${logArgs.join(" ")}\` exited ${out.code}: ${out.stderr.trim() || "(no stderr)"}\n` +
                    `fix: open the job in the browser (\`gh run view ${runId} -R ${repo} --web\`), or rerun ` +
                    "once the run has finished and its logs are available.",
            );
        }
        failedJobs.push({
            name,
            job_id: job.databaseId,
            failed_steps: failedSteps,
            log_tail: splitLines(out.stdout).slice(-tail),
        });
    }
    return {
        run_id: run.databaseId ?? runId,
        status: run.status ?? null,
        conclusion: run.conclusion ?? null,
        failed_jobs: failedJobs,
    };
}

export async function main(argv: string[], runner: GhRunner = realGh): Promise<number> {
    let opts: Options | null;
    try {
        opts = parseOptions(argv);
    } catch (error) {
        if (!(error instanceof UsageError)) throw error;
        eprint(USAGE.split("\n")[0]);
        eprint(`run_log_digest.ts: error: ${error.message}`);
        return 2;
    }
    if (opts === null) {
        console.log(USAGE);
        return 0;
    }
    try {
        const result = await digest(opts, runner);
        console.log(toJson(result));
        return 0;
    } catch (error) {
        if (error instanceof GhError) {
            eprint(error.message);
            return 1;
        }
        if (error instanceof ShapeError) {
            eprint(`error: ${error.message}`);
            eprint("fix: check the installed gh version (`gh --version`) against the gh run view JSON fields.");
            return 2;
        }
        throw error;
    }
}

if (import.meta.main) Deno.exit(await main(Deno.args));
