#!/usr/bin/env -S deno run --no-prompt --allow-run=gh
// Lists the review threads of one pull request of this repository as JSON, through one fixed read-only
// GraphQL query. It exists so agents can read review threads without an allow rule on `gh api graphql`,
// which would also match mutations (.agents/knowledge/agent-authority.md); it takes no query text.
//
// Usage: scripts/review_threads.ts <number>    (called by `just review-threads <number>`)
//
// Prints a JSON list, one object per thread (the first 100): isResolved, isOutdated, path, line, and the
// first comment's author login, createdAt, and the first 200 characters of its body. Exit 0 on success,
// 2 on a usage error or when gh or its response is not what this script relies on.
import { CommandError, eprint, type Output, REPO, runAsync, UsageError } from "./lib/repo.ts";

export const USAGE = `usage: review_threads.ts <number>

Lists the review threads of pull request <number> of ${REPO} as JSON (the first 100 threads).`;

/** The one query this script sends; callers supply only the pull request number. */
export const QUERY = `query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
        totalCount
        nodes {
          isResolved
          isOutdated
          path
          line
          comments(first: 1) {
            nodes { author { login } createdAt body }
          }
        }
      }
    }
  }
}`;

export const BODY_LIMIT = 200;
const MAX_INT = 2 ** 31 - 1; // GraphQL Int

/** Runs gh with the given arguments; tests replace it. */
export type GhRunner = (args: string[]) => Promise<Output>;

export const runGh: GhRunner = (args) => runAsync("gh", args);

export interface Thread {
    isResolved: boolean;
    isOutdated: boolean;
    path: string;
    line: number | null;
    author: string | null;
    createdAt: string;
    body: string;
}

/** Parses the pull request number; anything but a positive integer is a UsageError. */
export function parseNumber(argv: string[]): number {
    if (argv.length !== 1) throw new UsageError(`expected exactly one argument, got ${argv.length}`);
    const arg = argv[0];
    if (!/^[1-9][0-9]*$/.test(arg) || Number(arg) > MAX_INT) {
        throw new UsageError(`the pull request number must be a positive integer, got ${JSON.stringify(arg)}`);
    }
    return Number(arg);
}

/** The gh arguments for the fixed query. */
export function ghArgs(number: number): string[] {
    const [owner, name] = REPO.split("/");
    return [
        "api",
        "graphql",
        "-F",
        `number=${number}`,
        "-f",
        `owner=${owner}`,
        "-f",
        `name=${name}`,
        "-f",
        `query=${QUERY}`,
    ];
}

function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function shapeError(where: string, value: unknown): CommandError {
    return new CommandError(
        `gh api graphql response: ${where} is ${JSON.stringify(value)}; ` +
            `expected the shape of the query in scripts/review_threads.ts — check gh and the GitHub GraphQL schema`,
    );
}

/** The first BODY_LIMIT characters (code points) of a comment body. */
export function truncate(body: string): string {
    const chars = Array.from(body);
    return chars.length <= BODY_LIMIT ? body : chars.slice(0, BODY_LIMIT).join("");
}

/** Checks the response shape and extracts the threads; returns them with the total thread count. */
export function parseResponse(stdout: string): { threads: Thread[]; totalCount: number } {
    let data: unknown;
    try {
        data = JSON.parse(stdout);
    } catch {
        throw new CommandError(`gh api graphql printed no JSON: ${JSON.stringify(stdout.slice(0, 200))}`);
    }
    if (!isObject(data)) throw shapeError("the response", data);
    if (data.errors !== undefined) {
        throw new CommandError(`gh api graphql returned errors: ${JSON.stringify(data.errors)}`);
    }
    const repository = isObject(data.data) ? data.data.repository : undefined;
    if (!isObject(repository)) throw shapeError("data.repository", repository);
    const pr = repository.pullRequest;
    if (!isObject(pr)) throw shapeError("data.repository.pullRequest", pr);
    const rt = pr.reviewThreads;
    if (!isObject(rt) || !Array.isArray(rt.nodes) || typeof rt.totalCount !== "number") {
        throw shapeError("pullRequest.reviewThreads", rt);
    }
    const threads = rt.nodes.map((node: unknown, i: number): Thread => {
        const where = `reviewThreads.nodes[${i}]`;
        if (
            !isObject(node) || typeof node.isResolved !== "boolean" || typeof node.isOutdated !== "boolean" ||
            typeof node.path !== "string" || !(node.line === null || typeof node.line === "number")
        ) throw shapeError(where, node);
        const comments = node.comments;
        if (!isObject(comments) || !Array.isArray(comments.nodes) || comments.nodes.length !== 1) {
            throw shapeError(`${where}.comments`, comments);
        }
        const first: unknown = comments.nodes[0];
        // author is null when the account was deleted (GitHub's schema documents it as nullable).
        if (
            !isObject(first) || typeof first.createdAt !== "string" || typeof first.body !== "string" ||
            !(first.author === null || (isObject(first.author) && typeof first.author.login === "string"))
        ) throw shapeError(`${where}.comments.nodes[0]`, first);
        const author = first.author as { login: string } | null;
        return {
            isResolved: node.isResolved,
            isOutdated: node.isOutdated,
            path: node.path,
            line: node.line as number | null,
            author: author === null ? null : author.login,
            createdAt: first.createdAt,
            body: truncate(first.body),
        };
    });
    return { threads, totalCount: rt.totalCount };
}

/** Fetches the threads of one pull request through the fixed query. */
export async function fetchThreads(number: number, gh: GhRunner = runGh): Promise<Thread[]> {
    const out = await gh(ghArgs(number));
    if (out.code !== 0) {
        throw new CommandError(
            `\`gh api graphql\` (review threads of #${number}) exited ${out.code}: ${out.stderr.trim()}`,
        );
    }
    const { threads, totalCount } = parseResponse(out.stdout);
    if (totalCount > threads.length) {
        eprint(`review_threads: #${number} has ${totalCount} review threads; listing the first ${threads.length}`);
    }
    return threads;
}

export async function main(argv: string[], gh: GhRunner = runGh): Promise<number> {
    if (argv.includes("--help") || argv.includes("-h")) {
        console.log(USAGE);
        return 0;
    }
    try {
        const number = parseNumber(argv);
        console.log(JSON.stringify(await fetchThreads(number, gh), null, 2));
        return 0;
    } catch (error) {
        if (error instanceof UsageError) {
            eprint(`review_threads: ${error.message}\n\n${USAGE}`);
            return 2;
        }
        if (error instanceof CommandError) {
            eprint(`review_threads: ${error.message}`);
            return 2;
        }
        throw error;
    }
}

if (import.meta.main) Deno.exit(await main(Deno.args));
