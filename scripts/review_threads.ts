#!/usr/bin/env -S deno run --no-prompt --allow-run=gh
// Lists the review threads of one pull request of this repository as JSON, through one fixed read-only
// GraphQL query. It exists so agents can read review threads without an allow rule on `gh api graphql`,
// which would also match mutations (.agents/knowledge/agent-authority.md); it takes no query text.
//
// Usage: scripts/review_threads.ts <number>    (called by `just review-threads <number>`)
//
// Pages through every thread (100 per request, at most MAX_PAGES requests) and prints a JSON list, one
// object per thread: isResolved, isOutdated, path, line; the first comment's author login, createdAt, and
// the first 200 characters of its body; and `comments`, every comment of the thread as {author, createdAt,
// body} with the same 200-character limit. Exit 0 on success, 2 on a usage error, when gh or its response
// is not what this script relies on, or when a thread has more than 100 comments or the pull request more
// than MAX_PAGES pages of threads (it fails rather than truncate).
import { CommandError, eprint, type Output, REPO, runAsync, UsageError } from "./lib/repo.ts";

export const USAGE = `usage: review_threads.ts <number>

Lists the review threads of pull request <number> of ${REPO} as JSON: every thread
with every comment (bodies cut to the first 200 characters).`;

/** The one query this script sends; callers supply only the pull request number. */
export const QUERY = `query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          isResolved
          isOutdated
          path
          line
          comments(first: 100) {
            totalCount
            nodes { author { login } createdAt body }
          }
        }
      }
    }
  }
}`;

export const BODY_LIMIT = 200;
/** Comments requested per thread; a thread with more fails instead of being truncated. */
export const COMMENT_LIMIT = 100;
/** Pages of 100 threads fetched at most; a pull request with more fails instead of being truncated. */
export const MAX_PAGES = 50;
const MAX_INT = 2 ** 31 - 1; // GraphQL Int

/** Runs gh with the given arguments; tests replace it. */
export type GhRunner = (args: string[]) => Promise<Output>;

export const runGh: GhRunner = (args) => runAsync("gh", args);

export interface Comment {
    author: string | null;
    createdAt: string;
    body: string;
}

export interface Thread {
    isResolved: boolean;
    isOutdated: boolean;
    path: string;
    line: number | null;
    author: string | null;
    createdAt: string;
    body: string;
    comments: Comment[];
}

/** One page of threads and where the next page starts. */
export interface Page {
    threads: Thread[];
    hasNextPage: boolean;
    endCursor: string | null;
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

/** The gh arguments for the fixed query; `after` is the cursor of the previous page, absent on the first. */
export function ghArgs(number: number, after?: string): string[] {
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
        ...(after === undefined ? [] : ["-f", `after=${after}`]),
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

function parseComment(value: unknown, where: string): Comment {
    // author is null when the account was deleted (GitHub's schema documents it as nullable).
    if (
        !isObject(value) || typeof value.createdAt !== "string" || typeof value.body !== "string" ||
        !(value.author === null || (isObject(value.author) && typeof value.author.login === "string"))
    ) throw shapeError(where, value);
    const author = value.author as { login: string } | null;
    return { author: author === null ? null : author.login, createdAt: value.createdAt, body: truncate(value.body) };
}

/** Checks the response shape of one page and extracts its threads and page info. */
export function parseResponse(stdout: string): Page {
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
    if (!isObject(rt) || !Array.isArray(rt.nodes)) throw shapeError("pullRequest.reviewThreads", rt);
    const pageInfo = rt.pageInfo;
    if (
        !isObject(pageInfo) || typeof pageInfo.hasNextPage !== "boolean" ||
        !(typeof pageInfo.endCursor === "string" || (pageInfo.endCursor === null && !pageInfo.hasNextPage))
    ) throw shapeError("reviewThreads.pageInfo", pageInfo);
    const threads = rt.nodes.map((node: unknown, i: number): Thread => {
        const where = `reviewThreads.nodes[${i}]`;
        if (
            !isObject(node) || typeof node.isResolved !== "boolean" || typeof node.isOutdated !== "boolean" ||
            typeof node.path !== "string" || !(node.line === null || typeof node.line === "number")
        ) throw shapeError(where, node);
        const c = node.comments;
        if (
            !isObject(c) || !Array.isArray(c.nodes) || c.nodes.length === 0 || typeof c.totalCount !== "number"
        ) throw shapeError(`${where}.comments`, c);
        if (c.totalCount > c.nodes.length) {
            throw new CommandError(
                `the review thread on ${node.path}:${node.line} has ${c.totalCount} comments; ` +
                    `this script reads at most ${COMMENT_LIMIT} per thread and refuses to truncate`,
            );
        }
        const comments = c.nodes.map((comment: unknown, j: number) =>
            parseComment(comment, `${where}.comments.nodes[${j}]`)
        );
        return {
            isResolved: node.isResolved,
            isOutdated: node.isOutdated,
            path: node.path,
            line: node.line as number | null,
            ...comments[0],
            comments,
        };
    });
    return { threads, hasNextPage: pageInfo.hasNextPage, endCursor: pageInfo.endCursor as string | null };
}

/** Fetches every thread of one pull request through the fixed query, one page of 100 at a time. */
export async function fetchThreads(number: number, gh: GhRunner = runGh): Promise<Thread[]> {
    const threads: Thread[] = [];
    let after: string | undefined;
    for (let page = 1; page <= MAX_PAGES; page++) {
        const out = await gh(ghArgs(number, after));
        if (out.code !== 0) {
            throw new CommandError(
                `\`gh api graphql\` (review threads of #${number}, page ${page}) exited ${out.code}: ${out.stderr.trim()}`,
            );
        }
        const result = parseResponse(out.stdout);
        threads.push(...result.threads);
        if (!result.hasNextPage) return threads;
        after = result.endCursor as string;
    }
    throw new CommandError(
        `#${number} has more than ${MAX_PAGES} pages of review threads (MAX_PAGES); refusing to truncate`,
    );
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
