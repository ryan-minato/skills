import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { CommandError, type Output, UsageError } from "./lib/repo.ts";
import {
    fetchThreads,
    ghArgs,
    main,
    MAX_PAGES,
    parseNumber,
    parseResponse,
    QUERY,
    truncate,
} from "./review_threads.ts";

function response(nodes: unknown[], endCursor: string | null = null): string {
    const pageInfo = { hasNextPage: endCursor !== null, endCursor };
    return JSON.stringify({ data: { repository: { pullRequest: { reviewThreads: { pageInfo, nodes } } } } });
}

function thread(overrides: Record<string, unknown> = {}, first: Record<string, unknown> = {}): unknown {
    return {
        isResolved: false,
        isOutdated: false,
        path: "scripts/a.ts",
        line: 12,
        comments: {
            totalCount: 1,
            nodes: [{ author: { login: "octo" }, createdAt: "2026-10-01T00:00:00Z", body: "fix this", ...first }],
        },
        ...overrides,
    };
}

function stub(out: Partial<Output>, calls: string[][] = []) {
    return (args: string[]): Promise<Output> => {
        calls.push(args);
        return Promise.resolve({ code: 0, stdout: "", stderr: "", ...out });
    };
}

/** A runner that answers the i-th call with the i-th stdout (the last one repeats). */
function pages(stdouts: string[], calls: string[][] = []) {
    return (args: string[]): Promise<Output> => {
        calls.push(args);
        return Promise.resolve({ code: 0, stdout: stdouts[Math.min(calls.length, stdouts.length) - 1], stderr: "" });
    };
}

async function rejection(promise: Promise<unknown>): Promise<CommandError> {
    const error = await promise.then(() => undefined, (e) => e);
    if (!(error instanceof CommandError)) throw new Error(`expected CommandError, got ${error}`);
    return error;
}

async function capture(fn: () => Promise<number>): Promise<{ code: number; out: string; err: string }> {
    const log = console.log, error = console.error;
    let out = "", err = "";
    console.log = (m: string) => out += m + "\n";
    console.error = (m: string) => err += m + "\n";
    try {
        return { code: await fn(), out, err };
    } finally {
        console.log = log;
        console.error = error;
    }
}

Deno.test("parseNumber accepts a positive integer only", () => {
    assertEquals(parseNumber(["117"]), 117);
    for (const bad of ["0", "-1", "abc", "1.5", "017", "1e3", "", " 1", "2147483648", "#117"]) {
        assertThrows(() => parseNumber([bad]), UsageError);
    }
    assertThrows(() => parseNumber([]), UsageError);
    assertThrows(() => parseNumber(["1", "2"]), UsageError);
});

Deno.test("ghArgs sends the fixed query with the number as a typed field", () => {
    const args = ghArgs(117);
    assertEquals(args.slice(0, 8), [
        "api",
        "graphql",
        "-F",
        "number=117",
        "-f",
        "owner=ryan-minato",
        "-f",
        "name=skills",
    ]);
    assertEquals(args.slice(8), ["-f", `query=${QUERY}`]);
    assertEquals(ghArgs(117, "Y3Vyc29y").slice(8), ["-f", "after=Y3Vyc29y", "-f", `query=${QUERY}`]);
    assertEquals(QUERY.includes("mutation"), false);
});

Deno.test("truncate keeps the first 200 code points", () => {
    assertEquals(truncate("short"), "short");
    assertEquals(truncate("x".repeat(250)).length, 200);
    assertEquals(Array.from(truncate("😀".repeat(201))).length, 200);
});

Deno.test("parseResponse extracts the threads", () => {
    const { threads, hasNextPage, endCursor } = parseResponse(response([
        thread(),
        thread({ isResolved: true, isOutdated: true, line: null }, { author: null, body: "y".repeat(300) }),
    ]));
    assertEquals(hasNextPage, false);
    assertEquals(endCursor, null);
    assertEquals(threads[0], {
        isResolved: false,
        isOutdated: false,
        path: "scripts/a.ts",
        line: 12,
        author: "octo",
        createdAt: "2026-10-01T00:00:00Z",
        body: "fix this",
        comments: [{ author: "octo", createdAt: "2026-10-01T00:00:00Z", body: "fix this" }],
    });
    assertEquals(threads[1].author, null);
    assertEquals(threads[1].line, null);
    assertEquals(threads[1].body.length, 200);
    assertEquals(threads[1].comments[0].body.length, 200);
});

Deno.test("parseResponse returns every comment of a thread, the first also at the top level", () => {
    const nodes = [
        { author: { login: "reviewer" }, createdAt: "2026-10-01T00:00:00Z", body: "fix this" },
        { author: { login: "author" }, createdAt: "2026-10-02T00:00:00Z", body: "z".repeat(300) },
        { author: null, createdAt: "2026-10-03T00:00:00Z", body: "done" },
    ];
    const { threads } = parseResponse(response([thread({ comments: { totalCount: 3, nodes } })]));
    assertEquals(threads[0].author, "reviewer");
    assertEquals(threads[0].body, "fix this");
    assertEquals(threads[0].comments.map((c) => c.author), ["reviewer", "author", null]);
    assertEquals(threads[0].comments[1].body.length, 200);
    assertEquals(threads[0].comments[2], { author: null, createdAt: "2026-10-03T00:00:00Z", body: "done" });
});

Deno.test("parseResponse fails on a thread with more than 100 comments", () => {
    const nodes = Array.from({ length: 100 }, () => ({ author: null, createdAt: "t", body: "b" }));
    const error = assertThrows(
        () => parseResponse(response([thread({ comments: { totalCount: 101, nodes } })])),
        CommandError,
    );
    assertStringIncludes(error.message, "scripts/a.ts:12 has 101 comments");
});

Deno.test("parseResponse with no threads is an empty list", () => {
    assertEquals(parseResponse(response([])).threads, []);
});

Deno.test("parseResponse rejects a malformed or incomplete response", () => {
    const cases = [
        "not json",
        "[]",
        JSON.stringify({ errors: [{ message: "boom" }] }),
        JSON.stringify({}),
        JSON.stringify({ data: { repository: null } }),
        JSON.stringify({ data: { repository: { pullRequest: null } } }),
        JSON.stringify({ data: { repository: { pullRequest: { reviewThreads: { nodes: [] } } } } }),
        JSON.stringify({
            data: {
                repository: {
                    pullRequest: { reviewThreads: { pageInfo: { hasNextPage: true, endCursor: null }, nodes: [] } },
                },
            },
        }),
        response([thread({ isResolved: "no" })]),
        response([thread({ path: undefined })]),
        response([thread({ line: "12" })]),
        response([thread({ comments: { totalCount: 0, nodes: [] } })]),
        response([thread({ comments: { nodes: [{ author: null, createdAt: "t", body: "b" }] } })]),
        response([thread({}, { createdAt: undefined })]),
        response([thread({}, { body: null })]),
        response([thread({}, { author: {} })]),
    ];
    for (const c of cases) assertThrows(() => parseResponse(c), CommandError);
});

Deno.test("fetchThreads fails on a gh error, naming the exit and stderr", async () => {
    const gh = stub({ code: 1, stderr: "GraphQL: Could not resolve to a PullRequest with the number of 99999." });
    const error = await rejection(fetchThreads(99999, gh));
    assertStringIncludes(error.message, "exited 1");
    assertStringIncludes(error.message, "Could not resolve");
});

Deno.test("main prints the threads as JSON and exits 0", async () => {
    const calls: string[][] = [];
    const r = await capture(() => main(["117"], stub({ stdout: response([thread()]) }, calls)));
    assertEquals(r.code, 0);
    assertEquals(calls.length, 1);
    assertEquals(JSON.parse(r.out)[0].author, "octo");
    assertEquals(r.err, "");
});

Deno.test("fetchThreads pages through every thread, passing the cursor after the first page", async () => {
    const calls: string[][] = [];
    const gh = pages(
        [response([thread({ path: "p1.ts" })], "Y3Vyc29yOjE="), response([thread({ path: "p2.ts" })])],
        calls,
    );
    const threads = await fetchThreads(117, gh);
    assertEquals(threads.map((t) => t.path), ["p1.ts", "p2.ts"]);
    assertEquals(calls.length, 2);
    assertEquals(calls[0].some((a) => a.startsWith("after=")), false);
    assertEquals(calls[1].includes("after=Y3Vyc29yOjE="), true);
    assertEquals(calls[1][calls[1].indexOf("after=Y3Vyc29yOjE=") - 1], "-f");
});

Deno.test("fetchThreads fails, naming the cap, instead of reading more than MAX_PAGES pages", async () => {
    const calls: string[][] = [];
    const error = await rejection(fetchThreads(117, pages([response([thread()], "more")], calls)));
    assertEquals(calls.length, MAX_PAGES);
    assertStringIncludes(error.message, `more than ${MAX_PAGES} pages`);
});

Deno.test("main exits 2 on a thread with more than 100 comments", async () => {
    const nodes = Array.from({ length: 100 }, () => ({ author: null, createdAt: "t", body: "b" }));
    const out = response([thread({ comments: { totalCount: 150, nodes } })]);
    const r = await capture(() => main(["5"], stub({ stdout: out })));
    assertEquals(r.code, 2);
    assertEquals(r.out, "");
    assertStringIncludes(r.err, "has 150 comments");
});

Deno.test("main refuses a non-numeric argument without calling gh", async () => {
    const calls: string[][] = [];
    for (const argv of [["abc"], [], ["1", "2"], ["--query", "x"]]) {
        const r = await capture(() => main(argv, stub({}, calls)));
        assertEquals(r.code, 2);
        assertStringIncludes(r.err, "usage:");
        assertEquals(r.out, "");
    }
    assertEquals(calls.length, 0);
});

Deno.test("main exits 2 on a gh failure, an errors payload, or missing data", async () => {
    for (
        const out of [
            { code: 1, stderr: "HTTP 401" },
            { stdout: JSON.stringify({ errors: [{ message: "x" }] }) },
            { stdout: JSON.stringify({ data: null }) },
            { stdout: "" },
        ]
    ) {
        const r = await capture(() => main(["117"], stub(out)));
        assertEquals(r.code, 2);
        assertEquals(r.out, "");
        assertStringIncludes(r.err, "review_threads:");
    }
});

Deno.test("--help prints usage to stdout and exits 0", async () => {
    const r = await capture(() => main(["--help"], stub({})));
    assertEquals(r.code, 0);
    assertStringIncludes(r.out, "usage: review_threads.ts <number>");
});
