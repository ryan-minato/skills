import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { CommandError, type Output, UsageError } from "./lib/repo.ts";
import { fetchThreads, ghArgs, main, parseNumber, parseResponse, QUERY, truncate } from "./review_threads.ts";

function response(nodes: unknown[], totalCount = nodes.length): string {
    return JSON.stringify({ data: { repository: { pullRequest: { reviewThreads: { totalCount, nodes } } } } });
}

function thread(overrides: Record<string, unknown> = {}, first: Record<string, unknown> = {}): unknown {
    return {
        isResolved: false,
        isOutdated: false,
        path: "scripts/a.ts",
        line: 12,
        comments: {
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
    assertEquals(QUERY.includes("mutation"), false);
});

Deno.test("truncate keeps the first 200 code points", () => {
    assertEquals(truncate("short"), "short");
    assertEquals(truncate("x".repeat(250)).length, 200);
    assertEquals(Array.from(truncate("😀".repeat(201))).length, 200);
});

Deno.test("parseResponse extracts the threads", () => {
    const { threads, totalCount } = parseResponse(response([
        thread(),
        thread({ isResolved: true, isOutdated: true, line: null }, { author: null, body: "y".repeat(300) }),
    ]));
    assertEquals(totalCount, 2);
    assertEquals(threads[0], {
        isResolved: false,
        isOutdated: false,
        path: "scripts/a.ts",
        line: 12,
        author: "octo",
        createdAt: "2026-10-01T00:00:00Z",
        body: "fix this",
    });
    assertEquals(threads[1].author, null);
    assertEquals(threads[1].line, null);
    assertEquals(threads[1].body.length, 200);
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
        response([thread({ isResolved: "no" })]),
        response([thread({ path: undefined })]),
        response([thread({ line: "12" })]),
        response([thread({ comments: { nodes: [] } })]),
        response([thread({}, { createdAt: undefined })]),
        response([thread({}, { body: null })]),
        response([thread({}, { author: {} })]),
    ];
    for (const c of cases) assertThrows(() => parseResponse(c), CommandError);
});

Deno.test("fetchThreads fails on a gh error, naming the exit and stderr", async () => {
    const gh = stub({ code: 1, stderr: "GraphQL: Could not resolve to a PullRequest with the number of 99999." });
    const error = await fetchThreads(99999, gh).then(() => undefined, (e) => e);
    if (!(error instanceof CommandError)) throw new Error(`expected CommandError, got ${error}`);
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

Deno.test("main warns on stderr when more than 100 threads exist", async () => {
    const r = await capture(() => main(["5"], stub({ stdout: response([thread()], 150) })));
    assertEquals(r.code, 0);
    assertStringIncludes(r.err, "has 150 review threads; listing the first 1");
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
