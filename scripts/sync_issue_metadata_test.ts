import { assert, assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import type { Output } from "./lib/repo.ts";
import {
    apply,
    type GhRunner,
    InputError,
    type Issue,
    loadIssueFromEvent,
    loadIssueViaGh,
    loadLabelNames,
    main,
    NEEDS_TRIAGE,
    parseIssue,
    plan,
    pythonJson,
    sectionValues,
} from "./sync_issue_metadata.ts";
import { CommandError } from "./lib/repo.ts";

const KNOWN = new Set([
    "bug",
    "enhancement",
    "task",
    "priority/critical",
    "priority/high",
    "priority/medium",
    "priority/low",
    "catalog/core",
    "catalog/meta",
    "catalog/sdd",
    NEEDS_TRIAGE,
]);

function form(priority: string, catalog: string): string {
    return `### Summary\n\nx\n\n### Priority\n\n${priority}\n\n### Catalog\n\n${catalog}\n\n### Notes\n\n_No response_`;
}

function issue(body: string, labels: string[]): Issue {
    return { number: 7, body, labels };
}

/** A gh stub that records its calls and answers from a list of responses (default: success, empty output). */
function stub(responses: Partial<Output>[] = []): GhRunner & { calls: { args: string[]; stdin?: string }[] } {
    const calls: { args: string[]; stdin?: string }[] = [];
    return {
        calls,
        gh(args, stdin) {
            calls.push({ args, stdin });
            const r = responses.shift() ?? {};
            return Promise.resolve({ code: r.code ?? 0, stdout: r.stdout ?? "", stderr: r.stderr ?? "" });
        },
    };
}

function tempDir(): string {
    return Deno.makeTempDirSync({ prefix: "sync_issue_metadata_test_" });
}

function writeJson(dir: string, name: string, value: unknown): string {
    const path = `${dir}/${name}`;
    Deno.writeTextFileSync(path, JSON.stringify(value));
    return path;
}

async function capture(fn: () => Promise<number>): Promise<{ code: number; stdout: string; stderr: string }> {
    const out: string[] = [];
    const err: string[] = [];
    const log = console.log;
    const error = console.error;
    console.log = (...a: unknown[]) => out.push(a.join(" "));
    console.error = (...a: unknown[]) => err.push(a.join(" "));
    try {
        const code = await fn();
        return { code, stdout: out.join("\n"), stderr: err.join("\n") };
    } finally {
        console.log = log;
        console.error = error;
    }
}

// --- sectionValues -------------------------------------------------------------------------------------------

Deno.test("sectionValues: absent heading is null, empty or _No response_ is []", () => {
    assertEquals(sectionValues("plain", "Priority"), null);
    assertEquals(sectionValues("###Priority\n\nhigh", "Priority"), null);
    assertEquals(sectionValues("text\n### Priority", "Priority"), null); // heading needs a following newline
    assertEquals(sectionValues("### Priority\n\n_No response_\n", "Priority"), []);
    assertEquals(sectionValues("### Priority\n\n### Catalog\n\ncore", "Priority"), []);
});

Deno.test("sectionValues: splits on commas and newlines, strips bullets and backticks, lowercases", () => {
    assertEquals(sectionValues(form("medium", "- `Core`\n* meta, sdd\n\n-  `x`"), "Catalog"), [
        "core",
        "meta",
        "sdd",
        "x",
    ]);
});

Deno.test("sectionValues: heading is case-insensitive, tolerates trailing space and CRLF", () => {
    assertEquals(sectionValues("### PRIORITY   \n\nHigh\n", "Priority"), ["high"]);
    assertEquals(sectionValues(form("low", "core, meta").replaceAll("\n", "\r\n"), "Catalog"), ["core", "meta"]);
});

Deno.test("sectionValues: only a level-3 heading ends a section; Python whitespace is stripped", () => {
    assertEquals(sectionValues("### Priority\n\nlow\n#### Detail\nhigh\n### Catalog\n\nsdd", "Priority"), [
        "low",
        "#### detail",
        "high",
    ]);
    assertEquals(sectionValues("### Priority\n\n\x1chigh\x85\n", "Priority"), ["high"]);
});

// --- plan ----------------------------------------------------------------------------------------------------

Deno.test("plan: adds the answered labels and clears needs-triage once a type and one priority exist", () => {
    const result = plan(issue(form("High", "core"), ["bug", NEEDS_TRIAGE]), KNOWN);
    assertEquals(result.plan, { number: 7, add: ["catalog/core", "priority/high"], remove: [NEEDS_TRIAGE] });
    assertEquals(result.findings, []);
});

Deno.test("plan: _No response_ removes the managed labels of that axis and keeps needs-triage", () => {
    const result = plan(
        issue(form("_No response_", "core"), ["task", "priority/low", "catalog/core", NEEDS_TRIAGE]),
        KNOWN,
    );
    assertEquals(result.plan, { number: 7, add: [], remove: ["priority/low"] });
});

Deno.test("plan: an axis the form did not ask is left alone", () => {
    const result = plan(
        issue("### Catalog\n\nmeta\n", ["bug", "priority/medium", "catalog/core", NEEDS_TRIAGE]),
        KNOWN,
    );
    assertEquals(result.plan, { number: 7, add: ["catalog/meta"], remove: ["catalog/core", NEEDS_TRIAGE] });
});

Deno.test("plan: needs-triage stays with no type label or with two priorities", () => {
    assertEquals(plan(issue(form("low", "core"), [NEEDS_TRIAGE]), KNOWN).plan.remove, []);
    assertEquals(plan(issue("", ["bug", "priority/low", "priority/high", NEEDS_TRIAGE]), KNOWN).plan.remove, []);
});

Deno.test("plan: an answer with no label is a finding; the rest of the plan stands", () => {
    const result = plan(issue(form("urgent", "meta, nowhere"), ["enhancement"]), KNOWN);
    assertEquals(result.plan.add, ["catalog/meta"]);
    assertEquals(result.findings.length, 2);
    assertStringIncludes(result.findings[0], "'urgent' under '### Priority' has no label 'priority/urgent'");
    assertStringIncludes(result.findings[1], "'nowhere' under '### Catalog'");
});

Deno.test("plan: an issue already in sync plans nothing", () => {
    assertEquals(plan(issue(form("low", "core"), ["bug", "priority/low", "catalog/core"]), KNOWN).plan, {
        number: 7,
        add: [],
        remove: [],
    });
});

Deno.test("plan: output is in code-point order and pythonJson escapes like ensure_ascii", () => {
    const result = plan(issue(form("low", "core"), ["catalog/\u{1F600}", "catalog/\uFF41", "catalog/z"]), KNOWN);
    assertEquals(result.plan.remove, ["catalog/z", "catalog/\uFF41", "catalog/\u{1F600}"]);
    assertEquals(pythonJson(["\u65e5", "\u{1F600}"]), '[\n  "\\u65e5",\n  "\\ud83d\\ude00"\n]');
});

// --- inputs --------------------------------------------------------------------------------------------------

Deno.test("loadLabelNames: reads names and fails on an unusable file", () => {
    const dir = tempDir();
    try {
        assertEquals(loadLabelNames(writeJson(dir, "ok.json", [{ name: " bug " }])), new Set(["bug"]));
        assertThrows(() => loadLabelNames(`${dir}/missing.json`), InputError, "cannot read");
        assertThrows(() => loadLabelNames(writeJson(dir, "obj.json", { name: "bug" })), InputError, "JSON array");
        assertThrows(() => loadLabelNames(writeJson(dir, "bad.json", [{ color: "fff" }])), InputError, "entry 0");
        // Stricter than the Python, which stringified or skipped these entries (intentional difference).
        assertThrows(() => loadLabelNames(writeJson(dir, "int.json", [{ name: 3 }])), InputError, "entry 0");
        assertThrows(() => loadLabelNames(writeJson(dir, "str.json", [{ name: "a" }, "bug"])), InputError, "entry 1");
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});

Deno.test("parseIssue: rejects what the Python ignored or passed through (intentional difference)", () => {
    const issue = (fields: Record<string, unknown>) => ({ number: 5, body: "", labels: [], ...fields });
    assertEquals(parseIssue(issue({ labels: [{ name: "bug" }] }), "src").labels, ["bug"]);
    for (const number of [0, -1, 5.5, "5", null, undefined]) {
        assertThrows(() => parseIssue(issue({ number }), "src"), InputError, "issue.number");
    }
    assertThrows(() => parseIssue(issue({ labels: ["bug"] }), "src"), InputError, "has no string");
    assertThrows(() => parseIssue(issue({ labels: [{ name: 7 }] }), "src"), InputError, "has no string");
    assertThrows(() => parseIssue(issue({ labels: [{ id: 1 }] }), "src"), InputError, "has no string");
    assertThrows(() => parseIssue(issue({ labels: {} }), "src"), InputError, "not an array");
    assertThrows(() => parseIssue(issue({ body: 3 }), "src"), InputError, "issue.body");
    assertThrows(() => parseIssue([], "src"), InputError, "no issue object");
});

Deno.test("loadIssueFromEvent: reads issue and repository; rejects other payloads", () => {
    const dir = tempDir();
    try {
        const ok = writeJson(dir, "ok.json", {
            issue: { number: 3, body: null, labels: [{ name: "bug" }] },
            repository: { full_name: "o/r" },
        });
        assertEquals(loadIssueFromEvent(ok), { issue: { number: 3, body: "", labels: ["bug"] }, repo: "o/r" });
        const noLabels = writeJson(dir, "nl.json", {
            issue: { number: 3, body: "b" },
            repository: { full_name: "o/r" },
        });
        assertEquals(loadIssueFromEvent(noLabels).issue.labels, []);
        assertThrows(
            () => loadIssueFromEvent(writeJson(dir, "pr.json", { pull_request: {} })),
            InputError,
            "not an issues",
        );
        assertThrows(
            () => loadIssueFromEvent(writeJson(dir, "norepo.json", { issue: { number: 1 } })),
            InputError,
            "repository.full_name",
        );
        assertThrows(
            () => loadIssueFromEvent(writeJson(dir, "nonum.json", { issue: {}, repository: { full_name: "o/r" } })),
            InputError,
            "issue.number",
        );
        assertThrows(() => loadIssueFromEvent(`${dir}/missing.json`), InputError, "cannot read event payload");
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});

Deno.test("loadIssueViaGh: calls gh api; a failed call or non-JSON output is a CommandError", async () => {
    const gh = stub([{ stdout: JSON.stringify({ number: 5, body: "x", labels: [] }) }]);
    assertEquals(await loadIssueViaGh(5, "o/r", gh), { number: 5, body: "x", labels: [] });
    assertEquals(gh.calls[0].args, ["api", "repos/o/r/issues/5"]);
    await assertRejects(() => loadIssueViaGh(5, "o/r", stub([{ code: 1, stderr: "HTTP 404" }])), CommandError, "404");
    await assertRejects(() => loadIssueViaGh(5, "o/r", stub([{ stdout: "<html>" }])), CommandError, "printed no JSON");
});

// --- apply ---------------------------------------------------------------------------------------------------

Deno.test("apply: one POST with the additions, one DELETE per removal", async () => {
    const gh = stub();
    await apply("o/r", 7, ["priority/low"], ["catalog/core", NEEDS_TRIAGE], gh);
    assertEquals(gh.calls, [
        {
            args: ["api", "repos/o/r/issues/7/labels", "--method", "POST", "--input", "-"],
            stdin: '{"labels":["priority/low"]}',
        },
        { args: ["api", "repos/o/r/issues/7/labels/catalog/core", "--method", "DELETE"], stdin: undefined },
        { args: ["api", `repos/o/r/issues/7/labels/${NEEDS_TRIAGE}`, "--method", "DELETE"], stdin: undefined },
    ]);
});

Deno.test("apply: a 404 on DELETE is already done; any other failure is a CommandError", async () => {
    await apply("o/r", 7, [], ["catalog/core"], stub([{ code: 1, stderr: "gh: Not Found (HTTP 404)" }]));
    await assertRejects(
        () => apply("o/r", 7, [], ["catalog/core"], stub([{ code: 1, stderr: "HTTP 403" }])),
        CommandError,
        "403",
    );
    await assertRejects(
        () => apply("o/r", 7, ["priority/low"], [], stub([{ code: 1, stderr: "HTTP 404" }])),
        CommandError,
        "POST",
    );
});

// --- main ----------------------------------------------------------------------------------------------------

Deno.test("main: dry run prints the plan, writes nothing, exits 0", async () => {
    const dir = tempDir();
    try {
        const event = writeJson(dir, "e.json", {
            issue: { number: 9, body: form("low", "core"), labels: [{ name: "bug" }, { name: NEEDS_TRIAGE }] },
            repository: { full_name: "o/r" },
        });
        const gh = stub();
        const r = await capture(() => main(["--event", event], gh));
        assertEquals(r.code, 0);
        assertEquals(gh.calls, []);
        assertEquals(JSON.parse(r.stdout), {
            number: 9,
            add: ["catalog/core", "priority/low"],
            remove: [NEEDS_TRIAGE],
            repo: "o/r",
            applied: false,
        });
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});

Deno.test("main: --apply executes a non-empty plan and reports applied; an empty plan calls nothing", async () => {
    const dir = tempDir();
    try {
        const event = writeJson(dir, "e.json", {
            issue: { number: 9, body: form("low", "core"), labels: [{ name: "bug" }] },
            repository: { full_name: "o/r" },
        });
        const gh = stub();
        const r = await capture(() => main(["--event", event, "--apply"], gh));
        assertEquals(r.code, 0);
        assertEquals(gh.calls.length, 1);
        assertEquals(JSON.parse(r.stdout).applied, true);

        const synced = writeJson(dir, "s.json", {
            issue: { number: 9, body: "", labels: [] },
            repository: { full_name: "o/r" },
        });
        const idle = stub();
        const s = await capture(() => main(["--event", synced, "--apply"], idle));
        assertEquals(idle.calls, []);
        assertEquals(JSON.parse(s.stdout).applied, false);
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});

Deno.test("main: a finding exits 1 after printing the plan; --apply still applies the rest", async () => {
    const dir = tempDir();
    try {
        const event = writeJson(dir, "e.json", {
            issue: { number: 9, body: form("urgent", "core"), labels: [] },
            repository: { full_name: "o/r" },
        });
        const gh = stub();
        const r = await capture(() => main(["--event", event, "--apply"], gh));
        assertEquals(r.code, 1);
        assertEquals(JSON.parse(r.stdout).add, ["catalog/core"]);
        assertStringIncludes(r.stderr, "'urgent'");
        assertEquals(gh.calls.length, 1);
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});

Deno.test("main: --issue fetches through gh; an API failure exits 2 with nothing on stdout", async () => {
    const ok = stub([{ stdout: JSON.stringify({ number: 4, body: form("low", "core"), labels: [] }) }]);
    const r = await capture(() => main(["--issue", "4", "--repo", "o/r"], ok));
    assertEquals(r.code, 0);
    assertEquals(JSON.parse(r.stdout).repo, "o/r");

    const bad = stub([{ code: 1, stderr: "HTTP 401" }]);
    const f = await capture(() => main(["--issue", "4", "--repo", "o/r"], bad));
    assertEquals(f.code, 2);
    assertEquals(f.stdout, "");
    assertStringIncludes(f.stderr, "HTTP 401");
});

Deno.test("main: unusable input files exit 2", async () => {
    const r = await capture(() => main(["--event", "/nonexistent/event.json"], stub()));
    assertEquals(r.code, 2);
    assertStringIncludes(r.stderr, "cannot read event payload");
    const dir = tempDir();
    try {
        const event = writeJson(dir, "e.json", { issue: { number: 1, labels: [] }, repository: { full_name: "o/r" } });
        const l = await capture(() => main(["--event", event, "--labels", `${dir}/none.json`], stub()));
        assertEquals(l.code, 2);
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});

Deno.test("main: usage errors exit 2; --help exits 0", async () => {
    for (
        const argv of [
            [],
            ["--issue", "4"],
            ["--issue", "x", "--repo", "o/r"],
            ["--event", "e.json", "--issue", "4", "--repo", "o/r"],
            ["--event", "e.json", "stray"],
            ["--bogus"],
            ["--apply=yes", "--event", "e.json"],
        ]
    ) {
        const r = await capture(() => main(argv, stub()));
        assertEquals(r.code, 2, argv.join(" "));
        assert(r.stderr.includes("usage:"), argv.join(" "));
    }
    const h = await capture(() => main(["-h"], stub()));
    assertEquals(h.code, 0);
    assertStringIncludes(h.stdout, "usage: sync_issue_metadata.ts");
});
