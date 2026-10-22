import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import type { Output } from "./lib/repo.ts";
import { CommandError } from "./lib/repo.ts";
import {
    buildPlan,
    fetchCurrent,
    type Label,
    LABEL_PAGE_LIMIT,
    LabelsFileError,
    main,
    parseDesired,
    pythonJson,
} from "./sync_labels.ts";

const CURRENT = [
    { name: "bug", color: "D73A4A", description: "Something isn't working " },
    { name: "docs", color: "0075ca", description: "Old text" },
    { name: "zeta", color: "000000", description: "" },
    { name: "Alpha", color: "111111", description: "" },
];
const DESIRED = [
    { name: "bug", color: "d73a4a", description: "Something isn't working", applied_by: "form" },
    { name: " New ", color: "#ABCDEF", description: "  Neu — ✓  " },
    { name: "Docs", color: "0075CA", description: "Improvements" },
];

/** A stub gh: answers `label list` with `list`, records every call, fails the named subcommand. */
function stubGh(list: string | Output, failOn?: string): { gh: (args: string[]) => Output; calls: string[][] } {
    const calls: string[][] = [];
    const gh = (args: string[]): Output => {
        calls.push(args);
        if (args[1] === "list") return typeof list === "string" ? { code: 0, stdout: list, stderr: "" } : list;
        if (args[1] === failOn) return { code: 1, stdout: "", stderr: `label ${failOn} refused` };
        return { code: 0, stdout: "", stderr: "" };
    };
    return { gh, calls };
}

/** Runs main with a labels file in a temp directory; captures stdout and stderr. */
function runMain(
    desired: unknown,
    argv: string[],
    gh: (args: string[]) => Output,
): { code: number; stdout: string; stderr: string } {
    const dir = Deno.makeTempDirSync();
    const file = `${dir}/labels.json`;
    Deno.writeTextFileSync(file, typeof desired === "string" ? desired : JSON.stringify(desired));
    const out: string[] = [];
    const err: string[] = [];
    const log = console.log;
    const error = console.error;
    console.log = (m: string) => out.push(m);
    console.error = (m: string) => err.push(m);
    try {
        const code = main(argv.map((a) => (a === "FILE" ? file : a)), gh);
        return { code, stdout: out.join("\n"), stderr: err.join("\n") };
    } finally {
        console.log = log;
        console.error = error;
        Deno.removeSync(dir, { recursive: true });
    }
}

Deno.test("parseDesired normalizes names, colors, and descriptions and drops extra keys", () => {
    assertEquals(parseDesired(JSON.stringify(DESIRED), "f"), [
        { name: "bug", color: "d73a4a", description: "Something isn't working" },
        { name: "New", color: "abcdef", description: "Neu — ✓" },
        { name: "Docs", color: "0075ca", description: "Improvements" },
    ]);
    assertEquals(parseDesired('[{"name": "x", "color": "000000"}]', "f")[0].description, "");
});

Deno.test("parseDesired refuses each invalid file shape", () => {
    const cases: [string, string][] = [
        ["not json", "cannot read f as JSON"],
        ["{}", "must be a non-empty JSON array"],
        ["[]", "must be a non-empty JSON array"],
        ['[{"color": "000000"}]', "f[0] must be an object with a non-empty string 'name'"],
        ['[{"name": "  ", "color": "000000"}]', "non-empty string 'name'"],
        ['["x"]', "non-empty string 'name'"],
        ['[{"name": "x", "color": "00000g"}]', "is not a 6-digit hex string"],
        ['[{"name": "x"}]', "color null is not a 6-digit hex string"],
        ['[{"name": "x", "color": "000000", "description": 5}]', "description 5 is not a string"],
        ['[{"name": "x", "color": "000000", "description": null}]', "description null is not a string"],
        ['[{"name": "x", "color": "000000"}, {"name": "X", "color": "111111"}]', 'f[1]: duplicate label name "X"'],
    ];
    for (const [text, message] of cases) assertThrows(() => parseDesired(text, "f"), LabelsFileError, message);
});

Deno.test("fetchCurrent keys by lower-cased name and checks gh output", () => {
    const current = fetchCurrent(stubGh(JSON.stringify(CURRENT)).gh, "o/r");
    assertEquals(current.get("bug"), { name: "bug", color: "d73a4a", description: "Something isn't working" });
    assertEquals(current.get("alpha")?.name, "Alpha");
    const { gh, calls } = stubGh("[]");
    fetchCurrent(gh, "o/r");
    assertEquals(calls, [["label", "list", "-R", "o/r", "--json", "name,color,description", "--limit", "1000"]]);
});

Deno.test("fetchCurrent fails on each broken gh interface", () => {
    const many = JSON.stringify(Array.from({ length: LABEL_PAGE_LIMIT }, (_, i) => ({ ...CURRENT[2], name: `l${i}` })));
    const cases: [string | Output, string][] = [
        [{ code: 1, stdout: "", stderr: "HTTP 404" }, "exited 1: HTTP 404"],
        ["not json", "did not print JSON"],
        ['{"labels": []}', "not a JSON list of labels"],
        ['[{"name": "x"}]', "entry 0 is"],
        [many, "at or above the 1000 listing limit"],
    ];
    for (const [list, message] of cases) {
        assertThrows(() => fetchCurrent(stubGh(list).gh, "o/r"), CommandError, message);
    }
});

Deno.test("buildPlan classifies create, update, skip, and sorted prune candidates", () => {
    const desired = parseDesired(JSON.stringify(DESIRED), "f");
    const current = fetchCurrent(stubGh(JSON.stringify(CURRENT)).gh, "o/r");
    const plan = buildPlan(desired, current);
    assertEquals(plan.create.map((l: Label) => l.name), ["New"]);
    assertEquals(plan.update.map((l: Label) => l.name), ["Docs"]);
    assertEquals(plan.skip, ["bug"]);
    assertEquals(plan.pruneCandidates, ["Alpha", "zeta"]);
});

Deno.test("pythonJson escapes non-ASCII like json.dump(indent=2)", () => {
    assertEquals(pythonJson({ a: "—😀\x7f", b: [] }), '{\n  "a": "\\u2014\\ud83d\\ude00\\u007f",\n  "b": []\n}');
});

Deno.test("main dry run prints the plan and writes nothing", () => {
    const { gh, calls } = stubGh(JSON.stringify(CURRENT));
    const r = runMain(DESIRED, ["--file", "FILE", "--repo", "o/r"], gh);
    assertEquals(r.code, 0);
    const plan = JSON.parse(r.stdout);
    assertEquals(plan.applied, false);
    assertEquals(plan.prune_candidates, ["Alpha", "zeta"]);
    assertEquals(plan.pruned, []);
    assertEquals(calls.length, 1);
});

Deno.test("main --apply creates and edits but does not prune", () => {
    const { gh, calls } = stubGh(JSON.stringify(CURRENT));
    const r = runMain(DESIRED, ["--file", "FILE", "--repo", "o/r", "--apply"], gh);
    assertEquals(r.code, 0);
    assertEquals(calls.slice(1), [
        ["label", "create", "New", "-R", "o/r", "--color", "abcdef", "--description", "Neu — ✓"],
        ["label", "edit", "Docs", "-R", "o/r", "--color", "0075ca", "--description", "Improvements"],
    ]);
    assertEquals(r.stderr, "create: New\nupdate: Docs");
    assertEquals(JSON.parse(r.stdout).applied, true);
});

Deno.test("main --apply --prune deletes the prune candidates", () => {
    const { gh, calls } = stubGh(JSON.stringify(CURRENT));
    const r = runMain(DESIRED, ["--file", "FILE", "--repo", "o/r", "--apply", "--prune"], gh);
    assertEquals(r.code, 0);
    assertEquals(calls.slice(3), [
        ["label", "delete", "Alpha", "-R", "o/r", "--yes"],
        ["label", "delete", "zeta", "-R", "o/r", "--yes"],
    ]);
    assertEquals(JSON.parse(r.stdout).pruned, ["Alpha", "zeta"]);
});

Deno.test("main stops at the first failed write, prints no plan, and exits 2", () => {
    for (const failOn of ["create", "edit", "delete"]) {
        const { gh } = stubGh(JSON.stringify(CURRENT), failOn);
        const r = runMain(DESIRED, ["--file", "FILE", "--repo", "o/r", "--apply", "--prune"], gh);
        assertEquals(r.code, 2);
        assertEquals(r.stdout, "");
        assertStringIncludes(r.stderr, `label ${failOn} refused`);
    }
});

Deno.test("main exits 1 on an invalid labels file before calling gh", () => {
    const { gh, calls } = stubGh("[]");
    const r = runMain("[]", ["--file", "FILE", "--repo", "o/r"], gh);
    assertEquals(r.code, 1);
    assertStringIncludes(r.stderr, "non-empty JSON array");
    assertEquals(calls, []);
    const missing = runMain(DESIRED, ["--file", "/nonexistent/labels.json", "--repo", "o/r"], gh);
    assertEquals(missing.code, 1);
    assertStringIncludes(missing.stderr, "cannot read /nonexistent/labels.json");
});

Deno.test("main exits 2 on usage errors and 0 on --help", () => {
    const { gh, calls } = stubGh("[]");
    for (
        const argv of [
            ["--repo", "o/r"],
            ["--file", "FILE"],
            ["--file", "FILE", "--repo", "bad"],
            ["--file", "FILE", "--repo", "o/r", "--prune"],
            ["--file", "FILE", "--repo", "o/r", "--bogus"],
            ["--file", "FILE", "--repo", "o/r", "extra"],
        ]
    ) {
        assertEquals(runMain(DESIRED, argv, gh).code, 2, argv.join(" "));
    }
    assertEquals(calls, []);
    const help = runMain(DESIRED, ["--help"], gh);
    assertEquals(help.code, 0);
    assertStringIncludes(help.stdout, "usage: sync_labels.ts");
});
