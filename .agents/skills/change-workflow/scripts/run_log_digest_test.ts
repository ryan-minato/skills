import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import type { Output } from "../../../../scripts/lib/repo.ts";
import { UsageError } from "../../../../scripts/lib/repo.ts";
import {
    digest,
    GhError,
    type GhRunner,
    main,
    parseOptions,
    selectJobs,
    ShapeError,
    splitLines,
    toJson,
    validateRun,
} from "./run_log_digest.ts";

const OPTS = { repo: "o/r", runId: 42, tail: 50 };

/** A stub gh: `run` answers the run view, `logs` the --log-failed calls by job id (missing → exit 1). */
function stub(run: Output | string, logs: Record<number, string> = {}): GhRunner & { calls: string[][] } {
    const calls: string[][] = [];
    return {
        calls,
        gh(args: string[]): Promise<Output> {
            calls.push(args);
            if (args.includes("--log-failed")) {
                const id = Number(args[args.indexOf("--job") + 1]);
                if (id in logs) return Promise.resolve({ code: 0, stdout: logs[id], stderr: "" });
                return Promise.resolve({ code: 1, stdout: "", stderr: `no log for job ${id}\n` });
            }
            return Promise.resolve(typeof run === "string" ? { code: 0, stdout: run, stderr: "" } : run);
        },
    };
}

function job(id: number, conclusion: string | null, steps: [string, string][] = [], status = "completed") {
    return {
        databaseId: id,
        name: `job${id}`,
        status,
        conclusion,
        steps: steps.map(([name, c]) => ({ name, conclusion: c })),
    };
}

const quiet = () => {};

Deno.test("parseOptions accepts the documented arguments and the default tail", () => {
    assertEquals(parseOptions(["--repo", "o/r", "--run-id", "007"]), { repo: "o/r", runId: 7, tail: 50 });
    assertEquals(parseOptions(["--repo=o/r", "--run-id=5", "--tail", "3"]), { repo: "o/r", runId: 5, tail: 3 });
    assertEquals(parseOptions(["-h"]), null);
});

Deno.test("parseOptions rejects bad arguments", () => {
    const bad: [string[], string][] = [
        [["--run-id", "1"], "--repo"],
        [["--repo", "o/r"], "--run-id"],
        [["--repo", "o", "--run-id", "1"], "OWNER/REPO"],
        [["--repo", "o/", "--run-id", "1"], "OWNER/REPO"],
        [["--repo", "o/r", "--run-id", "x"], "invalid int"],
        [["--repo", "o/r", "--run-id", "0"], "positive"],
        [["--repo", "o/r", "--run-id", "1", "--tail", "-1"], "positive"],
        [["--repo", "o/r", "--run-id", "1", "extra"], "unrecognized"],
        [["--repo", "o/r", "--run-id", "1", "--bogus"], "unknown option"],
    ];
    for (const [argv, text] of bad) assertThrows(() => parseOptions(argv), UsageError, text);
});

Deno.test("validateRun requires a jobs array of objects with numeric ids", () => {
    assertThrows(() => validateRun({ databaseId: 1 }), ShapeError, "no jobs array");
    assertThrows(() => validateRun([]), ShapeError, "not an object");
    assertThrows(() => validateRun({ jobs: [1] }), ShapeError, "jobs[0]");
    assertThrows(() => validateRun({ jobs: [{ name: "a" }] }), ShapeError, "databaseId");
    assertThrows(() => validateRun({ jobs: [{ databaseId: 1, steps: "x" }] }), ShapeError, "steps");
    assertEquals(validateRun({ jobs: [{ databaseId: 1, steps: null }] }).jobs.length, 1);
});

Deno.test("selectJobs takes every unsuccessful conclusion; cancelled only inside a run not cancelled", () => {
    const run = validateRun({
        conclusion: "failure",
        jobs: ["failure", "timed_out", "cancelled", "startup_failure", "success", "skipped", null].map((c, i) =>
            job(i, c)
        ),
    });
    const selected = selectJobs(run);
    assertEquals(selected.jobs.map((j) => j.databaseId), [0, 1, 2, 3]);
    const superseded = validateRun({ conclusion: "cancelled", jobs: [job(1, "cancelled")] });
    assertEquals(selectJobs(superseded).jobs, []);
    assertEquals(selected.note, undefined);
});

Deno.test("selectJobs falls back to completed jobs when only the run is unsuccessful", () => {
    const run = validateRun({ conclusion: "failure", jobs: [job(1, "neutral"), job(2, null, [], "in_progress")] });
    const selected = selectJobs(run);
    assertEquals(selected.jobs.map((j) => j.databaseId), [1]);
    assertStringIncludes(selected.note ?? "", "falling back to all 1 completed job(s)");
    assertEquals(selectJobs(validateRun({ conclusion: "success", jobs: [job(1, "success")] })).jobs, []);
});

Deno.test("splitLines follows Python's str.splitlines", () => {
    assertEquals(splitLines("a\r\nb\rc\nd\x0be\x1cf g\n\nh\n"), ["a", "b", "c", "d", "e", "f", "g", "", "h"]);
    assertEquals(splitLines(""), []);
    assertEquals(splitLines("x"), ["x"]);
});

Deno.test("toJson matches Python's json.dump(indent=2) with ASCII escapes", () => {
    assertEquals(
        toJson({ a: ["✓ 😀"], b: [], c: null }),
        '{\n  "a": [\n    "\\u2713 \\ud83d\\ude00"\n  ],\n  "b": [],\n  "c": null\n}',
    );
});

Deno.test("digest reports failed jobs, failed steps, and the log tail", async () => {
    const run = {
        databaseId: 9,
        status: "completed",
        conclusion: "failure",
        jobs: [job(1, "failure", [["s1", "success"], ["s2", "failure"]]), job(2, "success")],
    };
    const gh = stub(JSON.stringify(run), { 1: "1\n2\n3\n4\n" });
    const result = await digest({ ...OPTS, tail: 2 }, gh, quiet);
    assertEquals(result, {
        run_id: 9,
        status: "completed",
        conclusion: "failure",
        failed_jobs: [{ name: "job1", job_id: 1, failed_steps: ["s2"], log_tail: ["3", "4"] }],
    });
    assertEquals(gh.calls, [
        ["run", "view", "42", "-R", "o/r", "--json", "databaseId,conclusion,status,jobs"],
        ["run", "view", "-R", "o/r", "--job", "1", "--log-failed"],
    ]);
});

Deno.test("digest keeps nulls for missing run fields and the requested id when databaseId is absent", async () => {
    const result = await digest(OPTS, stub(JSON.stringify({ jobs: [] })), quiet);
    assertEquals(result, { run_id: 42, status: null, conclusion: null, failed_jobs: [] });
});

Deno.test("digest fails naming the job when its log cannot be fetched", async () => {
    const gh = stub(JSON.stringify({ conclusion: "failure", jobs: [job(7, "failure")] }));
    await assertRejects(() => digest(OPTS, gh, quiet), GhError, "job 7 (job7)");
    await assertRejects(() => digest(OPTS, gh, quiet), GhError, "no log for job 7");
});

Deno.test("digest fails on a gh error, unparseable JSON, a missing gh, and a run without jobs", async () => {
    await assertRejects(
        () => digest(OPTS, stub({ code: 1, stdout: "", stderr: "HTTP 404" }), quiet),
        GhError,
        "HTTP 404",
    );
    await assertRejects(() => digest(OPTS, stub("not json"), quiet), GhError, "could not parse gh JSON");
    const missing: GhRunner = { gh: () => Promise.reject(new Deno.errors.NotFound("gh")) };
    await assertRejects(() => digest(OPTS, missing, quiet), GhError, "gh not found");
    await assertRejects(() => digest(OPTS, stub("{}"), quiet), ShapeError, "no jobs array");
});

Deno.test("main maps outcomes to exit codes", async () => {
    const ok = stub(JSON.stringify({ databaseId: 1, status: "completed", conclusion: "success", jobs: [] }));
    assertEquals(await main(["--repo", "o/r", "--run-id", "1"], ok), 0);
    assertEquals(await main(["--help"], ok), 0);
    assertEquals(await main(["--repo", "o"], ok), 2);
    assertEquals(await main(["--repo", "o/r", "--run-id", "1"], stub({ code: 1, stdout: "", stderr: "x" })), 1);
    assertEquals(
        await main(
            ["--repo", "o/r", "--run-id", "1"],
            stub(JSON.stringify({ conclusion: "failure", jobs: [job(3, "failure")] })),
        ),
        1,
    );
    assertEquals(await main(["--repo", "o/r", "--run-id", "1"], stub("{}")), 2);
    const boom: GhRunner = { gh: () => Promise.reject(new Error("unexpected")) };
    await assertRejects(() => main(["--repo", "o/r", "--run-id", "1"], boom), Error, "unexpected");
});
