import { assertEquals, assertThrows } from "@std/assert";
import { decide, main, NeedsError, parseNeeds } from "./ci_gate.ts";
import { UsageError } from "./lib/repo.ts";

// The shape `${{ toJSON(needs) }}` renders: pretty-printed, with an `outputs` object per job.
function needs(results: Record<string, string>): string {
    const context: Record<string, unknown> = {};
    for (const [job, result] of Object.entries(results)) context[job] = { result, outputs: {} };
    return JSON.stringify(context, null, 2);
}

// Runs main() with stdout and stderr captured.
function capture(argv: string[], env: string | undefined): { code: number; out: string; err: string } {
    const out: string[] = [];
    const err: string[] = [];
    const log = console.log;
    const error = console.error;
    console.log = (...args: unknown[]) => out.push(args.join(" "));
    console.error = (...args: unknown[]) => err.push(args.join(" "));
    try {
        return { code: main(argv, env), out: out.join("\n"), err: err.join("\n") };
    } finally {
        console.log = log;
        console.error = error;
    }
}

Deno.test("every dependency succeeded: exit 0, results on stdout", () => {
    const r = capture([], needs({ quality: "success", spec: "success" }));
    assertEquals(r.code, 0);
    assertEquals(r.out, "quality: success\nspec: success");
    assertEquals(r.err, "");
});

Deno.test("a failed dependency fails the gate and is named", () => {
    const r = capture([], needs({ quality: "failure", spec: "success" }));
    assertEquals(r.code, 1);
    assertEquals(r.err.split("\n")[0], "ERROR: quality: failure");
});

Deno.test("a cancelled dependency fails the gate and is named", () => {
    const r = capture([], needs({ quality: "success", spec: "cancelled" }));
    assertEquals(r.code, 1);
    assertEquals(r.err.split("\n")[0], "ERROR: spec: cancelled");
});

Deno.test("every bad dependency is named, in NEEDS order", () => {
    const results = parseNeeds(needs({ a: "cancelled", b: "success", c: "failure", d: "skipped" }));
    assertEquals(decide(results, []), [
        "a: cancelled",
        "c: failure",
        "d: skipped, and it is not listed with --allow-skip",
    ]);
});

Deno.test("a skipped dependency fails unless listed with --allow-skip", () => {
    const env = needs({ quality: "success", devcontainer: "skipped" });
    const refused = capture([], env);
    assertEquals(refused.code, 1);
    assertEquals(refused.err.split("\n")[0], "ERROR: devcontainer: skipped, and it is not listed with --allow-skip");
    const allowed = capture(["--allow-skip", "devcontainer"], env);
    assertEquals(allowed.code, 0);
    assertEquals(allowed.out, "quality: success\ndevcontainer: skipped");
});

Deno.test("--allow-skip does not excuse a failure, and repeats", () => {
    const env = needs({ a: "skipped", b: "skipped", c: "failure" });
    assertEquals(capture(["--allow-skip", "a", "--allow-skip=b"], env).code, 1);
    assertEquals(decide(parseNeeds(env), ["a", "b", "c"]), ["c: failure"]);
    assertEquals(capture(["--allow-skip", "a", "--allow-skip", "b"], needs({ a: "skipped", b: "skipped" })).code, 0);
});

Deno.test("--allow-skip naming a job outside NEEDS is a usage error", () => {
    assertThrows(() => decide(parseNeeds(needs({ a: "success" })), ["b"]), UsageError, "does not need: b");
    assertEquals(capture(["--allow-skip", "b"], needs({ a: "success" })).code, 2);
});

Deno.test("missing or malformed NEEDS exits 2 naming the problem", () => {
    const cases: [string | undefined, string][] = [
        [undefined, "NEEDS is not set"],
        ["  ", "NEEDS is not set"],
        ["{not json", "NEEDS is not JSON"],
        ["[]", "must be a JSON object"],
        ["null", "must be a JSON object"],
        ['"success"', "must be a JSON object"],
        ["{}", "NEEDS is empty"],
        ['{"a": "success"}', "NEEDS.a.result must be one of"],
        ['{"a": {"outputs": {}}}', "NEEDS.a.result must be one of"],
        ['{"a": {"result": "neutral"}}', "NEEDS.a.result must be one of"],
        ['{"a": {"result": null}}', "NEEDS.a.result must be one of"],
    ];
    for (const [env, message] of cases) {
        assertThrows(() => parseNeeds(env), NeedsError, message);
        const r = capture([], env);
        assertEquals(r.code, 2, `NEEDS=${env}`);
        assertEquals(r.out, "", `NEEDS=${env}`);
    }
});

Deno.test("arguments: --help exits 0, unknown options and positionals exit 2", () => {
    const help = capture(["--help"], undefined);
    assertEquals(help.code, 0);
    assertEquals(help.out.startsWith("usage:"), true);
    assertEquals(capture(["-h"], undefined).code, 0);
    assertEquals(capture(["--nope"], needs({ a: "success" })).code, 2);
    assertEquals(capture(["--allow-skip"], needs({ a: "success" })).code, 2);
    assertEquals(capture(["extra"], needs({ a: "success" })).code, 2);
});

Deno.test("the script runs with only --allow-env=NEEDS", async () => {
    const script = new URL("./ci_gate.ts", import.meta.url).pathname;
    const runWith = async (env: Record<string, string>, args: string[] = []) => {
        const out = await new Deno.Command(script, { args, env, clearEnv: false, stdout: "piped", stderr: "piped" })
            .output();
        return out.code;
    };
    assertEquals(await runWith({ NEEDS: needs({ a: "success" }) }), 0);
    assertEquals(await runWith({ NEEDS: needs({ a: "failure" }) }), 1);
    assertEquals(await runWith({ NEEDS: needs({ a: "skipped" }) }, ["--allow-skip", "a"]), 0);
    assertEquals(await runWith({ NEEDS: "nope" }), 2);
});
