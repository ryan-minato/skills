import { assert, assertEquals, assertThrows } from "@std/assert";
import { CommandError, exists, join, rel, ROOT, run, runAsync, runChecked } from "./repo.ts";

Deno.test("ROOT is the repository root", () => {
    assert(exists(join(ROOT, "AGENTS.md")));
    assertEquals(rel(join(ROOT, "scripts", "lib")), "scripts/lib");
});

Deno.test("run returns the status; runChecked names a failed command", () => {
    assertEquals(run("git", ["--version"]).code, 0);
    assertThrows(() => runChecked("git", ["no-such-subcommand"]), CommandError, "`git no-such-subcommand` exited");
});

Deno.test("runAsync passes stdin", async () => {
    const out = await runAsync("git", ["hash-object", "--stdin"], { stdin: "x\n" });
    assertEquals(out.code, 0);
    assertEquals(out.stdout.trim().length, 40);
});
