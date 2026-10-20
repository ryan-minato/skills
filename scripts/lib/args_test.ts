import { assertEquals, assertThrows } from "@std/assert";
import { parseArgs } from "./args.ts";
import { UsageError } from "./repo.ts";

Deno.test("parseArgs reads values, repeated values, flags, and positionals", () => {
    const parsed = parseArgs(["show", "--doc=tasks", "--change", "a", "--change", "b", "--json", "x"], {
        string: ["doc"],
        multiple: ["change"],
        boolean: ["json", "dry-run"],
    });
    assertEquals(parsed.values, { doc: "tasks" });
    assertEquals(parsed.lists, { change: ["a", "b"] });
    assertEquals(parsed.flags, { json: true, "dry-run": false });
    assertEquals(parsed.positional, ["show", "x"]);
});

Deno.test("parseArgs refuses unknown options, a missing value, and a value on a flag", () => {
    assertThrows(() => parseArgs(["--nope"], {}), UsageError, "unknown option --nope");
    assertThrows(() => parseArgs(["--doc"], { string: ["doc"] }), UsageError, "--doc needs a value");
    assertThrows(() => parseArgs(["--json=1"], { boolean: ["json"] }), UsageError, "--json takes no value");
    assertThrows(() => parseArgs(["-x"], {}), UsageError, "unknown option -x");
});

Deno.test("parseArgs maps short aliases and stops at --", () => {
    const parsed = parseArgs(["-h", "--", "--doc"], { boolean: ["help"], string: ["doc"], alias: { h: "help" } });
    assertEquals(parsed.flags.help, true);
    assertEquals(parsed.positional, ["--doc"]);
});
