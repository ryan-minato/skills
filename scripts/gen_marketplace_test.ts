import { assertEquals, assertThrows } from "@std/assert";
import {
    catalogSkillPaths,
    coverageWarnings,
    main,
    ManifestError,
    MARKETPLACE,
    parseManifest,
    regenerate,
    render,
} from "./gen_marketplace.ts";
import { join } from "./lib/repo.ts";

function skill(root: string, catalog: string, name: string): void {
    Deno.mkdirSync(join(root, "skills", catalog, name), { recursive: true });
    Deno.writeTextFileSync(join(root, "skills", catalog, name, "SKILL.md"), "---\nname: x\n---\n");
}

/** A fixture repository: core (b, a, and a dir without SKILL.md), extra (one skill, unlisted), empty catalog. */
function fixture(plugins: unknown[]): string {
    const root = Deno.makeTempDirSync();
    skill(root, "core", "b-skill");
    skill(root, "core", "a-skill");
    Deno.mkdirSync(join(root, "skills", "core", "not-a-skill"), { recursive: true });
    Deno.writeTextFileSync(join(root, "skills", "core", "README.md"), "x");
    skill(root, "extra", "only");
    Deno.mkdirSync(join(root, "skills", "empty"), { recursive: true });
    Deno.mkdirSync(join(root, ".claude-plugin"));
    Deno.writeTextFileSync(
        join(root, MARKETPLACE),
        JSON.stringify({ name: "m", plugins, "description": "Ünïcode — kept" }, null, 2) + "\n",
    );
    return root;
}

function captureStderr(fn: () => number): { code: number; stderr: string[]; stdout: string[] } {
    const stderr: string[] = [];
    const stdout: string[] = [];
    const [err, log] = [console.error, console.log];
    console.error = (m: string) => stderr.push(m);
    console.log = (m: string) => stdout.push(m);
    try {
        return { code: fn(), stderr, stdout };
    } finally {
        [console.error, console.log] = [err, log];
    }
}

Deno.test("catalogSkillPaths lists only directories with SKILL.md, sorted; a missing catalog is empty", () => {
    const root = fixture([]);
    try {
        const skills = join(root, "skills");
        assertEquals(catalogSkillPaths(skills, "core"), ["./skills/core/a-skill", "./skills/core/b-skill"]);
        assertEquals(catalogSkillPaths(skills, "empty"), []);
        assertEquals(catalogSkillPaths(skills, "missing"), []);
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
});

Deno.test("a symlinked skill directory counts, as Python's is_dir followed links", () => {
    const root = fixture([]);
    try {
        Deno.symlinkSync(join(root, "skills", "extra", "only"), join(root, "skills", "core", "linked"));
        assertEquals(catalogSkillPaths(join(root, "skills"), "core"), [
            "./skills/core/a-skill",
            "./skills/core/b-skill",
            "./skills/core/linked",
        ]);
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
});

Deno.test("coverageWarnings names a non-empty catalog with no entry, not an empty one", () => {
    const root = fixture([]);
    try {
        assertEquals(coverageWarnings(join(root, "skills"), new Set(["core"])), [
            "catalog `extra` has skills but no plugin entry; add one with a name and description, then re-run.",
        ]);
        assertEquals(coverageWarnings(join(root, "nowhere"), new Set()), []);
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
});

Deno.test("regenerate rewrites skills, keeps other fields and order, warns for an empty catalog", () => {
    const manifest = parseManifest(JSON.stringify({
        plugins: [
            { name: "empty", source: "./", skills: ["./skills/empty/gone"], description: "d" },
            { name: "core", source: "./", strict: false, description: "c" },
        ],
    }));
    const root = fixture([]);
    try {
        const warnings = regenerate(manifest, join(root, "skills"));
        assertEquals(warnings, [
            "plugin `empty`: catalog skills/empty/ is empty or missing; remove the entry or add a skill.",
            "catalog `extra` has skills but no plugin entry; add one with a name and description, then re-run.",
        ]);
        assertEquals(manifest.plugins.map((p) => p.name), ["empty", "core"]);
        assertEquals(manifest.plugins[0], { name: "empty", source: "./", skills: [], description: "d" });
        // A new `skills` key is appended after the existing ones, as Python's dict assignment does.
        assertEquals(Object.keys(manifest.plugins[1]), ["name", "source", "strict", "description", "skills"]);
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
});

Deno.test("render matches json.dumps(indent=2, ensure_ascii=False) for this file's shapes", () => {
    const manifest = parseManifest('{"a":"é—x","plugins":[{"name":"n","skills":[],"o":{}}]}');
    assertEquals(
        render(manifest),
        '{\n  "a": "é—x",\n  "plugins": [\n    {\n      "name": "n",\n      "skills": [],\n      "o": {}\n    }\n  ]\n}\n',
    );
});

Deno.test("parseManifest rejects what the generator cannot rewrite", () => {
    assertThrows(() => parseManifest("{"), ManifestError, "is not valid JSON");
    assertThrows(() => parseManifest("[]"), ManifestError, "the top level is not an object");
    assertThrows(() => parseManifest("{}"), ManifestError, "`plugins` is undefined, not an array");
    assertThrows(() => parseManifest('{"plugins":[1]}'), ManifestError, "plugins[0] is not an object");
    assertThrows(() => parseManifest('{"plugins":[{}]}'), ManifestError, "plugins[0].name is undefined");
    assertThrows(() => parseManifest('{"plugins":[{"name":""}]}'), ManifestError, 'plugins[0].name is ""');
});

Deno.test("main writes the file, prints the path, and sends warnings to stderr", () => {
    const root = fixture([{ name: "core", source: "./", skills: ["./skills/core/old"] }, { name: "empty" }]);
    try {
        const out = captureStderr(() => main([], root));
        assertEquals(out.code, 0);
        assertEquals(out.stdout, [`Wrote ${MARKETPLACE}.`]);
        assertEquals(out.stderr.length, 2);
        const written = JSON.parse(Deno.readTextFileSync(join(root, MARKETPLACE)));
        assertEquals(written.plugins[0].skills, ["./skills/core/a-skill", "./skills/core/b-skill"]);
        assertEquals(written.plugins[1].skills, []);
        assertEquals(written.description, "Ünïcode — kept");
        // Idempotent: a second run leaves the bytes unchanged.
        const first = Deno.readTextFileSync(join(root, MARKETPLACE));
        captureStderr(() => main([], root));
        assertEquals(Deno.readTextFileSync(join(root, MARKETPLACE)), first);
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
});

Deno.test("main exits 2 on a malformed manifest without writing", () => {
    const root = fixture([]);
    try {
        Deno.writeTextFileSync(join(root, MARKETPLACE), '{"plugins": {}}');
        const out = captureStderr(() => main([], root));
        assertEquals(out.code, 2);
        assertEquals(out.stderr, [`ERROR: ${MARKETPLACE}: \`plugins\` is {}, not an array; fix it by hand.`]);
        assertEquals(Deno.readTextFileSync(join(root, MARKETPLACE)), '{"plugins": {}}');
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
});

Deno.test("main: --help exits 0; --check and positionals are usage errors", () => {
    assertEquals(captureStderr(() => main(["--help"], "/nonexistent")).code, 0);
    assertEquals(captureStderr(() => main(["-h"], "/nonexistent")).stdout[0].startsWith("usage:"), true);
    const check = captureStderr(() => main(["--check"], "/nonexistent"));
    assertEquals(check.code, 2);
    assertEquals(check.stderr.at(-1), "gen_marketplace.ts: error: unknown option --check");
    assertEquals(captureStderr(() => main(["extra"], "/nonexistent")).code, 2);
});
