import { assert, assertEquals, assertMatch, assertThrows } from "@std/assert";
import {
    checkSkillDir,
    codePoints,
    decodeUtf8,
    ERROR,
    type Finding,
    main,
    normalizePath,
    parseFrontmatter,
    pyStrip,
    relTo,
    repoSkillDirs,
    splitLines,
    WARNING,
} from "./check_skill.ts";

/** Runs `fn` with console output captured; returns its result and what it printed. */
function capture<T>(fn: () => T): { result: T; stdout: string; stderr: string } {
    const log = console.log;
    const error = console.error;
    let stdout = "";
    let stderr = "";
    console.log = (...args: unknown[]) => void (stdout += args.join(" ") + "\n");
    console.error = (...args: unknown[]) => void (stderr += args.join(" ") + "\n");
    try {
        return { result: fn(), stdout, stderr };
    } finally {
        console.log = log;
        console.error = error;
    }
}

/** A temporary directory removed after `fn`. */
function withTemp(fn: (dir: string) => void): void {
    const dir = Deno.realPathSync(Deno.makeTempDirSync());
    try {
        fn(dir);
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
}

function skill(root: string, name: string, frontmatter: string, body = "# Body\n"): string {
    const dir = `${root}/${name}`;
    Deno.mkdirSync(dir, { recursive: true });
    Deno.writeTextFileSync(`${dir}/SKILL.md`, `---\n${frontmatter}---\n${body}`);
    return dir;
}

function messages(findings: Finding[], severity: string): string[] {
    return findings.filter(([s]) => s === severity).map(([, m]) => m);
}

Deno.test("splitLines follows Python's str.splitlines", () => {
    assertEquals(splitLines(""), []);
    assertEquals(splitLines("a\n"), ["a"]);
    assertEquals(splitLines("a\n\n"), ["a", ""]);
    assertEquals(splitLines("\n"), [""]);
    assertEquals(splitLines("a\r\nb\rc\x0bd\u2028e"), ["a", "b", "c", "d", "e"]);
});

Deno.test("decodeUtf8 reads like Python's text mode: BOM kept, universal newlines, invalid UTF-8 throws", () => {
    const bytes = (text: string) => new TextEncoder().encode(text);
    assertEquals(decodeUtf8(bytes("\ufeffa\r\nb\rc\r\r\nd\n"), "x"), "\ufeffa\nb\nc\n\nd\n");
    assertThrows(() => decodeUtf8(new Uint8Array([0xff]), "x.md"), Error, "x.md: not valid UTF-8");
});

Deno.test("pyStrip and codePoints follow Python", () => {
    assertEquals(pyStrip("\x1c --- \x85"), "---");
    assertEquals(pyStrip("\ufeff---"), "\ufeff---");
    assertEquals(codePoints("😀a"), 2);
});

Deno.test("parseFrontmatter reads single-line, quoted, and block-scalar values", () => {
    const [fields, body] = parseFrontmatter(
        "---\nname: \"x\"\ndescription: >-\n  one\n\n  two\nother: 'y'\n---\nbody\nmore\n",
    );
    assertEquals(fields, { name: "x", description: "one two", other: "y" });
    assertEquals(body, "body\nmore");
    assertEquals(parseFrontmatter("---\r\nname: a\r\n---\r\nb\r\n"), [{ name: "a" }, "b"]);
});

Deno.test("parseFrontmatter returns null for missing or unterminated frontmatter", () => {
    assertEquals(parseFrontmatter("# Title\n")[0], null);
    assertEquals(parseFrontmatter("---\nname: a\n")[0], null);
    assertEquals(parseFrontmatter("\ufeff---\nname: a\n---\n")[0], null);
    assertEquals(parseFrontmatter("")[0], null);
});

Deno.test("normalizePath and relTo print paths as pathlib does", () => {
    assertEquals(normalizePath("./a//b/"), "a/b");
    assertEquals(normalizePath("."), ".");
    assertEquals(normalizePath("/r/a/./b"), "/r/a/b");
    assertEquals(relTo("/r", "/r/skills/x/"), "skills/x");
    assertEquals(relTo("/r", "/other/x"), "/other/x");
    assertEquals(relTo("/r", "fx/x"), "fx/x");
});

Deno.test("a clean skill has no findings", () => {
    withTemp((root) => {
        const dir = skill(root, "good", "name: good\ndescription: Does a thing. Use when x.\n");
        assertEquals(checkSkillDir(dir, root), []);
    });
});

Deno.test("missing SKILL.md and missing frontmatter stop the lint", () => {
    withTemp((root) => {
        Deno.mkdirSync(`${root}/empty`);
        assertEquals(messages(checkSkillDir(`${root}/empty`, root), ERROR).length, 1);
        assertMatch(checkSkillDir(`${root}/empty`, root)[0][1], /^empty: missing SKILL\.md/);
        Deno.mkdirSync(`${root}/plain`);
        Deno.writeTextFileSync(`${root}/plain/SKILL.md`, "# Plain\n");
        assertMatch(checkSkillDir(`${root}/plain`, root)[0][1], /^plain\/SKILL\.md: frontmatter missing/);
    });
});

Deno.test("name rules: missing, invalid, too long, mismatched", () => {
    withTemp((root) => {
        const noName = messages(checkSkillDir(skill(root, "no-name", "description: x\n"), root), ERROR);
        assertEquals(noName.length, 1);
        assertMatch(noName[0], /missing `name`.*Fix: add `name: no-name`/);

        const bad = messages(checkSkillDir(skill(root, "Bad_Name", "name: Bad_Name\ndescription: x\n"), root), ERROR);
        assertEquals(bad.length, 1);
        assertMatch(bad[0], /name `Bad_Name` is invalid/);

        const long = "a".repeat(65);
        const tooLong = messages(checkSkillDir(skill(root, long, `name: ${long}\ndescription: x\n`), root), ERROR);
        assertEquals(tooLong.length, 1);
        assertMatch(tooLong[0], /name is 65 chars; the spec maximum is 64/);

        const other = messages(checkSkillDir(skill(root, "dir", "name: other\ndescription: x\n"), root), ERROR);
        assertEquals(other.length, 1);
        assertMatch(other[0], /does not match the directory name `dir`/);
    });
});

Deno.test("description rules: missing, too long, warning band, code points", () => {
    withTemp((root) => {
        assertMatch(
            messages(checkSkillDir(skill(root, "d0", 'name: d0\ndescription: ""\n'), root), ERROR)[0],
            /missing or empty `description`/,
        );
        const long = checkSkillDir(skill(root, "d1", `name: d1\ndescription: ${"d".repeat(1025)}\n`), root);
        assertMatch(messages(long, ERROR)[0], /description is 1025 chars; the spec maximum is 1024/);
        const near = checkSkillDir(skill(root, "d2", `name: d2\ndescription: ${"d".repeat(901)}\n`), root);
        assertEquals(messages(near, ERROR), []);
        assertMatch(messages(near, WARNING)[0], /description is 901 chars, approaching/);
        // 1,000 astral characters are 1,000 characters for the spec, not 2,000 UTF-16 units.
        const astral = checkSkillDir(skill(root, "d3", `name: d3\ndescription: ${"😀".repeat(1000)}\n`), root);
        assertEquals(messages(astral, ERROR), []);
        assertMatch(messages(astral, WARNING)[0], /description is 1000 chars/);
    });
});

Deno.test("compatibility over the limit is an error", () => {
    withTemp((root) => {
        const findings = checkSkillDir(
            skill(root, "c", `name: c\ndescription: x\ncompatibility: ${"c".repeat(501)}\n`),
            root,
        );
        assertMatch(messages(findings, ERROR)[0], /compatibility is 501 chars; the spec maximum is 500/);
    });
});

Deno.test("a large body warns by lines or by tokens", () => {
    withTemp((root) => {
        const lines = checkSkillDir(skill(root, "l", "name: l\ndescription: x\n", "line\n".repeat(501)), root);
        assertMatch(messages(lines, WARNING)[0], /body is 501 lines/);
        const tokens = checkSkillDir(skill(root, "t", "name: t\ndescription: x\n", "t".repeat(20004)), root);
        assertMatch(messages(tokens, WARNING)[0], /body is 1 lines \/ ~5001 tokens/);
    });
});

Deno.test("README.md in a skill root is an error; empty subdirectories warn", () => {
    withTemp((root) => {
        const dir = skill(root, "r", "name: r\ndescription: x\n");
        Deno.writeTextFileSync(`${dir}/README.md`, "");
        for (const sub of ["scripts", "references", "assets"]) Deno.mkdirSync(`${dir}/${sub}`);
        const findings = checkSkillDir(dir, root);
        assertMatch(messages(findings, ERROR)[0], /^r: contains README\.md/);
        assertEquals(messages(findings, WARNING).map((m) => m.split(":")[0]), [
            "r/scripts/",
            "r/references/",
            "r/assets/",
        ]);
    });
});

Deno.test("repoSkillDirs lists public skills and real project skills, not symlinks", () => {
    withTemp((root) => {
        skill(`${root}/skills/core`, "a", "name: a\ndescription: x\n");
        Deno.writeTextFileSync(`${root}/skills/core/README.md`, "");
        skill(`${root}/.agents/skills`, "p", "name: p\ndescription: x\n");
        Deno.symlinkSync("../../skills/core/a", `${root}/.agents/skills/a`);
        assertEquals(repoSkillDirs(root), [`${root}/skills/core/a`, `${root}/.agents/skills/p`]);
    });
});

Deno.test("main: exit codes and streams", () => {
    withTemp((root) => {
        const good = skill(root, "good", "name: good\ndescription: x\n");
        const warn = skill(root, "warn", `name: warn\ndescription: ${"d".repeat(901)}\n`);
        const bad = skill(root, "bad", "name: other\ndescription: x\n");

        const ok = capture(() => main([good]));
        assertEquals([ok.result, ok.stdout, ok.stderr], [0, "OK: all skill checks passed.\n", ""]);

        const warned = capture(() => main([warn]));
        assertEquals(warned.result, 0);
        assertEquals(warned.stdout, "OK with 1 warning(s).\n");
        assert(warned.stderr.startsWith("WARNING: "));

        const failed = capture(() => main([good, bad]));
        assertEquals([failed.result, failed.stdout], [1, ""]);
        assertMatch(failed.stderr, /FAIL: 1 error\(s\), 0 warning\(s\)/);

        assertEquals(capture(() => main([`${root}/nope`])).result, 2);
        assertEquals(capture(() => main([])).result, 2);
        // The --all mode was dropped: validate_skills.ts covers every skill.
        const all = capture(() => main(["--all"]));
        assertEquals(all.result, 2);
        assertMatch(all.stderr, /unknown option --all/);
        const help = capture(() => main(["--help"]));
        assertEquals(help.result, 0);
        assert(help.stdout.startsWith("usage: "));
    });
});
