import { assert, assertEquals, assertMatch } from "@std/assert";
import { ROOT } from "./lib/repo.ts";
import {
    catalogSkillSubdirs,
    DISPOSABLE_MARKER,
    INSTALL_COMMAND,
    INSTALL_COMMAND_RE,
    main,
    markdownSection,
    pyListRepr,
    validate,
} from "./validate_skills.ts";

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

function write(path: string, text: string): void {
    Deno.mkdirSync(path.replace(/\/[^/]*$/, ""), { recursive: true });
    Deno.writeTextFileSync(path, text);
}

const METHODOLOGY = "## Harness Methodology\n\nShared text.\n\n";

/** Adds a public skill with its symlink; the marketplace is written by writeMarketplace. */
function addSkill(root: string, catalog: string, name: string, description: string, body = "# Body\n"): string {
    const dir = `${root}/skills/${catalog}/${name}`;
    write(`${dir}/SKILL.md`, `---\nname: ${name}\ndescription: ${description}\n---\n${body}`);
    Deno.symlinkSync(`../../skills/${catalog}/${name}`, `${root}/.agents/skills/${name}`);
    return dir;
}

/** Writes marketplace.json from the catalogs on disk, as `just gen-marketplace` does. */
function writeMarketplace(root: string, edit: (manifest: Record<string, unknown>) => void = () => {}): void {
    const plugins = [...Deno.readDirSync(`${root}/skills`)].map((e) => e.name).sort().map((catalog) => ({
        name: catalog,
        source: "./",
        skills: catalogSkillSubdirs(`${root}/skills/${catalog}`).map((d) =>
            `./skills/${catalog}/${d.split("/").pop()}`
        ),
    })).filter((p) => p.skills.length > 0);
    const manifest: Record<string, unknown> = { name: "skills", owner: { name: "owner" }, plugins };
    edit(manifest);
    write(`${root}/.claude-plugin/marketplace.json`, JSON.stringify(manifest, null, 2));
}

/** A minimal repository that passes every check. */
function fixture(): string {
    const root = Deno.realPathSync(Deno.makeTempDirSync());
    Deno.mkdirSync(`${root}/.agents/skills`, { recursive: true });
    for (const catalog of ["core", "meta"]) {
        for (const file of ["README.md", "README.zh.md", "CONTEXT.md"]) write(`${root}/skills/${catalog}/${file}`, "");
    }
    write(`${root}/skills/core/README.md`, `Install: ${INSTALL_COMMAND} ryan-minato/skills --skill <name>\n`);
    addSkill(root, "core", "alpha", "Does a thing. Use when x.");
    addSkill(
        root,
        "core",
        "ryan-minato-skills-installing",
        "Installs skills. Use when y.",
        `Run \`${INSTALL_COMMAND} ryan-minato/skills\`.\n`,
    );
    addSkill(root, "core", "meta-harness", "Explains harnesses. Use when z.", `# Meta\n\n${METHODOLOGY}`);
    addSkill(root, "meta", "meta-builder", `${DISPOSABLE_MARKER} builds. Use when w.`);
    addSkill(
        root,
        "meta",
        "meta-harness-architecture",
        `${DISPOSABLE_MARKER} manual.`,
        `# A\n\n${METHODOLOGY}## Next\n`,
    );
    write(`${root}/.agents/skills/project-only/SKILL.md`, "---\nname: project-only\ndescription: x\n---\n");
    write(`${root}/.agents/skills/project-only/notes.md`, `${INSTALL_COMMAND} is fine here\n`);
    write(`${root}/ARCHITECTURE.md`, "# A\n\n## Catalogs\n\n- `core` — c.\n- `meta` — m.\n\n## Other\n- `ghost`\n");
    write(`${root}/README.md`, "");
    write(`${root}/README.zh.md`, "");
    writeMarketplace(root);
    return root;
}

/** Runs `fn` on a fresh fixture and returns the validation result. */
function check(fn: (root: string) => void = () => {}): { errors: string[]; warnings: string[] } {
    const root = fixture();
    try {
        fn(root);
        return validate(root);
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
}

function only(result: { errors: string[] }, pattern: RegExp): string {
    assertEquals(result.errors.length, 1, result.errors.join("\n"));
    assertMatch(result.errors[0], pattern);
    return result.errors[0];
}

Deno.test("the fixture repository passes", () => {
    assertEquals(check(), { errors: [], warnings: [] });
});

Deno.test("the current repository passes, with no install command in a public skill", () => {
    const result = validate(ROOT);
    assertEquals(result.errors, []);
});

Deno.test("per-skill errors fail and warnings are collected", () => {
    only(check((root) => write(`${root}/.agents/skills/project-only/README.md`, "")), /contains README\.md/);
    const warned = check((root) =>
        write(
            `${root}/.agents/skills/project-only/SKILL.md`,
            `---\nname: project-only\ndescription: ${"d".repeat(901)}\n---\n`,
        )
    );
    assertEquals(warned.errors, []);
    assertEquals(warned.warnings.length, 1);
});

Deno.test("self-containment: an escaping link fails and the fix routes through the installer skill", () => {
    const message = only(
        check((root) => write(`${root}/skills/core/alpha/references/x.md`, "See [x](../../other/SKILL.md).\n")),
        /^skills\/core\/alpha\/references\/x\.md: link \]\(\.\.\/\.\.\/other\/SKILL\.md\) escapes/,
    );
    assertMatch(message, /load the `ryan-minato-skills-installing` skill/);
    assert(!message.includes(INSTALL_COMMAND), "the fix must not print an install command (D4)");
    only(check((root) => write(`${root}/skills/core/alpha/references/y.md`, "[y](/abs)\n")), /link \]\(\/abs\)/);
});

Deno.test("install command: fails in a public skill file, with its line", () => {
    const message = only(
        check((root) => write(`${root}/skills/meta/meta-builder/assets/setup.sh`, `#!/bin/sh\n${INSTALL_COMMAND} x\n`)),
        /^skills\/meta\/meta-builder\/assets\/setup\.sh:2: contains the install command `npx skills add`/,
    );
    assertMatch(message, /`ryan-minato-skills-installing`/);
    // Any file counts, not only Markdown, and every line is reported.
    const twice = check((root) =>
        write(
            `${root}/skills/core/alpha/SKILL.md`,
            `---\nname: alpha\ndescription: x\n---\n${INSTALL_COMMAND}\n${INSTALL_COMMAND}\n`,
        )
    );
    assertEquals(twice.errors.map((m) => m.split(": ")[0]), [
        "skills/core/alpha/SKILL.md:5",
        "skills/core/alpha/SKILL.md:6",
    ]);
});

Deno.test("install command: the installer skill, catalog files, and project skills are not scanned", () => {
    assertEquals(
        check((root) => {
            write(`${root}/skills/core/ryan-minato-skills-installing/references/more.md`, `${INSTALL_COMMAND}\n`);
            write(`${root}/skills/core/CONTEXT.md`, `${INSTALL_COMMAND}\n`);
            write(`${root}/skills/meta/README.zh.md`, `${INSTALL_COMMAND}\n`);
        }).errors,
        [],
    );
});

Deno.test("catalogs need README.md, README.zh.md, and CONTEXT.md", () => {
    only(check((root) => Deno.removeSync(`${root}/skills/meta/CONTEXT.md`)), /^skills\/meta: missing CONTEXT\.md/);
});

Deno.test("ARCHITECTURE.md catalog list", () => {
    only(check((root) => Deno.removeSync(`${root}/ARCHITECTURE.md`)), /^ARCHITECTURE\.md: missing\./);
    only(check((root) => write(`${root}/ARCHITECTURE.md`, "# A\n")), /no `## Catalogs` section found/);
    const result = check((root) => write(`${root}/ARCHITECTURE.md`, "## Catalogs\n- `core` — c.\n- `ghost` — g.\n"));
    assertEquals(result.errors.length, 2);
    assertMatch(result.errors[0], /catalog `meta` exists in skills\/ but is not listed.*add `- \\`meta\\``/);
    assertMatch(result.errors[1], /catalog `ghost` is listed but skills\/ghost\/ does not exist/);
});

Deno.test("prefix-reserving catalogs and the disposable marker", () => {
    only(
        check((root) => {
            addSkill(root, "meta", "builder", `${DISPOSABLE_MARKER} x.`);
            writeMarketplace(root);
        }),
        /^skills\/meta\/builder: name does not start with `meta-`/,
    );
    only(
        check((root) =>
            write(`${root}/skills/meta/meta-builder/SKILL.md`, "---\nname: meta-builder\ndescription: Builds.\n---\n")
        ),
        /^skills\/meta\/meta-builder\/SKILL\.md: description does not open with `Disposable builder skill/,
    );
});

Deno.test("symlinks: dangling, missing, wrong target, missing directory", () => {
    only(
        check((root) => Deno.symlinkSync("../../skills/core/nowhere", `${root}/.agents/skills/zz`)),
        /^\.agents\/skills\/zz: dangling symlink/,
    );
    only(
        check((root) => Deno.removeSync(`${root}/.agents/skills/alpha`)),
        /^skills\/core\/alpha: no symlink in \.agents\/skills\/\. .*ln -s \.\.\/\.\.\/skills\/core\/alpha \.agents\/skills\/alpha$/,
    );
    let root = "";
    const wrong = check((r) => {
        root = r;
        Deno.removeSync(`${r}/.agents/skills/alpha`);
        Deno.symlinkSync("../../skills/meta/meta-builder", `${r}/.agents/skills/alpha`);
    });
    only(
        wrong,
        new RegExp(`^\\.agents/skills/alpha: points to ${root}/skills/meta/meta-builder instead of skills/core/alpha`),
    );
    const dangling = check((r) => {
        root = r;
        Deno.removeSync(`${r}/.agents/skills/alpha`);
        Deno.symlinkSync("../../skills/core/gone", `${r}/.agents/skills/alpha`);
    });
    assertEquals(dangling.errors.length, 2);
    assertMatch(dangling.errors[1], new RegExp(`points to ${root}/skills/core/gone instead of`));
    const missing = check((r) => Deno.removeSync(`${r}/.agents/skills`, { recursive: true }));
    assertEquals(missing.errors, [missing.errors.find((m) => m.startsWith(".agents/skills/: missing."))]);
});

Deno.test("legacy plugin.json guard is kept unchanged", () => {
    const message = only(
        check((root) => write(`${root}/.claude-plugin/plugin.json`, '{"name": "skills"}\n')),
        /^\.claude-plugin\/plugin\.json:/,
    );
    assertEquals(
        message,
        ".claude-plugin/plugin.json: the repo now publishes as a marketplace (.claude-plugin/marketplace.json), " +
            "not one plugin. A leftover plugin.json is merged into every marketplace-root plugin and cross-loads " +
            "all catalogs. Fix: remove it (git rm .claude-plugin/plugin.json).",
    );
});

Deno.test("marketplace: missing, invalid, and malformed top level", () => {
    only(check((root) => Deno.removeSync(`${root}/.claude-plugin/marketplace.json`)), /marketplace\.json: missing\./);
    only(check((root) => write(`${root}/.claude-plugin/marketplace.json`, "{")), /invalid JSON \(/);
    only(check((root) => write(`${root}/.claude-plugin/marketplace.json`, "[]")), /top level must be an object/);
    const result = check((root) =>
        writeMarketplace(root, (m) => {
            delete m.name;
            m.owner = {};
        })
    );
    assertEquals(result.errors.length, 2);
    assertMatch(result.errors[0], /missing top-level `name`/);
    assertMatch(result.errors[1], /missing `owner\.name`/);
    only(check((root) => writeMarketplace(root, (m) => (m.plugins = []))), /`plugins` must be a non-empty list/);
});

Deno.test("marketplace: plugin entries", () => {
    type Plugin = { name: string; source: string; skills: unknown };
    const edit = (fn: (plugins: Plugin[]) => void) => (root: string) =>
        writeMarketplace(root, (m) => fn(m.plugins as Plugin[]));
    only(check(edit((p) => (p as unknown[]).push("x"))), /every `plugins` entry must be an object/);
    const renamed = check(edit((p) => (p[0].name = "Core_X")));
    assertMatch(renamed.errors[0], /plugin name `Core_X` is not kebab-case/);
    assertMatch(renamed.errors[1], /plugin `Core_X` has no catalog skills\/Core_X\//);
    assertMatch(renamed.errors[2], /catalog `core` has skills but no plugin entry/);
    only(check(edit((p) => (p[1].source = "./skills"))), /plugin `meta` source must be `".\/"`/);
    const traversal = check(edit((p) => (p[1].skills as string[]).push("../x")));
    assertMatch(traversal.errors[0], /skills path `\.\.\/x` must be a relative path starting with `\.\/`/);
    assertMatch(traversal.errors[1], /out of sync with skills\/meta\/ \(stale \['\.\.\/x'\]\)/);
    only(
        check(edit((p) => (p[1].skills = "./skills/meta"))),
        /out of sync with skills\/meta\/ \(missing \['\.\/skills\/meta\/meta-builder', '\.\/skills\/meta\/meta-harness-architecture'\], stale \['\.\/skills\/meta'\]\)/,
    );
    only(check(edit((p) => (p[1].skills = 3))), /plugin `meta` `skills` must be a list of path strings/);
    only(check(edit((p) => p.splice(1, 1))), /catalog `meta` has skills but no plugin entry/);
    only(
        check(edit((p) => p.push({ name: "ghost", source: "./", skills: [] }))),
        /plugin `ghost` has no catalog skills\/ghost\/ with skills/,
    );
});

Deno.test("root README.zh.md must exist beside README.md", () => {
    only(check((root) => Deno.removeSync(`${root}/README.zh.md`)), /^README\.zh\.md: missing\./);
});

Deno.test("harness methodology sections stay identical", () => {
    only(
        check((root) =>
            write(
                `${root}/skills/core/meta-harness/SKILL.md`,
                `---\nname: meta-harness\ndescription: x\n---\n${METHODOLOGY}more\n`,
            )
        ),
        /^meta methodology drifted/,
    );
    only(
        check((root) =>
            write(`${root}/skills/core/meta-harness/SKILL.md`, "---\nname: meta-harness\ndescription: x\n---\n")
        ),
        /^skills\/core\/meta-harness\/SKILL\.md: missing `## Harness Methodology`/,
    );
});

Deno.test("CRLF files read as Python's universal newlines do", () => {
    const toCrlf = (path: string) => write(path, Deno.readTextFileSync(path).replaceAll("\n", "\r\n"));
    // ARCHITECTURE.md and both methodology sources in CRLF: the `$`-anchored headings still match.
    assertEquals(
        check((root) => {
            toCrlf(`${root}/ARCHITECTURE.md`);
            toCrlf(`${root}/skills/core/meta-harness/SKILL.md`);
            toCrlf(`${root}/skills/meta/meta-harness-architecture/SKILL.md`);
        }),
        { errors: [], warnings: [] },
    );
    // One methodology source CRLF, the other LF: still identical after the newline translation.
    assertEquals(check((root) => toCrlf(`${root}/skills/core/meta-harness/SKILL.md`)).errors, []);
    // A CRLF ARCHITECTURE.md still reports real catalog drift rather than a missing section.
    only(
        check((root) => write(`${root}/ARCHITECTURE.md`, "## Catalogs\r\n- `core` — c.\r\n")),
        /catalog `meta` exists in skills\/ but is not listed/,
    );
});

Deno.test("markdownSection and pyListRepr match the Python helpers", () => {
    assertEquals(markdownSection("x\n## A\nbody\n### sub\n## B\n", "## A"), "## A\nbody\n### sub");
    assertEquals(markdownSection("## A \n", "## A"), null);
    assertEquals(pyListRepr(["a", "it's", "q\"'"]), `['a', "it's", 'q"\\'']`);
});

Deno.test("main: OK line on stdout, findings on stderr, argument errors exit 2", () => {
    const ok = capture(() => main([]));
    assertEquals(ok.result, 0);
    assertMatch(ok.stdout, /^OK: skill layout and harness consistency checks passed/);
    const root = fixture();
    try {
        Deno.removeSync(`${root}/README.zh.md`);
        const failed = capture(() => main([], root));
        assertEquals([failed.result, failed.stdout], [1, ""]);
        assertMatch(failed.stderr, /^FAIL: 1 problem\(s\) found\n\n {2}\* README\.zh\.md: missing\./);
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
    assertEquals(capture(() => main(["extra"])).result, 2);
    const help = capture(() => main(["-h"]));
    assertEquals(help.result, 0);
    assert(help.stdout.startsWith("usage: "));
});

Deno.test("install command: any package runner invoking skills add is found", () => {
    for (
        const line of ["npx skills add a/b", "pnpm dlx skills add a/b", "npx -y skills add a/b", "bunx skills add x"]
    ) {
        assert(INSTALL_COMMAND_RE.test(line), line);
    }
    assert(!INSTALL_COMMAND_RE.test("the skills add-on list"));
});
