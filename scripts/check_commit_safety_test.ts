import { assert, assertEquals, assertThrows } from "@std/assert";
import {
    attributionLines,
    checkRange,
    checkStaged,
    isAnonymousEmail,
    looksLikePlaceholder,
    main,
    parseRange,
    scanDiff,
} from "./check_commit_safety.ts";
import { runChecked, UsageError } from "./lib/repo.ts";

// Sensitive-looking fixtures are assembled from parts so that this file passes the repository's secret
// scanners and the gate itself; no vendor address appears literally.
const KEY_MARKER = ["-----BEGIN", "RSA", "PRIVATE", "KEY-----"].join(" ");
const KEY_ASSIGNMENT = ["api", "key"].join("_") + " = " + "zq81value";
const PRIVATE_EMAIL = ["jane.doe", "corp-mail.dev"].join("@");
const VENDOR_NOREPLY = ["noreply", "anthropic.com"].join("@");
const NOREPLY = "1+dev@users.noreply.github.com";
const SCRIPT = new URL("./check_commit_safety.ts", import.meta.url).pathname;

// Fixture repositories must not read the developer's git configuration (identity, hooks, signing).
const emptyConfig = Deno.makeTempFileSync();
Deno.env.set("GIT_CONFIG_GLOBAL", emptyConfig);
Deno.env.set("GIT_CONFIG_NOSYSTEM", "1");
for (const name of ["GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"]) Deno.env.delete(name);

function git(dir: string, ...args: string[]): string {
    return runChecked("git", args, { cwd: dir });
}

function fixture(email: string | null = NOREPLY): string {
    const dir = Deno.makeTempDirSync({ prefix: "commit-safety-" });
    git(dir, "init", "-q", "-b", "main");
    git(dir, "config", "user.name", "Dev");
    if (email !== null) git(dir, "config", "user.email", email);
    git(dir, "config", "commit.gpgsign", "false");
    Deno.writeTextFileSync(`${dir}/README.md`, "base\n");
    git(dir, "add", "README.md");
    git(dir, "-c", `user.email=${NOREPLY}`, "commit", "-q", "-m", "chore: base");
    return dir;
}

function commit(dir: string, file: string, content: string, message: string, email = NOREPLY): void {
    Deno.writeTextFileSync(`${dir}/${file}`, content);
    git(dir, "add", "-A");
    git(dir, "-c", `user.email=${email}`, "commit", "-q", "-m", message);
}

function quietly<T>(fn: () => T): T {
    const log = console.log;
    const error = console.error;
    console.log = () => {};
    console.error = () => {};
    try {
        return fn();
    } finally {
        console.log = log;
        console.error = error;
    }
}

function withFixture(email: string | null, fn: (dir: string) => void): void {
    const dir = fixture(email);
    try {
        fn(dir);
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
}

// ---- pure functions --------------------------------------------------------------------------------------------

Deno.test("isAnonymousEmail accepts GitHub and GitLab noreply only", () => {
    assert(isAnonymousEmail(NOREPLY));
    assert(isAnonymousEmail("dev@noreply.gitlab.com"));
    assert(isAnonymousEmail("dev@USERS.NOREPLY.GITLAB.COM"));
    assert(!isAnonymousEmail(PRIVATE_EMAIL));
    assert(!isAnonymousEmail("a b@users.noreply.github.com"));
    assert(!isAnonymousEmail(""));
});

Deno.test("looksLikePlaceholder", () => {
    for (
        const value of ["", "$TOKEN", "${TOKEN}", "${TOKEN", "<your-key>", "...", "REDACTED", "changeme", "'example'"]
    ) {
        assert(looksLikePlaceholder(value), value);
    }
    for (const value of ["zq81value", "$1abc", "example2"]) assert(!looksLikePlaceholder(value), value);
});

Deno.test("scanDiff reports keys, credentials, and personal emails under their file", () => {
    const diff = [
        "diff --git a/a.txt b/a.txt",
        "--- a/a.txt",
        "+++ b/a.txt",
        "@@ -0,0 +1,3 @@",
        `+${KEY_MARKER}`,
        `+${KEY_ASSIGNMENT}`,
        `+mail ${PRIVATE_EMAIL} and dev@example.com and ${NOREPLY}`,
        "diff --git a/b.txt b/b.txt",
        "+++ b/b.txt",
        "-removed " + PRIVATE_EMAIL,
        `+${KEY_ASSIGNMENT} # pragma: allowlist secret`,
        "+token = $GITHUB_TOKEN",
        "+password: <placeholder>",
    ].join("\n");
    assertEquals(scanDiff(diff), [
        "a.txt: added private key marker",
        "a.txt: added credential-like assignment for `api_key`",
        `a.txt: added personal email \`${PRIVATE_EMAIL}\``,
    ]);
});

Deno.test("scanDiff keeps Python's Unicode word boundaries", () => {
    // Python's \b treats non-ASCII letters as word characters; these results are the Python original's.
    assertEquals(scanDiff(`+++ b/x\n+\u00e9${PRIVATE_EMAIL}`), [
        `x: added personal email \`.${PRIVATE_EMAIL.slice("jane.".length)}\``,
    ]);
    assertEquals(scanDiff(`+++ b/x\n+${["ab", "corp-mail.dev"].join("@")}\u00e9`), []);
    assertEquals(scanDiff(`+++ b/x\n+(${PRIVATE_EMAIL})`), [`x: added personal email \`${PRIVATE_EMAIL}\``]);
    assertEquals(scanDiff("+++ b/x\n+mytoken = abcdef"), []);
});

Deno.test("scanDiff splits on newlines only, so a form feed cannot hide an added line's tail", () => {
    // Intentional difference: Python's str.splitlines() also split at \f, \v, \x1c-\x1e, \x85, and U+2028/9,
    // and the tail after such a character, lacking the leading "+", escaped the scan.
    assertEquals(scanDiff(`+++ b/f\n+x\f${KEY_ASSIGNMENT}`), ["f: added credential-like assignment for `api_key`"]);
    assertEquals(scanDiff(`+++ b/f\n+x\u2028${PRIVATE_EMAIL}`), [`f: added personal email \`${PRIVATE_EMAIL}\``]);
});

Deno.test("scanDiff honours a nosec suppression only at a Unicode word boundary, as Python does", () => {
    const assignment = ["pass", "word"].join("") + ": s3cr3tVal";
    assertEquals(scanDiff(`+++ b/x\n+${assignment} # nosec`), []);
    assertEquals(scanDiff(`+++ b/x\n+${assignment} #nosec B105`), []);
    assertEquals(scanDiff(`+++ b/x\n+${assignment} # nosec\u00e9`), [
        "x: added credential-like assignment for `password`",
    ]);
    assertEquals(scanDiff(`+++ b/x\n+${assignment} # nosecurity`), [
        "x: added credential-like assignment for `password`",
    ]);
});

Deno.test("attributionLines flags AI co-authors, vendor noreply addresses, and tool lines", () => {
    const flagged = [
        `Co-Authored-By: Claude Opus 5.5 <${VENDOR_NOREPLY}>`,
        "Co-authored-by: Claude <claude@users.noreply.github.com>",
        "co-authored-by: Claude Code",
        "Co-authored-by: claude[bot] <209825114+claude[bot]@users.noreply.github.com>",
        "Co-authored-by: claude-code <1+claude-code@users.noreply.github.com>",
        "Co-authored-by: Claude Agent <1+agent@users.noreply.github.com>",
        "Co-authored-by: Claude AI <1+agent@users.noreply.github.com>",
        "Co-authored-by: Claude.ai <1+agent@users.noreply.github.com>",
        "Co-authored-by: Claude 3.5 Sonnet <1+agent@users.noreply.github.com>",
        "Co-authored-by: ChatGPT <bot@users.noreply.github.com>",
        "Co-authored-by: GPT-5 <bot@users.noreply.github.com>",
        "Co-authored-by: OpenAI Codex <codex@users.noreply.github.com>",
        "Co-authored-by: Gemini <gemini@users.noreply.github.com>",
        "Co-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>",
        `Co-authored-by: Helper <${VENDOR_NOREPLY}>`,
        "\u{1F916} Generated with [Claude Code](https://claude.com/claude-code)",
        "Generated with Codex",
    ];
    for (const line of flagged) assertEquals(attributionLines(`feat: x\n\nbody\n\n${line}\n`), [line], line);

    const allowed = [
        "Co-authored-by: Jane Doe <1+jane@users.noreply.github.com>",
        "Co-authored-by: Claude Monet <2+monet@users.noreply.github.com>",
        "Co-authored-by: Claude Aimée Martin <3+claude@users.noreply.github.com>",
        "Co-authored-by: Claudette Colbert <4+cc@users.noreply.github.com>",
        "Signed-off-by: Jane Doe <1+jane@users.noreply.github.com>",
        "Co-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
        "Generated with `just gen-marketplace`.",
        "The Claude skill docs were generated with care.",
    ];
    for (const line of allowed) assertEquals(attributionLines(`feat: x\n\n${line}\n`), [], line);
});

Deno.test("parseRange", () => {
    assertEquals(parseRange("main..HEAD"), ["main", "HEAD"]);
    assertEquals(parseRange("v1.0..v1.1"), ["v1.0", "v1.1"]);
    for (const bad of ["main", "main...HEAD", "..HEAD", "main..", "a..b..c", "--all..HEAD", "main..-x"]) {
        assertThrows(() => parseRange(bad), UsageError, undefined, bad);
    }
});

// ---- staged mode -----------------------------------------------------------------------------------------------

Deno.test("staged: a clean change with a noreply email passes", () => {
    withFixture(NOREPLY, (dir) => {
        Deno.writeTextFileSync(`${dir}/a.txt`, "hello\n");
        git(dir, "add", "a.txt");
        assertEquals(quietly(() => checkStaged(dir)), []);
        assertEquals(quietly(() => main([], dir)), 0);
    });
});

Deno.test("staged: nothing staged fails", () => {
    withFixture(NOREPLY, (dir) => {
        assertEquals(quietly(() => checkStaged(dir)), [
            "No staged changes. Stage one atomic logical change before committing.",
        ]);
        assertEquals(quietly(() => main([], dir)), 1);
    });
});

Deno.test("staged: a private or empty author email fails", () => {
    withFixture(PRIVATE_EMAIL, (dir) => {
        Deno.writeTextFileSync(`${dir}/a.txt`, "hello\n");
        git(dir, "add", "a.txt");
        const errors = quietly(() => checkStaged(dir));
        assertEquals(errors.length, 1);
        assert(errors[0].startsWith("git config user.email is not a GitHub/GitLab anonymous address."));
    });
    withFixture(null, (dir) => {
        Deno.writeTextFileSync(`${dir}/a.txt`, "hello\n");
        git(dir, "add", "a.txt");
        assertEquals(quietly(() => checkStaged(dir)), [
            "git config user.email is empty. Configure an anonymous email.",
        ]);
    });
});

Deno.test("staged: each added-line finding fails", () => {
    for (
        const [content, expected] of [
            [`${KEY_MARKER}\n`, "k.txt: added private key marker"],
            [`${KEY_ASSIGNMENT}\n`, "k.txt: added credential-like assignment for `api_key`"],
            [`${PRIVATE_EMAIL}\n`, `k.txt: added personal email \`${PRIVATE_EMAIL}\``],
        ]
    ) {
        withFixture(NOREPLY, (dir) => {
            Deno.writeTextFileSync(`${dir}/k.txt`, content);
            git(dir, "add", "k.txt");
            assertEquals(quietly(() => checkStaged(dir)), [expected]);
            assertEquals(quietly(() => main([], dir)), 1);
        });
    }
});

Deno.test("staged: a non-UTF-8 added line is scanned, not a crash", () => {
    // Intentional difference: Python's text-mode decode raised UnicodeDecodeError (a traceback, exit 1);
    // the port decodes with U+FFFD replacement and still reports the finding.
    withFixture(NOREPLY, (dir) => {
        const bytes = new Uint8Array([
            0x63,
            0x61,
            0x66,
            0xe9,
            0x20,
            ...new TextEncoder().encode(`${KEY_ASSIGNMENT}\n`),
        ]);
        Deno.writeFileSync(`${dir}/l.txt`, bytes);
        git(dir, "add", "l.txt");
        assertEquals(quietly(() => checkStaged(dir)), ["l.txt: added credential-like assignment for `api_key`"]);
    });
});

// ---- range mode ------------------------------------------------------------------------------------------------

Deno.test("range: a clean noreply range passes", () => {
    withFixture(NOREPLY, (dir) => {
        commit(dir, "a.txt", "one\n", "feat: one");
        commit(dir, "b.txt", "two\n", "fix: two\n\nCo-authored-by: Jane Doe <1+jane@users.noreply.github.com>");
        assertEquals(quietly(() => checkRange("HEAD~2..HEAD", false, dir)), []);
        assertEquals(quietly(() => main(["--range", "HEAD~2..HEAD"], dir)), 0);
    });
});

Deno.test("range: a private author email fails on a same-repository range and passes with --fork", () => {
    withFixture(NOREPLY, (dir) => {
        commit(dir, "a.txt", "one\n", "feat: one", PRIVATE_EMAIL);
        const errors = quietly(() => checkRange("HEAD~1..HEAD", false, dir));
        assertEquals(errors.length, 1);
        assert(errors[0].includes("author email is not a GitHub/GitLab noreply address"));
        assert(!errors[0].includes(PRIVATE_EMAIL));
        assertEquals(quietly(() => main(["--range", "HEAD~1..HEAD"], dir)), 1);
        assertEquals(quietly(() => main(["--range", "HEAD~1..HEAD", "--fork"], dir)), 0);
    });
});

Deno.test("range: a key line fails", () => {
    withFixture(NOREPLY, (dir) => {
        commit(dir, "k.txt", `${KEY_MARKER}\n`, "feat: key");
        const sha = git(dir, "rev-parse", "HEAD").trim().slice(0, 12);
        assertEquals(quietly(() => checkRange("HEAD~1..HEAD", false, dir)), [
            `${sha}: k.txt: added private key marker`,
        ]);
    });
});

Deno.test("range: a key added and removed in a later commit still fails", () => {
    withFixture(NOREPLY, (dir) => {
        commit(dir, "k.txt", `x\n${KEY_ASSIGNMENT}\n`, "feat: add");
        const added = git(dir, "rev-parse", "HEAD").trim().slice(0, 12);
        commit(dir, "k.txt", "x\n", "fix: remove");
        assertEquals(git(dir, "diff", "HEAD~2", "HEAD", "--", "k.txt").includes(KEY_ASSIGNMENT), false);
        assertEquals(quietly(() => checkRange("HEAD~2..HEAD", false, dir)), [
            `${added}: k.txt: added credential-like assignment for \`api_key\``,
        ]);
        assertEquals(quietly(() => main(["--range", "HEAD~2..HEAD", "--fork"], dir)), 1);
    });
});

Deno.test("range: an attribution trailer fails, also with --fork", () => {
    withFixture(NOREPLY, (dir) => {
        const trailer = `Co-Authored-By: Claude Opus 5.5 <${VENDOR_NOREPLY}>`;
        commit(dir, "a.txt", "one\n", `feat: one\n\nBody.\n\n${trailer}`);
        const sha = git(dir, "rev-parse", "HEAD").trim().slice(0, 12);
        assertEquals(quietly(() => checkRange("HEAD~1..HEAD", true, dir)), [
            `${sha}: agent-attribution trailer \`${trailer}\`; remove it from the message`,
        ]);
        assertEquals(quietly(() => main(["--range", "HEAD~1..HEAD", "--fork"], dir)), 1);
    });
});

Deno.test("range: an empty range fails and an unknown revision is an interface error", () => {
    withFixture(NOREPLY, (dir) => {
        assertEquals(quietly(() => checkRange("HEAD..HEAD", false, dir)), [
            "No commits in HEAD..HEAD; check the base and head.",
        ]);
        assertEquals(quietly(() => main(["--range", "HEAD..HEAD"], dir)), 1);
        assertEquals(quietly(() => main(["--range", "HEAD..no-such-ref"], dir)), 2);
    });
});

Deno.test("range: a claude-code co-author fails through the shebang", () => {
    withFixture(NOREPLY, (dir) => {
        const trailer = "Co-authored-by: claude-code <1+claude-code@users.noreply.github.com>";
        commit(dir, "a.txt", "one\n", `feat: one\n\n${trailer}`);
        const out = new Deno.Command(SCRIPT, { args: ["--range", "HEAD~1..HEAD"], cwd: dir, stderr: "piped" })
            .outputSync();
        assertEquals(out.code, 1);
        assert(new TextDecoder().decode(out.stderr).includes(`agent-attribution trailer \`${trailer}\``));
    });
});

// ---- arguments and the real invocation -------------------------------------------------------------------------

Deno.test("argument errors exit 2; --help exits 0", () => {
    assertEquals(quietly(() => main(["--help"])), 0);
    assertEquals(quietly(() => main(["-h"])), 0);
    assertEquals(quietly(() => main(["--fork"])), 2);
    assertEquals(quietly(() => main(["--allow-private-email"])), 2);
    assertEquals(quietly(() => main(["extra"])), 2);
    assertEquals(quietly(() => main(["--range"])), 2);
    assertEquals(quietly(() => main(["--range", "main...HEAD"])), 2);
});

Deno.test("the shebang's permissions suffice for both modes", () => {
    withFixture(NOREPLY, (dir) => {
        commit(dir, "a.txt", "one\n", "feat: one");
        Deno.writeTextFileSync(`${dir}/b.txt`, "two\n");
        git(dir, "add", "b.txt");
        for (const args of [[], ["--range", "HEAD~1..HEAD"]]) {
            const out = new Deno.Command(SCRIPT, { args, cwd: dir, stdout: "piped", stderr: "piped" }).outputSync();
            const stdout = new TextDecoder().decode(out.stdout);
            assertEquals(out.code, 0, new TextDecoder().decode(out.stderr));
            assert(stdout.endsWith("Commit safety gate passed.\n"), stdout);
        }
    });
});
