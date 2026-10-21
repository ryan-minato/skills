import { assert, assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import {
    Api,
    b64decode,
    buildSnapshot,
    type Change,
    changeNames,
    cmpCodePoints,
    codeSpan,
    cpLen,
    cpSlice,
    Failure,
    type FetchFn,
    headDir,
    type Io,
    main,
    parseCommandLine,
    pyJson,
    pyQuote,
    pyStrip,
    select,
    type Snapshot,
    snapshotProblem,
    statusMarkdown,
    taskCounts,
} from "./spec_changes.ts";
import { UsageError } from "./lib/repo.ts";

// --- helpers -----------------------------------------------------------------------------------------------

function capture(overrides: Partial<Io> = {}): Io & { stdout: string; stderr: string } {
    const io = {
        stdout: "",
        stderr: "",
        out(text: string) {
            io.stdout += text;
        },
        err(text: string) {
            io.stderr += text;
        },
        env: (_name: string): string | undefined => undefined,
        fetch: (() => Promise.reject(new Error("no network in tests"))) as FetchFn,
        sleep: () => Promise.resolve(),
        ...overrides,
    };
    return io;
}

function git(cwd: string, ...args: string[]): string {
    const out = new Deno.Command("git", {
        args: ["-c", "user.name=t", "-c", "user.email=t@example.com", ...args],
        cwd,
        stdout: "piped",
        stderr: "piped",
    }).outputSync();
    if (out.code !== 0) throw new Error(`git ${args.join(" ")}: ${new TextDecoder().decode(out.stderr)}`);
    return new TextDecoder().decode(out.stdout).trim();
}

function write(root: string, path: string, text: string): void {
    Deno.mkdirSync(`${root}/${path}`.replace(/\/[^/]*$/, ""), { recursive: true });
    Deno.writeTextFileSync(`${root}/${path}`, text);
}

/** A repository whose base holds an active and an archived change and whose head adds the interesting states. */
function fixtureRepo(): { root: string; base: string; head: string } {
    const root = Deno.makeTempDirSync({ prefix: "spec_changes_test_" });
    git(root, "init", "-q", "-b", "main");
    write(root, "openspec/changes/gone/tasks.md", "- [ ] never\n");
    write(root, "openspec/changes/moving/tasks.md", "- [x] a\n- [ ] b\n");
    write(root, "openspec/changes/archive/2026-01-01-old/tasks.md", "- [x] a\n");
    git(root, "add", "-A");
    git(root, "commit", "-q", "-m", "base");
    const base = git(root, "rev-parse", "HEAD");
    Deno.removeSync(`${root}/openspec/changes/gone`, { recursive: true });
    Deno.removeSync(`${root}/openspec/changes/moving`, { recursive: true });
    write(root, "openspec/changes/archive/2026-02-02-moving/tasks.md", "- [x] a\n- [ ] b\n");
    write(root, "openspec/changes/open/proposal.md", "# Open\n");
    write(root, "openspec/changes/open/tasks.md", "- [x] 1.1 done\n- [ ] 1.2 open\n");
    write(root, "openspec/changes/open/specs/cap/spec.md", "## ADDED Requirements\n");
    write(root, "openspec/changes/repo/.openspec.yaml", "schema: skill-change\nskip_specs: true\n");
    write(root, "openspec/changes/repo/tasks.md", "- [x] only\n");
    git(root, "add", "-A");
    git(root, "commit", "-q", "-m", "head");
    return { root, base, head: git(root, "rev-parse", "HEAD") };
}

function change(name: string, state: Change["state"], progress: Change["tasks"]["progress"], open = 0): Change {
    return {
        name,
        state,
        path: state === "removed" ? null : `openspec/changes/${name}`,
        at_base: "removed",
        skip_specs: false,
        tasks: { done: progress === "not-started" ? 0 : 1, open, open_tasks: [], progress },
    };
}

// --- Python string semantics -------------------------------------------------------------------------------

Deno.test("text helpers count, slice, order, and strip as Python strings do", () => {
    assertEquals(cpLen("a😀b"), 3);
    assertEquals(cpSlice("😀😀😀", 2), "😀😀");
    assertEquals(cpSlice("ab", 5), "ab");
    // U+FF21 sorts before U+1F600 by code point, though its UTF-16 unit is larger than the surrogate's.
    assertEquals(["😀", "Ａ", "a"].sort(cmpCodePoints), ["a", "Ａ", "😀"]);
    assertEquals(pyStrip("\x1c\x85 task\u00a0\u3000"), "task");
    assertEquals(pyStrip("\ufefftask"), "\ufefftask");
});

Deno.test("pyJson escapes every non-ASCII character like json.dumps", () => {
    assertEquals(pyJson({ a: "—😀\x7f", b: [] }), '{\n  "a": "\\u2014\\ud83d\\ude00\\u007f",\n  "b": []\n}');
});

Deno.test("pyQuote encodes like urllib.parse.quote", () => {
    assertEquals(pyQuote("a b/c!*'()~_.-é"), "a%20b/c%21%2A%27%28%29~_.-%C3%A9");
    assertEquals(pyQuote("a/b", ""), "a%2Fb");
});

Deno.test("b64decode follows base64.b64decode without validation", () => {
    const text = (s: string) => new TextDecoder().decode(b64decode(s));
    assertEquals(text("YWJj\nZGVm\n"), "abcdef");
    assertEquals(text("YQ==YQ=="), "a");
    assertEquals(text("YQ==="), "a");
    assertThrows(() => b64decode("YQ"), Failure, "Incorrect padding");
    assertThrows(() => b64decode("YWJjZ"), Failure, "1 more than a multiple of 4");
    assertThrows(() => b64decode("YQ==é"), Failure, "only ASCII");
});

// --- decision logic ----------------------------------------------------------------------------------------

Deno.test("taskCounts reads ticked, open, and untitled tasks", () => {
    assertEquals(taskCounts(null), { done: 0, open: 0, open_tasks: [], progress: "not-started" });
    assertEquals(taskCounts(""), { done: 0, open: 0, open_tasks: [], progress: "not-started" });
    const tasks = taskCounts("- [x] a\n  - [X] b\r\n- [ ]\n-[ ]tight \r\n* [ ] not a task\n- [ ] c\u2028d\n");
    assertEquals(tasks.done, 2);
    assertEquals(tasks.open_tasks, ["(untitled task)", "tight", "c\u2028d"]);
    assertEquals(tasks.progress, "in-progress");
    assertEquals(taskCounts("- [x] a\n").progress, "done");
    assertEquals(taskCounts("- [ ] a\n").progress, "not-started");
});

Deno.test("changeNames and headDir resolve active and archived directories", () => {
    const names = changeNames(
        [
            "openspec/changes/b/tasks.md",
            "openspec/changes/archive/2026-01-02-a/tasks.md",
            "openspec/changes/archive/loose.md",
            "openspec/changes/loose.md",
            "openspec/specs/x/spec.md",
        ],
        "openspec/changes",
    );
    assertEquals(names, ["a", "b"]);
    const dirs = [
        "openspec/changes/archive/2026-01-01-a",
        "openspec/changes/archive/2026-03-01-a",
        "openspec/changes/b",
    ];
    assertEquals(headDir(dirs, "openspec/changes", "a"), "openspec/changes/archive/2026-03-01-a");
    assertEquals(headDir(dirs, "openspec/changes", "b"), "openspec/changes/b");
    assertEquals(headDir(dirs, "openspec/changes", "c"), null);
});

Deno.test("select refuses a change the request does not touch", () => {
    const changes = [change("a", "active", "done")];
    assertEquals(select(changes, []), changes);
    assertThrows(() => select(changes, ["b`x"]), UsageError, "``b`x``. Related changes: `a`.");
});

// --- rendering ---------------------------------------------------------------------------------------------

Deno.test("codeSpan keeps request text inside one code span", () => {
    assertEquals(codeSpan("a  b\nc"), "`a b c`");
    assertEquals(codeSpan("x`y"), "``x`y``");
    assertEquals(codeSpan("`x"), "`` `x ``");
    assertEquals(codeSpan(""), "`  `");
    assertEquals(codeSpan("a|b", true), "`a\\|b`");
});

Deno.test("statusMarkdown cuts long tasks by code point", () => {
    assertEquals(
        statusMarkdown([]),
        "No related OpenSpec change: the diff touches nothing under the changes directory.\n",
    );
    const c = change("x", "active", "in-progress", 1);
    c.tasks.open_tasks = ["😀".repeat(121)];
    const lines = statusMarkdown([c]).split("\n");
    assertEquals(lines[2], "| `x` | active | 1 done, 1 open | in-progress |");
    assertEquals(lines[5], "- `" + "😀".repeat(117) + "...`");
});

// --- the snapshot document ---------------------------------------------------------------------------------

function snapshotDoc(): Snapshot {
    return {
        schema: "spec-snapshot/1",
        repo: "o/r",
        pull_request: 1,
        changes_dir: "openspec/changes",
        changed_paths: ["openspec/changes/a/tasks.md"],
        base: { sha: "b", dirs: [] },
        head: {
            sha: "h",
            dirs: ["openspec/changes/a"],
            paths: ["openspec/changes/a/tasks.md"],
            files: { "openspec/changes/a/tasks.md": "- [x] a\n" },
        },
        bytes: 8,
    };
}

Deno.test("snapshotProblem names the first unusable part", () => {
    assertEquals(snapshotProblem(snapshotDoc(), "openspec/changes"), null);
    assertEquals(snapshotProblem([], "openspec/changes"), "it is not a spec-snapshot/1 document");
    assertStringIncludes(snapshotProblem(snapshotDoc(), "x")!, "built for the changes directory 'openspec/changes'");
    const doc = snapshotDoc() as unknown as Record<string, Record<string, unknown>>;
    doc.head.paths = "x";
    assertEquals(snapshotProblem(doc, "openspec/changes"), "`head.paths` is missing or not a list");
    doc.head.paths = [1];
    assertEquals(snapshotProblem(doc, "openspec/changes"), "`head.paths` holds a non-string");
});

// --- the REST reads, through a stubbed fetch ---------------------------------------------------------------

type Routes = Record<string, unknown | (() => Response)>;

function fakeFetch(routes: Routes, seen: string[] = []): FetchFn {
    return (url: string) => {
        const path = url.replace("https://api.test/", "");
        seen.push(path);
        if (!(path in routes)) return Promise.resolve(new Response("Not Found", { status: 404 }));
        const route = routes[path];
        if (typeof route === "function") return Promise.resolve((route as () => Response)());
        return Promise.resolve(new Response(JSON.stringify(route), { status: 200 }));
    };
}

const b64 = (text: string) => btoa(String.fromCharCode(...new TextEncoder().encode(text))).replace(/(.{8})/g, "$1\n");

function githubRoutes(): Routes {
    const tree = (entries: unknown[]) => ({ truncated: false, tree: entries });
    return {
        "repos/o/r/pulls/7": { base: { sha: "B" }, head: { sha: "H" }, changed_files: 3 },
        "repos/o/r/pulls/7/files?per_page=100&page=1": [
            { filename: "openspec/changes/a/tasks.md" },
            {
                filename: "openspec/changes/archive/2026-01-01-c/tasks.md",
                previous_filename: "openspec/changes/c/tasks.md",
            },
            { filename: "README.md" },
        ],
        "repos/o/r/git/trees/B": tree([{ path: "openspec", type: "tree", sha: "Bo" }]),
        "repos/o/r/git/trees/Bo": tree([{ path: "changes", type: "tree", sha: "Bc" }]),
        "repos/o/r/git/trees/Bc": tree([{ path: "c", type: "tree", sha: "Bcc" }]),
        "repos/o/r/git/trees/H": tree([{ path: "openspec", type: "tree", sha: "Ho" }]),
        "repos/o/r/git/trees/Ho": tree([{ path: "changes", type: "tree", sha: "Hc" }]),
        "repos/o/r/git/trees/Hc": tree([
            { path: "a", type: "tree", sha: "Ha" },
            { path: "archive", type: "tree", sha: "Harch" },
            { path: "x.md", type: "blob", sha: "Hx", size: 1 },
        ]),
        "repos/o/r/git/trees/Harch": tree([{ path: "2026-01-01-c", type: "tree", sha: "Hcc" }]),
        "repos/o/r/git/trees/Ha?recursive=1": tree([
            { path: "tasks.md", type: "blob", sha: "t1", size: 8 },
            { path: "notes.md", type: "blob", sha: "n1", size: 5 },
            { path: "specs", type: "tree", sha: "s" },
            { path: "specs/cap/spec.md", type: "blob", sha: "s1", size: 3 },
        ]),
        "repos/o/r/git/trees/Hcc?recursive=1": tree([{ path: "tasks.md", type: "blob", sha: "t2", size: 2 }]),
        "repos/o/r/git/blobs/t1": { encoding: "base64", content: b64("- [x] a\n") },
        "repos/o/r/git/blobs/s1": { encoding: "base64", content: b64("— \n") },
        "repos/o/r/git/blobs/t2": { encoding: "base64", content: b64("ok") },
    };
}

const CAPS = { maxFiles: 3000, maxFileBytes: 1_000_000, maxTotalBytes: 8_000_000 };

function api(routes: Routes, options: { maxCalls?: number; seen?: string[]; slept?: number[] } = {}): Api {
    const slept = options.slept ?? [];
    return new Api(
        "o/r",
        "https://api.test/",
        "token",
        options.maxCalls ?? 200,
        fakeFetch(routes, options.seen),
        (ms) => {
            slept.push(ms);
            return Promise.resolve();
        },
    );
}

Deno.test("buildSnapshot reads the related change documents and nothing else", async () => {
    const seen: string[] = [];
    const client = api(githubRoutes(), { seen });
    const doc = await buildSnapshot(client, 7, "openspec/changes", CAPS);
    assertEquals(doc.changed_paths, [
        "openspec/changes/a/tasks.md",
        "openspec/changes/archive/2026-01-01-c/tasks.md",
        "openspec/changes/c/tasks.md",
        "README.md",
    ]);
    assertEquals(doc.base.dirs, ["openspec/changes/c"]);
    assertEquals(doc.head.dirs, [
        "openspec/changes/a",
        "openspec/changes/archive",
        "openspec/changes/archive/2026-01-01-c",
    ]);
    assertEquals(doc.head.paths, [
        "openspec/changes/a/notes.md",
        "openspec/changes/a/specs/cap/spec.md",
        "openspec/changes/a/tasks.md",
        "openspec/changes/archive/2026-01-01-c/tasks.md",
    ]);
    assertEquals(doc.head.files, {
        "openspec/changes/a/tasks.md": "- [x] a\n",
        "openspec/changes/a/specs/cap/spec.md": "— \n",
        "openspec/changes/archive/2026-01-01-c/tasks.md": "ok",
    });
    assertEquals(doc.bytes, 13);
    assert(!seen.some((path) => path.includes("blobs/n1")), "notes.md is not a document the commands read");
    assertEquals(client.calls, seen.length);
    assertEquals(snapshotProblem(JSON.parse(pyJson(doc)), "openspec/changes"), null);
});

Deno.test("the REST reads retry transient failures and fail on everything else", async () => {
    const slept: number[] = [];
    let calls = 0;
    const flaky = githubRoutes();
    flaky["repos/o/r/pulls/7"] = () =>
        ++calls < 3
            ? new Response("busy", { status: 503 })
            : new Response(JSON.stringify({ base: { sha: "B" }, head: { sha: "H" }, changed_files: 3 }));
    await buildSnapshot(api(flaky, { slept }), 7, "openspec/changes", CAPS);
    assertEquals(slept, [2000, 4000]);

    const down = githubRoutes();
    down["repos/o/r/pulls/7"] = () => new Response("busy", { status: 503 });
    await assertRejects(
        () => buildSnapshot(api(down), 7, "openspec/changes", CAPS),
        Failure,
        "answered HTTP 503: busy",
    );

    await assertRejects(() => api({}).pullHeads(8), Failure, "Likely cause: the object is gone");

    const offline = new Api(
        "o/r",
        "https://api.test",
        "t",
        200,
        () => Promise.reject(new TypeError("dns")),
        () => Promise.resolve(),
    );
    await assertRejects(
        () => offline.pullHeads(7),
        Failure,
        "cannot reach https://api.test for GET repos/o/r/pulls/7: dns",
    );
    assertEquals(offline.calls, 3);

    await assertRejects(
        () => buildSnapshot(api(githubRoutes(), { maxCalls: 4 }), 7, "openspec/changes", CAPS),
        Failure,
        "--max-calls cap (4 API requests)",
    );
});

Deno.test("the REST reads refuse partial or malformed answers", async () => {
    const cases: [string, (routes: Routes) => void, string][] = [
        ["not JSON", (r) => (r["repos/o/r/pulls/7"] = () => new Response("<html>")), "did not return JSON"],
        ["wrong kind", (r) => (r["repos/o/r/pulls/7"] = []), "returned a list where the API documents an object"],
        ["missing key", (r) => (r["repos/o/r/pulls/7"] = { base: {}, head: {} }), "without `changed_files`"],
        [
            "no sha",
            (r) => (r["repos/o/r/pulls/7"] = { base: {}, head: {}, changed_files: 1 }),
            "a `base` without a `sha`",
        ],
        [
            "short listing",
            (r) => (r["repos/o/r/pulls/7"] = { base: { sha: "B" }, head: { sha: "H" }, changed_files: 9 }),
            "listed 3 of the request's 9",
        ],
        [
            "truncated",
            (r) => ((r["repos/o/r/git/trees/B"] as { truncated: boolean }).truncated = true),
            "came back truncated",
        ],
        [
            "entry without sha",
            (
                r,
            ) => (r["repos/o/r/git/trees/Hcc?recursive=1"] = {
                truncated: false,
                tree: [{ path: "x", type: "blob", size: 1 }],
            }),
            "without its path, type, sha",
        ],
        [
            "encoding",
            (r) => (r["repos/o/r/git/blobs/t1"] = { encoding: "utf-8", content: "x" }),
            "encoding 'utf-8', not base64",
        ],
        [
            "not UTF-8",
            (r) => (r["repos/o/r/git/blobs/t1"] = { encoding: "base64", content: "/w==" }),
            "is not UTF-8 text",
        ],
        [
            "unsafe name",
            (r) => (r["repos/o/r/pulls/7/files?per_page=100&page=1"] = [
                { filename: "openspec/changes/.hidden/tasks.md" },
                { filename: "a" },
                { filename: "b" },
            ]),
            "which is not a plain change name",
        ],
    ];
    for (const [name, mutate, message] of cases) {
        const routes = githubRoutes();
        mutate(routes);
        await assertRejects(() => buildSnapshot(api(routes), 7, "openspec/changes", CAPS), Failure, message, name);
    }
    await assertRejects(
        () => buildSnapshot(api(githubRoutes()), 7, "openspec/changes", { ...CAPS, maxFileBytes: 4 }),
        Failure,
        "over the --max-file-bytes cap (4)",
    );
    await assertRejects(
        () => buildSnapshot(api(githubRoutes()), 7, "openspec/changes", { ...CAPS, maxTotalBytes: 10 }),
        Failure,
        "--max-total-bytes cap (10)",
    );
    await assertRejects(
        () => buildSnapshot(api(githubRoutes()), 7, "openspec/changes", { ...CAPS, maxFiles: 2 }),
        Failure,
        "over the --max-files cap (2)",
    );
});

// --- the command line --------------------------------------------------------------------------------------

Deno.test("parseCommandLine keeps the original's options, defaults, and refusals", () => {
    const env = (name: string) => (name === "GITHUB_API_URL" ? "https://ghe.test/api/v3" : undefined);
    const opts = parseCommandLine(["--root", "/r", "--changes-dir=x/", "snapshot", "--repo", "o/r", "--pr", "12"], env);
    assert(typeof opts !== "string");
    assertEquals([opts.root, opts.changesDir, opts.pr, opts.apiUrl, opts.maxCalls], [
        "/r",
        "x/",
        12,
        "https://ghe.test/api/v3",
        200,
    ]);
    const status = parseCommandLine(["status", "--change", "a", "--change", "b"], env);
    assert(typeof status !== "string");
    assertEquals(status.change, ["a", "b"]);
    assertStringIncludes(parseCommandLine(["--help"], env) as string, "usage: spec_changes.ts");
    assertStringIncludes(parseCommandLine(["snapshot", "-h"], env) as string, "--max-calls N");
    const refused: [string[], string][] = [
        [[], "a command is required"],
        [["nope"], "invalid choice: 'nope'"],
        [["status", "--root", "x"], "unknown option --root"],
        [["status", "extra"], "unrecognized arguments: extra"],
        [["show"], "invalid choice: 'show'"],
        [["check", "--shape", "round"], "invalid choice: 'round'"],
        [["snapshot", "--repo", "o/r", "--pr", "0"], "0 is not a positive number"],
        [["snapshot", "--repo", "o/r", "--pr", "1", "--max-files", "ten"], "invalid positive value: 'ten'"],
        [["snapshot", "--pr", "1"], "required: --repo"],
        [["snapshot", "--repo", "o/r/x", "--pr", "1"], "is not an OWNER/NAME pair"],
        [["snapshot", "--repo", "o/r\n", "--pr", "1"], "is not an OWNER/NAME pair"],
    ];
    for (const [argv, message] of refused) assertThrows(() => parseCommandLine(argv, env), UsageError, message);
});

Deno.test("main exits 2 on bad arguments and 0 with help", async () => {
    let io = capture();
    assertEquals(await main(["status", "--bogus"], io), 2);
    assertStringIncludes(io.stderr, "unknown option --bogus");
    assertEquals(io.stdout, "");
    io = capture();
    assertEquals(await main(["show"], io), 2);
    assertStringIncludes(io.stderr, "invalid choice: 'show'");
    assertEquals(io.stdout, "");
    io = capture();
    assertEquals(await main(["-h"], io), 0);
    assertStringIncludes(io.stdout, "commands:");
    // The label command and check's --draft are gone: the state label is spec_approval.ts's, and a request
    // never fails for an unarchived change.
    io = capture();
    assertEquals(await main(["labels", "--taxonomy"], io), 2);
    assertStringIncludes(io.stderr, "invalid choice: 'labels'");
    io = capture();
    assertEquals(await main(["check", "--all", "--draft"], io), 2);
    assertStringIncludes(io.stderr, "unknown option --draft");
});

// --- the commands over a fixture repository ----------------------------------------------------------------

Deno.test("status, labels, and check read the git source", async () => {
    const { root, base, head } = fixtureRepo();
    try {
        const src = ["--base", base, "--head", head];
        let io = capture();
        assertEquals(await main(["--root", root, "status", ...src, "--json"], io), 0);
        const status = JSON.parse(io.stdout);
        assertEquals(status.base, base);
        assertEquals(
            status.changes.map((c: Change) => [c.name, c.state, c.at_base, c.skip_specs, c.tasks.progress]),
            [
                ["gone", "removed", "active", false, "not-started"],
                ["moving", "archived", "active", false, "in-progress"],
                ["open", "active", "removed", false, "in-progress"],
                ["repo", "active", "removed", true, "done"],
            ],
        );

        io = capture();
        assertEquals(await main(["--root", root, "status", ...src, "--change", "nope"], io), 2);
        assertStringIncludes(io.stderr, "not a change this request touches: `nope`");

        io = capture();
        // A request with unarchived changes passes with a warning for each: the archive gate holds the merge.
        assertEquals(await main(["--root", root, "check", ...src, "--no-validate"], io), 0);
        assertEquals(
            io.stdout.split("\n").filter((line) => line).map((line) => line.split(":").slice(0, 2).join(":")),
            [
                "warning: archived change moving still has 1 open task(s).",
                "warning: unarchived change open",
                "warning: unarchived change repo",
            ],
        );
        assertStringIncludes(io.stdout, "the archive gate holds the merge");
        io = capture();
        assertEquals(await main(["--root", root, "check", "--all", "--no-validate"], io), 1);
        assertEquals(
            io.stdout,
            [
                "unarchived change open: the integration branch holds only archived changes.",
                "unarchived change repo: the integration branch holds only archived changes.",
                "",
            ].join("\n"),
        );
        io = capture();
        assertEquals(await main(["--root", root, "check", "--all", "--no-validate", "--shape", "split"], io), 0);

        io = capture();
        assertEquals(await main(["--root", root, "status", "--base", "nope", "--head", head], io), 2);
        assertStringIncludes(io.stderr, "--base 'nope' does not resolve to a commit");
        io = capture();
        assertEquals(await main(["--root", root, "check", ...src, "--openspec", "no-such-openspec-cli"], io), 1);
        assertStringIncludes(io.stderr, "`no-such-openspec-cli`");
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
});

Deno.test("archive refuses open tasks, plans complete changes, and needs the checked-out head", async () => {
    const { root, base, head } = fixtureRepo();
    try {
        const src = ["--base", base, "--head", head];
        let io = capture();
        assertEquals(await main(["--root", root, "archive", ...src, "--dry-run", "--json"], io), 1);
        assertEquals(JSON.parse(io.stdout).refused, [{ name: "open", open: 1, tasks: ["1.2 open"] }]);
        assertStringIncludes(io.stderr, "nothing archived");

        write(root, "openspec/changes/open/tasks.md", "- [x] 1.1 done\n- [x] 1.2 done\n");
        git(root, "commit", "-q", "-am", "tick");
        const ticked = git(root, "rev-parse", "HEAD");
        io = capture();
        assertEquals(await main(["--root", root, "archive", "--base", base, "--head", ticked, "--dry-run"], io), 0);
        assertEquals(
            io.stdout,
            "would run: openspec archive open --yes\nwould run: openspec archive repo --yes --skip-specs\n",
        );
        io = capture();
        assertEquals(await main(["--root", root, "archive", ...src, "--dry-run"], io), 2);
        assertStringIncludes(io.stderr, "--head must be the checked-out HEAD");
        write(root, "openspec/dirty.md", "x");
        io = capture();
        assertEquals(await main(["--root", root, "archive", "--base", base, "--head", ticked, "--dry-run"], io), 1);
        assertStringIncludes(io.stderr, "openspec has uncommitted changes");
    } finally {
        Deno.removeSync(root, { recursive: true });
    }
});

Deno.test("snapshot writes the document the snapshot source replays", async () => {
    const dir = Deno.makeTempDirSync({ prefix: "spec_changes_snapshot_" });
    try {
        const out = `${dir}/snapshot.json`;
        const env = (name: string) => (name === "GH_TOKEN" ? "token" : undefined);
        let io = capture({ env, fetch: fakeFetch(githubRoutes()) });
        const argv = ["snapshot", "--repo", "o/r", "--pr", "7", "--out", out, "--api-url", "https://api.test"];
        assertEquals(await main(argv, io), 0);
        assertStringIncludes(io.stderr, "snapshot of o/r#7 at H: 3 file(s), 13 byte(s)");
        assertStringIncludes(Deno.readTextFileSync(out), '"\\u2014 \\n"');

        io = capture();
        assertEquals(await main(["status", "--snapshot", out, "--json"], io), 0);
        // `a` is done; the archived `c` has a task list with no task, so it has not started.
        assertEquals(
            JSON.parse(io.stdout).changes.map((c: Change) => [c.name, c.state, c.tasks.progress]),
            [["a", "active", "done"], ["c", "archived", "not-started"]],
        );
        io = capture();
        assertEquals(await main(["archive", "--snapshot", out], io), 2);
        io = capture();
        assertEquals(await main(["status", "--snapshot", out, "--base", "x"], io), 2);
        io = capture();
        assertEquals(await main(["--changes-dir", "other", "status", "--snapshot", out], io), 1);
        assertStringIncludes(io.stderr, "rebuild it with the snapshot command and the same --changes-dir");
        Deno.writeTextFileSync(out, "{");
        io = capture();
        assertEquals(await main(["status", "--snapshot", out], io), 1);
        assertStringIncludes(io.stderr, "cannot read the snapshot");

        io = capture();
        assertEquals(await main(argv, io), 1);
        assertStringIncludes(io.stderr, "needs a GitHub token in GH_TOKEN or GITHUB_TOKEN");
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});
