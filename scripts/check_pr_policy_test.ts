import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import {
    bodySections,
    check,
    checkSubject,
    commitSubjects,
    type GhRunner,
    InputError,
    loadEvent,
    loadPrViaGh,
    main,
    parseEvent,
    type PullRequest,
    pyRepr,
    pyStrip,
    splitLines,
    TEMPLATE,
    templateHeadings,
} from "./check_pr_policy.ts";
import { CommandError, run } from "./lib/repo.ts";

const CHECKLIST = [
    "`just check` passes locally",
    "The maintainer closed the package deliberation",
    "The scenarios are met and recorded above",
    "Every task of the linked change is done",
    "No secrets, credentials, or personal data in the diff, description, or commits",
    "Documentation is updated where required",
    "Catalog READMEs are synchronized",
];
const SPEC_LINK =
    "Spec: [openspec/changes/foo](https://github.com/ryan-minato/skills/tree/feat/x/openspec/changes/foo)";

interface BodyParts {
    spec?: string;
    phase?: string;
    related?: string;
    changes?: string;
    validation?: string;
    ticks?: string[];
    items?: string[];
    drop?: string[];
}

function body(parts: BodyParts = {}): string {
    const items = parts.items ?? CHECKLIST;
    const ticks = parts.ticks ?? items.map(() => "x");
    const sections: [string, string][] = [
        ["Why", "Because it helps."],
        ["Specification", `<!-- a comment -->\n${parts.spec ?? SPEC_LINK}\n${parts.phase ?? "Phase: implementation"}`],
        ["Related work", parts.related ?? "Closes #12"],
        ["Changes", parts.changes ?? "- [scripts/x.ts](https://github.com/ryan-minato/skills/blob/abc/scripts/x.ts)"],
        ["Validation", parts.validation ?? "- `just check`: pass"],
        ["Checklist", items.map((item, i) => `- [${ticks[i]}] ${item}`).join("\n")],
    ];
    const kept = sections.filter(([name]) => !(parts.drop ?? []).includes(name));
    return ["The goal.", ...kept.map(([name, text]) => `## ${name}\n\n${text}`)].join("\n\n") + "\n";
}

function pr(overrides: Partial<PullRequest> = {}): PullRequest {
    return {
        number: 7,
        title: "feat(x): add a thing",
        body: body(),
        draft: false,
        fork: false,
        baseSha: "b".repeat(40),
        headSha: "h".repeat(40),
        author: "ryan-minato",
        ...overrides,
    };
}

const goodSubjects = () => ["feat(x): add a thing", "fix: repair it"];
const findings = (p: PullRequest, subjects = goodSubjects) => check(p, TEMPLATE, subjects);

function eventPayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return {
        action: "edited",
        pull_request: {
            number: 7,
            title: "fix(x): repair it",
            body: body(),
            draft: false,
            user: { login: "ryan-minato" },
            head: { sha: "h".repeat(40), repo: { fork: true } },
            base: { sha: "b".repeat(40) },
            ...overrides,
        },
    };
}

function writeTemp(content: string): string {
    const path = Deno.makeTempFileSync({ suffix: ".json" });
    Deno.writeTextFileSync(path, content);
    return path;
}

Deno.test("splitLines follows str.splitlines", () => {
    assertEquals(splitLines(""), []);
    assertEquals(splitLines("a\r\nb\rc\nd\n"), ["a", "b", "c", "d"]);
    assertEquals(splitLines("a\n\n"), ["a", ""]);
    assertEquals(splitLines("a\u2028b\x1cc\fd"), ["a", "b", "c", "d"]);
});

Deno.test("pyRepr quotes like Python's repr", () => {
    assertEquals(pyRepr("plain"), "'plain'");
    assertEquals(pyRepr("it's"), `"it's"`);
    assertEquals(pyRepr(`it's "x"`), `'it\\'s "x"'`);
    assertEquals(pyRepr("a\\b\tc\n"), "'a\\\\b\\tc\\n'");
    assertEquals(pyRepr("caf\u00e9 \u00a0\u200b"), "'café \\xa0\\u200b'");
});

Deno.test("the repository template yields every role", () => {
    const { headings, roles } = templateHeadings(TEMPLATE);
    assert(headings.includes("Checklist"));
    assertEquals(roles.related, "Related work");
    assertEquals(roles.checklist, "Checklist");
});

Deno.test("a template without headings, without a role, or missing is an input error", () => {
    const dir = Deno.makeTempDirSync();
    try {
        Deno.writeTextFileSync(`${dir}/none.md`, "# Title\n<!-- ## Hidden -->\n");
        Deno.writeTextFileSync(`${dir}/roles.md`, "## Why\n## Related\n## Checklist\n");
        assertThrows(() => templateHeadings(`${dir}/none.md`), InputError, "has no '## ' headings");
        assertThrows(() => templateHeadings(`${dir}/roles.md`), InputError, "lacks a heading for: changes, validation");
        assertThrows(() => templateHeadings(`${dir}/absent.md`), InputError, "cannot read");
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});

Deno.test("bodySections starts a repeated heading over and ignores the preamble", () => {
    const sections = bodySections("intro\n## A\none\n## B\ntwo\n## A\nthree\n");
    assertEquals(sections.get("A"), "three\n");
    assertEquals(sections.get("B"), "two\n");
    assertEquals(sections.size, 2);
});

Deno.test("a compliant ready pull request has no findings", () => {
    assertEquals(findings(pr()), []);
    assertEquals(findings(pr({ body: body().replaceAll("\n", "\r\n") })), []);
});

Deno.test("missing template sections are findings, and a commented-out heading does not count", () => {
    const result = findings(pr({ body: body({ drop: ["Why"] }).replace("## Validation", "<!-- ## Validation -->") }));
    assertEquals(result.length, 3);
    assertStringIncludes(result[0], "missing the section '## Why'");
    assertStringIncludes(result[1], "missing the section '## Validation'");
    assertStringIncludes(result[2], "'## Validation' is empty");
});

Deno.test("the related section needs a closing keyword or N/A with a reason", () => {
    for (const related of ["Closes #12", "fixes #3", "Resolved #4 too", "N/A — chore", "n/a - nothing"]) {
        assertEquals(findings(pr({ body: body({ related }) })), [], related);
    }
    for (const related of ["Closes #", "N/A", "see #12"]) {
        const result = findings(pr({ body: body({ related }) }));
        assertEquals(result.length, 1, related);
        assertStringIncludes(result[0], "needs `Closes #N`");
    }
});

// The Python original's `re` and str.strip() semantics, kept character for character (see the pattern
// comment in check_pr_policy.ts). Each case below once differed between the two scripts.
const RELATED_FINDING = "needs `Closes #N`";

Deno.test("pyStrip strips Python's whitespace set, not trim()'s", () => {
    assertEquals(pyStrip(" \t\x1c\x1d\x1e\x1f\x85\u2028\u3000a b\u00a0\n"), "a b");
    assertEquals(pyStrip("\ufeff"), "\ufeff");
    assertEquals(pyStrip(""), "");
});

Deno.test("Python semantics: a letter right before the keyword or after the number is no closing reference", () => {
    for (const related of ["修复Closes #12", "Closes #12é", "Closes #12_", "xfixes #3"]) {
        const result = findings(pr({ body: body({ related }) }));
        assertEquals(result.length, 1, related);
        assertStringIncludes(result[0], RELATED_FINDING);
    }
    assertEquals(findings(pr({ body: body({ related: "(Closes #12)." }) })), []);
});

Deno.test("Python semantics: any decimal digit counts and case folds as in Python", () => {
    const accepted = [
        "Closes #\uff11\uff12",
        "Closes #\u0663",
        "clo\u017fes #12",
        "f\u0131xes #3",
        "F\u0130XES #3",
    ];
    for (const related of accepted) assertEquals(findings(pr({ body: body({ related }) })), [], related);
});

Deno.test("Python semantics: only \\n starts a Phase or Spec line", () => {
    for (const sep of ["\r", "\u2028", "\u2029", "\x85"]) {
        const phase = findings(pr({ body: body({ phase: `Note${sep}Phase: implementation` }) }));
        assertEquals(phase.length, 1, JSON.stringify(sep));
        assertStringIncludes(phase[0], "`Phase: implementation` line");
        const spec = findings(pr({ body: body({ spec: `Note${sep}${SPEC_LINK}` }) }));
        assertEquals(spec.length, 1, JSON.stringify(sep));
        assertStringIncludes(spec[0], "needs a `Spec:");
    }
    assertEquals(findings(pr({ body: body({ phase: "Phase: implementation\x1f", spec: `${SPEC_LINK}\x1c` }) })), []);
});

Deno.test("Python semantics: a section is empty by str.strip(), and a heading ends at Python whitespace", () => {
    const unitSep = findings(pr({ body: body({ changes: "\x1f" }) }));
    assertEquals(unitSep.length, 1);
    assertStringIncludes(unitSep[0], "'## Changes' is empty");
    assertEquals(findings(pr({ body: body({ validation: "\ufeff" }) })), []);
    assertEquals([...bodySections("## Changes\x1f\nx\n## Why\ufeff\ny\n").keys()], ["Changes", "Why\ufeff"]);
});

Deno.test("Python semantics: `.` stops only at \\n, and `$` also matches before a final \\n", () => {
    const out: string[] = [];
    for (const subject of ["fix(x): a\u2028b", "fix(x): a\rb", "fix(x): ab\n"]) checkSubject(subject, "Title", out);
    assertEquals(out, []);
    for (const subject of ["fix(x): a\nb", "fix(x): ab\n\n", "fix(x): a\n"]) checkSubject(subject, "Title", out);
    assertEquals(out.length, 3);
    assertEquals(findings(pr({ fork: true, title: "fix(x): a\u2028b" })), []);
});

Deno.test("the security checklist line must exist, even on a draft", () => {
    const items = CHECKLIST.filter((item) => !item.includes("secrets"));
    const result = findings(pr({ draft: true, body: body({ items }) }));
    assertEquals(result.length, 1);
    assertStringIncludes(result[0], "containing 'secrets' is missing");
});

Deno.test("a ready pull request needs filled sections, the implementation phase, and ticked boxes", () => {
    const result = findings(pr({
        body: body({
            changes: "_Reserved: filled in later._",
            validation: "-\n  -  \n",
            phase: "Phase: specification",
            ticks: [" ", "x", "x", "x", "x", "x", " "],
            items: [CHECKLIST[0], "The maintainer's call", ...CHECKLIST.slice(2)],
        }),
    }));
    assertEquals(result, [
        "'## Changes' still holds the template's reserved line on a ready pull request. " +
        "Fix: remove it and list every touched file as a permalink to its commit in the PR description.",
        "'## Validation' is empty on a ready pull request. " +
        "Fix: record what you ran and each scenario's result in the PR description.",
        "A ready pull request needs a `Phase: implementation` line. " +
        "Fix: get the specification approved on the draft, then set the line.",
        "Checklist item not ticked on a ready pull request: '`just check` passes locally'. " +
        "Fix: complete it and tick it.",
        "Checklist item not ticked on a ready pull request: 'Catalog READMEs are synchronized'. " +
        "Fix: complete it and tick it.",
    ]);
});

Deno.test("a draft skips the ready-only checks", () => {
    const draft = body({ changes: "_Reserved: x_", validation: "", phase: "Phase: specification", spec: "" });
    assertEquals(findings(pr({ draft: true, body: draft.replaceAll("- [x]", "- [ ]") })), []);
});

Deno.test("Spec accepts a link, a bare URL, an archived change, or none with a reason", () => {
    for (
        const spec of [
            SPEC_LINK,
            "Spec: https://github.com/ryan-minato/skills/blob/main/openspec/changes/foo/",
            "Spec: [openspec/changes/archive/2026-01-02-foo](https://github.com/o/r/tree/main/openspec/changes/archive/2026-01-02-foo)",
            "Spec: none — too small to plan",
        ]
    ) {
        assertEquals(findings(pr({ body: body({ spec }) })), [], spec);
    }
});

Deno.test("Spec refuses a bare path, a placeholder, or no line", () => {
    for (
        const spec of [
            "Spec: openspec/changes/foo",
            "Spec: openspec/changes/archive/2026-01-02-foo/",
            "Spec: [openspec/changes/<slug>](https://github.com/ryan-minato/skills/tree/<branch>/openspec/changes/<slug>)",
            "Spec: none",
            "",
        ]
    ) {
        const result = findings(pr({ body: body({ spec }) }));
        assertEquals(result.length, 1, spec);
        assertStringIncludes(result[0], "needs a `Spec: [openspec/changes/<slug>](<its URL on the branch>)` line");
    }
});

Deno.test("commit subjects follow Conventional Commits; merges, reverts, and fixups are exempt", () => {
    const out: string[] = [];
    for (
        const subject of [
            "feat(a, b-c): add",
            "fix!: break",
            "Merge pull request #1",
            'Revert "x"',
            "fixup! y",
            "squash! z",
        ]
    ) {
        checkSubject(subject, "Commit subject", out);
    }
    assertEquals(out, []);
    for (const subject of ["Fix the thing", "feat(X): upper scope", "feat: Upper", "feature: x", "feat:x"]) {
        checkSubject(subject, "Commit subject", out);
    }
    assertEquals(out.length, 5);
    assertStringIncludes(out[0], "Commit subject 'Fix the thing' is not a Conventional Commit subject");
});

Deno.test("a same-repository pull request is checked over its range, a fork over its title", () => {
    const seen: string[] = [];
    const subjects = (base: string, head: string) => {
        seen.push(`${base}..${head}`);
        return ["Bad subject"];
    };
    assertEquals(check(pr({ title: "Bad title" }), TEMPLATE, subjects).length, 1);
    assertEquals(seen, [`${"b".repeat(40)}..${"h".repeat(40)}`]);
    const fork = check(pr({ fork: true, title: "Bad title" }), TEMPLATE, subjects);
    assertEquals(fork.length, 1);
    assertStringIncludes(fork[0], "Pull request title (fork PRs are squash-merged) 'Bad title'");
    assertEquals(seen.length, 1);
});

Deno.test("Dependabot skips the body checks but not the subject check", () => {
    assertEquals(findings(pr({ author: "dependabot[bot]", body: "" })), []);
    const result = findings(pr({ author: "dependabot[bot]", body: "", fork: true, title: "Bump x" }));
    assertEquals(result.length, 1);
    assertStringIncludes(result[0], "'Bump x'");
});

Deno.test("an empty base or head SHA on a same-repository pull request is an input error", () => {
    assertThrows(() => findings(pr({ baseSha: "" })), InputError, "base and head SHAs are required");
});

Deno.test("parseEvent reads the fields and checks their shape", () => {
    const parsed = parseEvent(eventPayload({ body: null, head: { sha: "h", repo: null } }), "e.json");
    assertEquals(parsed, {
        number: 7,
        title: "fix(x): repair it",
        body: "",
        draft: false,
        fork: false,
        baseSha: "b".repeat(40),
        headSha: "h",
        author: "ryan-minato",
    });
    assertEquals(parseEvent(eventPayload(), "e.json").fork, true);
    assertThrows(() => parseEvent({ action: "push" }, "e.json"), InputError, "is not a pull_request event payload");
    assertThrows(() => parseEvent(eventPayload({ draft: "no" }), "e.json"), InputError, 'pull_request.draft is "no"');
    assertThrows(() => parseEvent(eventPayload({ base: { sha: null } }), "e.json"), InputError, "base.sha is null");
    assertThrows(() => parseEvent(eventPayload({ number: undefined }), "e.json"), InputError, "number is missing");
});

Deno.test("loadEvent reports an unreadable or invalid payload", () => {
    const bad = writeTemp("{not json");
    try {
        assertThrows(() => loadEvent(bad), InputError, "cannot read event payload");
        assertThrows(() => loadEvent(`${bad}.absent`), InputError, "cannot read event payload");
    } finally {
        Deno.removeSync(bad);
    }
});

function ghStub(code: number, stdout: string, calls: string[][] = []): GhRunner {
    return (args) => {
        calls.push(args);
        return { code, stdout, stderr: code === 0 ? "" : "HTTP 404: Not Found" };
    };
}

const GH_PR = {
    number: 12,
    title: "fix(x): repair it",
    body: "b",
    isDraft: true,
    isCrossRepository: true,
    baseRefOid: "base",
    headRefOid: "head",
    author: { login: "app/dependabot", is_bot: true },
};

Deno.test("loadPrViaGh maps gh pr view output", () => {
    const calls: string[][] = [];
    assertEquals(loadPrViaGh(12, ghStub(0, JSON.stringify(GH_PR), calls)), {
        number: 12,
        title: "fix(x): repair it",
        body: "b",
        draft: true,
        fork: true,
        baseSha: "base",
        headSha: "head",
        author: "app/dependabot",
    });
    assertEquals(calls[0].slice(0, 5), ["pr", "view", "12", "--repo", "ryan-minato/skills"]);
});

Deno.test("loadPrViaGh fails on a gh error, non-JSON output, or a wrong shape", () => {
    assertThrows(() => loadPrViaGh(12, ghStub(1, "")), CommandError, "exited 1: HTTP 404");
    assertThrows(() => loadPrViaGh(12, ghStub(0, "not json")), CommandError, "printed no JSON");
    assertThrows(() => loadPrViaGh(12, ghStub(0, "[]")), CommandError, "expected an object");
    assertThrows(
        () => loadPrViaGh(12, ghStub(0, JSON.stringify({ ...GH_PR, isDraft: null }))),
        InputError,
        "isDraft is null",
    );
});

Deno.test("commitSubjects lists the range and fails on an unknown revision", () => {
    const dir = Deno.makeTempDirSync();
    const git = (...args: string[]) => {
        const out = run("git", ["-c", "user.name=T", "-c", "user.email=t@example.com", ...args], { cwd: dir });
        assertEquals(out.code, 0, out.stderr);
        return out.stdout.trim();
    };
    try {
        git("init", "-q");
        git("commit", "-q", "--allow-empty", "-m", "chore: base");
        const base = git("rev-parse", "HEAD");
        git("commit", "-q", "--allow-empty", "-m", "Not conventional");
        git("commit", "-q", "--allow-empty", "-m", "feat: fine");
        const head = git("rev-parse", "HEAD");
        assertEquals(commitSubjects(base, head, dir), ["feat: fine", "Not conventional"]);
        assertThrows(
            () => commitSubjects("0".repeat(40), head, dir),
            CommandError,
            "shallow checkout or missing fetch",
        );
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});

Deno.test("main: help, usage errors, and exit codes", () => {
    assertEquals(main(["--help"]), 0);
    assertEquals(main([]), 2);
    assertEquals(main(["--event", "a", "--pr", "1"]), 2);
    assertEquals(main(["--pr", "abc"]), 2);
    assertEquals(main(["--bogus"]), 2);
    const good = writeTemp(JSON.stringify(eventPayload()));
    const bad = writeTemp(JSON.stringify(eventPayload({ title: "Bad title" })));
    const notPr = writeTemp(JSON.stringify({ action: "push" }));
    try {
        assertEquals(main(["--event", good]), 0);
        assertEquals(main(["--event", bad]), 1);
        assertEquals(main(["--event", notPr]), 2);
        assertEquals(main(["--event", good, "--template", `${good}.absent`]), 2);
    } finally {
        for (const path of [good, bad, notPr]) Deno.removeSync(path);
    }
    const fork = { ...GH_PR, body: body(), isDraft: false, author: { login: "ryan-minato" } };
    assertEquals(main(["--pr", "12"], ghStub(0, JSON.stringify(fork))), 0);
    assertEquals(main(["--pr", "12"], ghStub(1, "")), 2);
});
