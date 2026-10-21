import { assert, assertEquals, assertMatch, assertStringIncludes, assertThrows } from "@std/assert";
import {
    checkBudget,
    checkChecksDoc,
    checkIntake,
    checkLabels,
    checkOpenspec,
    checkPointers,
    CHECKS,
    checkSpecLabels,
    checkSuite,
    checkTemplate,
    checkTools,
    checkWorkflows,
    type Deps,
    eventExpressions,
    formDropdownOptions,
    frontmatter,
    Harness,
    importSpecifiers,
    jobNames,
    justRecipes,
    main,
    markdownSection,
    pathLike,
    prContent,
    privilegedList,
    quote,
    quoteList,
    realDeps,
    scriptsRun,
    ShapeError,
    triggersOf,
    writeGrants,
} from "./validate_harness.ts";
import { ROOT, run } from "./lib/repo.ts";
import { LABEL, STATE_LABELS } from "./spec_approval.ts";

// --- fixture helpers ---------------------------------------------------------------------------------

const SHA = "3d3c42e5".repeat(5); // any 40-hex commit id
// Built by concatenation so this tracked file does not itself trip the `tools` check.
const OPENSPEC_PIN = "@fission-ai/openspec" + "@1.2.3";
const SETUP_TOFU = "setup-" + "opentofu";
const TOFU_PIN = "tofu_" + "version: 1.8.0";

function fixture(files: Record<string, string>): string {
    const root = Deno.makeTempDirSync({ prefix: "validate_harness_" });
    for (const [path, content] of Object.entries(files)) write(root, path, content);
    return root;
}

function write(root: string, path: string, content: string): void {
    const full = `${root}/${path}`;
    Deno.mkdirSync(full.slice(0, full.lastIndexOf("/")), { recursive: true });
    Deno.writeTextFileSync(full, content);
}

function cleanup(root: string): void {
    Deno.removeSync(root, { recursive: true });
}

const noDeps: Deps = {
    taxonomy: () => {
        throw new Error("taxonomy not stubbed");
    },
    trackedFiles: () => [],
    grepTracked: () => [],
};

/** Runs one check over a fixture and returns its findings. */
function findings(
    files: Record<string, string>,
    check: (h: Harness) => void,
    deps: Deps = noDeps,
): { errors: string[]; warnings: string[] } {
    const root = fixture(files);
    try {
        const h = new Harness(root, deps);
        check(h);
        return { errors: h.errors, warnings: h.warnings };
    } finally {
        cleanup(root);
    }
}

// --- workflows: one fixture per rule fault -----------------------------------------------------------

const CHECKS_DOC_LINE =
    "Privileged workflows: `labels.yml` (`pull_request_target`), `command.yml` (`issue_comment`), " +
    "`triage.yml` (`issues`).\n";

/** A pull_request workflow that passes every rule; `edit` swaps one line or block to inject a fault. */
const GOOD_PR = `name: ok
on:
  pull_request:
    types: [opened]
permissions: {}
jobs:
  build:
    name: ok / build
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@${SHA} # v7.0.1
        with:
          persist-credentials: false
      - uses: ./.github/actions/setup-tools
        with:
          just: "true"
      - name: Run
        env:
          N: \${{ github.event.pull_request.number }}
        run: echo "$N"
`;

/** A privileged workflow that passes every rule (listed with its trigger, base checkout, setup with no inputs). */
const GOOD_PRIVILEGED = `name: labels
on:
  pull_request_target:
    types: [opened]
permissions: {}
jobs:
  labels:
    name: spec / labels
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@${SHA} # v7.0.1
        with:
          persist-credentials: false
      - uses: ./.github/actions/setup-tools
      - name: Plan
        env:
          N: \${{ github.event.pull_request.number }}
        run: ./scripts/plan.ts --pr "$N"
`;

const LOCAL_SCRIPT = `import { helper } from "./lib/helper.ts";\nhelper();\n`;
const LOCAL_LIB = `export function helper() {}\n`;

function workflowFindings(workflows: Record<string, string>, extra: Record<string, string> = {}): string[] {
    const files: Record<string, string> = {
        ".agents/knowledge/github-checks.md": CHECKS_DOC_LINE,
        "scripts/plan.ts": LOCAL_SCRIPT,
        "scripts/lib/helper.ts": LOCAL_LIB,
        ...extra,
    };
    for (const [name, text] of Object.entries(workflows)) files[`.github/workflows/${name}`] = text;
    return findings(files, checkWorkflows).errors;
}

/** Asserts exactly one finding, naming `file:line`, whose text includes `fragment`. */
function assertOne(errors: string[], file: string, line: number, fragment: string): void {
    assertEquals(errors.length, 1, errors.join("\n"));
    assert(errors[0].startsWith(`${file}:${line}: `), errors[0]);
    assertStringIncludes(errors[0], fragment);
}

const PR = ".github/workflows/pr.yml";
const LABELS_WF = ".github/workflows/labels.yml";

Deno.test("workflows: the good fixtures pass, local setup-tools included", () => {
    assertEquals(workflowFindings({ "pr.yml": GOOD_PR, "labels.yml": GOOD_PRIVILEGED }), []);
});

Deno.test("workflows: missing top-level permissions", () => {
    assertOne(workflowFindings({ "pr.yml": GOOD_PR.replace("permissions: {}\n", "") }), PR, 1, "top-level");
});

Deno.test("workflows: missing job permissions", () => {
    const text = GOOD_PR.replace("    permissions:\n      contents: read\n", "");
    assertOne(workflowFindings({ "pr.yml": text }), PR, 7, "job 'build' declares no `permissions:`");
});

Deno.test("workflows: unpinned external action in a workflow", () => {
    const text = GOOD_PR.replace(`actions/checkout@${SHA} # v7.0.1`, "actions/checkout@v7");
    assertOne(workflowFindings({ "pr.yml": text }), PR, 14, "`actions/checkout@v7` is not pinned");
});

Deno.test("workflows: unpinned external action in an action file", () => {
    const action = `name: a
runs:
  using: composite
  steps:
    - uses: denoland/setup-deno@v2
      with:
        deno-version: vx.x.x
`;
    const errors = workflowFindings({ "pr.yml": GOOD_PR }, { ".github/actions/setup-tools/action.yml": action });
    assertOne(errors, ".github/actions/setup-tools/action.yml", 5, "`denoland/setup-deno@v2` is not pinned");
});

Deno.test("workflows: a pinned action without a version comment", () => {
    const text = GOOD_PR.replace(`actions/checkout@${SHA} # v7.0.1`, `actions/checkout@${SHA}`);
    assertOne(workflowFindings({ "pr.yml": text }), PR, 14, "version comment");
});

Deno.test("workflows: an event expression in run:", () => {
    const text = GOOD_PR.replace('run: echo "$N"', "run: echo ${{ github.event.pull_request.title }}");
    assertOne(workflowFindings({ "pr.yml": text }), PR, 23, "reaches a `run:` line");
});

Deno.test("workflows: the whole event in run: (bare or through toJSON); event_name passes", () => {
    for (const expr of ["${{ github.event }}", "${{ toJSON(github.event) }}"]) {
        const text = GOOD_PR.replace('run: echo "$N"', `run: echo "${expr}"`);
        assertOne(workflowFindings({ "pr.yml": text }), PR, 23, "reaches a `run:` line");
    }
    const ok = GOOD_PR.replace('run: echo "$N"', 'run: echo "${{ github.event_name }}"');
    assertEquals(workflowFindings({ "pr.yml": ok }), []);
});

Deno.test("workflows: github.head_ref in a multi-line run: block names its own line", () => {
    const text = GOOD_PR.replace('run: echo "$N"', 'run: |\n          set -e\n          echo "${{ github.head_ref }}"');
    assertOne(workflowFindings({ "pr.yml": text }), PR, 25, "`${{ github.head_ref }}` reaches a `run:` line");
});

Deno.test("workflows: an event expression in a script: input", () => {
    const text = GOOD_PR + `      - uses: actions/github-script@${SHA} # v8.0.0
        with:
          script: console.log("\${{ github.event.issue.title }}")
`;
    assertOne(workflowFindings({ "pr.yml": text }), PR, 26, "reaches a `script:` input");
});

Deno.test("workflows: event expressions outside run and script pass (env, with, if)", () => {
    const text = GOOD_PR.replace(
        "      - name: Run\n",
        "      - name: Run\n        if: github.event.pull_request.draft\n",
    );
    assertEquals(workflowFindings({ "pr.yml": text }), []);
});

Deno.test("workflows: checkout without persist-credentials: false", () => {
    const text = GOOD_PR.replace("        with:\n          persist-credentials: false\n", "");
    assertOne(workflowFindings({ "pr.yml": text }), PR, 14, "persist-credentials: false");
});

Deno.test("workflows: missing timeout-minutes", () => {
    const text = GOOD_PR.replace("    timeout-minutes: 5\n", "");
    assertOne(workflowFindings({ "pr.yml": text }), PR, 7, "timeout-minutes");
});

Deno.test("workflows: an unlisted pull_request_target workflow", () => {
    const text = GOOD_PRIVILEGED.replace("name: labels", "name: other");
    assertOne(workflowFindings({ "other.yml": text }), ".github/workflows/other.yml", 3, "`pull_request_target`");
});

Deno.test("workflows: an unlisted issue_comment workflow", () => {
    const text = GOOD_PRIVILEGED.replace("  pull_request_target:\n    types: [opened]", "  issue_comment:");
    // labels.yml is listed only with pull_request_target.
    assertOne(workflowFindings({ "labels.yml": text }), LABELS_WF, 3, "`issue_comment`");
});

Deno.test("workflows: a listed issue_comment workflow passes", () => {
    const text = GOOD_PRIVILEGED.replace("  pull_request_target:\n    types: [opened]", "  issue_comment:");
    assertEquals(workflowFindings({ "command.yml": text }), []);
});

Deno.test("workflows: a head checkout under pull_request_target", () => {
    const text = GOOD_PRIVILEGED.replace(
        "          persist-credentials: false\n",
        "          persist-credentials: false\n          ref: ${{ github.event.pull_request.head.sha }}\n",
    );
    assertOne(workflowFindings({ "labels.yml": text }), LABELS_WF, 18, "checks out with `ref:`");
});

/** GOOD_PRIVILEGED's deciding step with the head commit id in its own env (a value it compares). */
const HEAD_ENV = GOOD_PRIVILEGED.replace(
    "          N: ${{ github.event.pull_request.number }}\n",
    "          N: ${{ github.event.pull_request.number }}\n          EVENT_HEAD: ${{ github.event.pull_request.head.sha }}\n",
);

Deno.test("workflows: the head commit id in a privileged step's own env passes", () => {
    assertEquals(workflowFindings({ "labels.yml": HEAD_ENV }), []);
});

Deno.test("workflows: the head commit id in a job-level env under pull_request_target", () => {
    const text = GOOD_PRIVILEGED.replace(
        "    timeout-minutes: 10\n",
        "    timeout-minutes: 10\n    env:\n      EVENT_HEAD: ${{ github.event.pull_request.head.sha }}\n",
    );
    assertOne(workflowFindings({ "labels.yml": text }), LABELS_WF, 12, "pull request content");
});

Deno.test("workflows: the head commit id in env of a step that runs git", () => {
    const text = HEAD_ENV.replace('run: ./scripts/plan.ts --pr "$N"', 'run: git fetch origin "$EVENT_HEAD"');
    assertOne(workflowFindings({ "labels.yml": text }), LABELS_WF, 22, "pull request content");
});

Deno.test("workflows: the head commit id inside a longer env value, or the head ref, stays a finding", () => {
    for (
        const value of [
            "refs/pull/${{ github.event.pull_request.head.sha }}",
            "${{ github.event.pull_request.head.ref }}",
        ]
    ) {
        const text = HEAD_ENV.replace("${{ github.event.pull_request.head.sha }}", value);
        assertOne(workflowFindings({ "labels.yml": text }), LABELS_WF, 22, "pull request content");
    }
});

Deno.test("workflows: the head commit id in a privileged step's run: or with: stays a finding", () => {
    const inRun = GOOD_PRIVILEGED.replace(
        'run: ./scripts/plan.ts --pr "$N"',
        "run: ./scripts/plan.ts --head ${{ github.event.pull_request.head.sha }}",
    );
    const errors = workflowFindings({ "labels.yml": inRun });
    assertEquals(errors.length, 2, errors.join("\n"));
    assertStringIncludes(errors[0], "reaches a `run:` line");
    assertStringIncludes(errors[1], "pull request content");
    const inWith = HEAD_ENV.replace(
        "          persist-credentials: false\n",
        "          persist-credentials: false\n          ref: ${{ github.event.pull_request.head.sha }}\n",
    );
    assertOne(workflowFindings({ "labels.yml": inWith }), LABELS_WF, 18, "checks out with `ref:`");
});

Deno.test("workflows: a head checkout under issue_comment", () => {
    const text = GOOD_PRIVILEGED.replace("  pull_request_target:\n    types: [opened]", "  issue_comment:")
        .replace('run: ./scripts/plan.ts --pr "$N"', 'run: gh pr checkout "$N"');
    assertOne(workflowFindings({ "command.yml": text }), ".github/workflows/command.yml", 21, "pull request content");
});

Deno.test("workflows: a fetch of refs/pull under issues", () => {
    const text = GOOD_PRIVILEGED.replace("  pull_request_target:\n    types: [opened]", "  issues:")
        .replace('run: ./scripts/plan.ts --pr "$N"', 'run: git fetch origin "refs/pull/$N/head"');
    assertOne(workflowFindings({ "triage.yml": text }), ".github/workflows/triage.yml", 21, "pull request content");
});

Deno.test("workflows: a workflow_run trigger", () => {
    const text = GOOD_PR.replace("  pull_request:\n    types: [opened]", "  workflow_run:\n    workflows: [checks]");
    assertOne(workflowFindings({ "pr.yml": text }), PR, 3, "`workflow_run` is forbidden");
});

Deno.test("workflows: a write permission in a pull_request workflow", () => {
    const text = GOOD_PR.replace("      contents: read\n", "      contents: read\n      pull-requests: write\n");
    assertOne(workflowFindings({ "pr.yml": text }), PR, 13, "`pull-requests: write`");
});

Deno.test("workflows: write-all at the top of a pull_request workflow", () => {
    const text = GOOD_PR.replace("permissions: {}", "permissions: write-all");
    assertOne(workflowFindings({ "pr.yml": text }), PR, 5, "`write-all`");
});

Deno.test("workflows: a tool input in a privileged job", () => {
    const text = GOOD_PRIVILEGED.replace(
        "      - uses: ./.github/actions/setup-tools\n",
        '      - uses: ./.github/actions/setup-tools\n        with:\n          just: "true"\n',
    );
    assertOne(workflowFindings({ "labels.yml": text }), LABELS_WF, 18, "`just`");
});

Deno.test("workflows: another action in a privileged job", () => {
    const text = GOOD_PRIVILEGED.replace(
        "      - uses: ./.github/actions/setup-tools\n",
        `      - uses: actions/cache@${SHA} # v4.0.0\n`,
    );
    assertOne(workflowFindings({ "labels.yml": text }), LABELS_WF, 18, "uses `actions/cache@");
});

Deno.test("workflows: a privileged job's script with a transitive remote import", () => {
    const errors = workflowFindings({ "labels.yml": GOOD_PRIVILEGED }, {
        "scripts/lib/helper.ts": `import { parse } from "@std/yaml";\nexport function helper() { parse(""); }\n`,
    });
    assertOne(errors, LABELS_WF, 22, "the remote import `@std/yaml`");
    assertStringIncludes(errors[0], "scripts/lib/helper.ts");
});

Deno.test("workflows: a privileged job's script that does not exist", () => {
    const errors = workflowFindings({ "labels.yml": GOOD_PRIVILEGED.replace("plan.ts", "gone.ts") });
    assertOne(errors, LABELS_WF, 22, "scripts/gone.ts, which does not exist");
});

Deno.test("workflows: a privileged job running a remote module through deno", () => {
    const text = GOOD_PRIVILEGED.replace('./scripts/plan.ts --pr "$N"', "deno run -A jsr:@example/tool");
    assertOne(workflowFindings({ "labels.yml": text }), LABELS_WF, 22, "remote module through deno");
});

Deno.test("workflows: the same faults in a pull_request workflow are not privileged findings", () => {
    const text = GOOD_PR.replace('run: echo "$N"', 'run: gh pr checkout "$N"');
    assertEquals(workflowFindings({ "pr.yml": text }), []);
});

Deno.test("workflows: a missing privileged-workflow line stops the run", () => {
    const root = fixture({ ".agents/knowledge/github-checks.md": "no line\n", [PR]: GOOD_PR });
    try {
        assertThrows(() => checkWorkflows(new Harness(root, noDeps)), ShapeError, "Privileged workflows:");
    } finally {
        cleanup(root);
    }
});

Deno.test("workflows: an unparsable workflow stops the run", () => {
    const root = fixture({ ".agents/knowledge/github-checks.md": CHECKS_DOC_LINE, [PR]: "on: [\n" });
    try {
        assertThrows(() => checkWorkflows(new Harness(root, noDeps)), ShapeError, "not valid YAML");
    } finally {
        cleanup(root);
    }
});

Deno.test("workflows: helpers", () => {
    assertEquals(
        privilegedList(CHECKS_DOC_LINE),
        new Map([
            ["labels.yml", new Set(["pull_request_target"])],
            ["command.yml", new Set(["issue_comment"])],
            ["triage.yml", new Set(["issues"])],
        ]),
    );
    assertEquals(triggersOf("push"), ["push"]);
    assertEquals(triggersOf(["push", "pull_request"]), ["push", "pull_request"]);
    assertEquals(triggersOf({ issues: null, schedule: [] }), ["issues", "schedule"]);
    assertEquals(writeGrants({ contents: "read", issues: "write" }), ["issues"]);
    assertEquals(writeGrants("write-all"), ["*"]);
    assertEquals(writeGrants("read-all"), []);
    assertEquals(eventExpressions("a ${{ github.event_name }} b ${{ github.event.x }} ${{ github['event'] }}"), [
        "${{ github.event.x }}",
    ]);
    assertEquals(eventExpressions("${{ github.event['x'] }}"), ["${{ github.event['x'] }}"]);
    assertEquals(prContent("gh api repos/x/pulls/1 --jq .head.repo"), null);
    assertEquals(prContent("headRefOid")?.source, undefined);
    assertMatch("x merge_commit_sha", prContent("x merge_commit_sha")!);
    assertEquals(scriptsRun('./scripts/a.ts --x "$N" && deno run scripts/lib/b.ts; echo c.tsx'), [
        "scripts/a.ts",
        "scripts/lib/b.ts",
    ]);
    assertEquals(
        importSpecifiers(
            `// import "commented"\nimport {\n  a,\n} from "./a.ts";\nexport * from "jsr:@std/x";\nimport "./side.ts";\n` +
                `const m = await import("./dyn.ts");\nconst n = await import(name);\n`,
        ),
        ["./a.ts", "jsr:@std/x", "./side.ts", "./dyn.ts", null],
    );
});

// --- tools -------------------------------------------------------------------------------------------

/** A git repository whose index holds `tracked`; `untracked` files are only written. */
function gitFixture(tracked: Record<string, string>, untracked: Record<string, string> = {}): string {
    const root = fixture({ ...tracked, ...untracked });
    assertEquals(run("git", ["init", "-q"], { cwd: root }).code, 0);
    assertEquals(run("git", ["add", "--", ...Object.keys(tracked)], { cwd: root }).code, 0);
    return root;
}

Deno.test("tools: an OpenSpec version outside the justfile, with file and line; untracked files ignored", () => {
    const root = gitFixture({
        "justfile": `x := "npm:${OPENSPEC_PIN}"\n`,
        ".github/workflows/a.yml": `a\nrun: deno run npm:${OPENSPEC_PIN}\n`,
        "docs/ok.md": "npm:@fission-ai/openspec@{{openspec_version}}\n",
    }, { "scratch.md": `${OPENSPEC_PIN}\n` });
    try {
        const h = new Harness(root, realDeps);
        checkTools(h);
        assertEquals(h.errors.length, 1, h.errors.join("\n"));
        assert(h.errors[0].startsWith(".github/workflows/a.yml:2: "), h.errors[0]);
    } finally {
        cleanup(root);
    }
});

Deno.test("tools: an OpenTofu pin where it takes effect, with file and line; prose elsewhere passes", () => {
    const root = gitFixture({
        ".github/workflows/a.yml": `steps:\n  - uses: opentofu/${SETUP_TOFU}@x\n    with:\n      ${TOFU_PIN}\n`,
        "skills/x/y/assets/ci.yml": `${TOFU_PIN}\n`,
        "openspec/changes/archive/old/design.md": `${SETUP_TOFU}\n`,
        "openspec/changes/x/design.md": `Rejected: opentofu/${SETUP_TOFU} in CI\n`,
    });
    try {
        const h = new Harness(root, realDeps);
        checkTools(h);
        assertEquals(h.errors.map((e) => e.split(": ")[0]), [".github/workflows/a.yml:2", ".github/workflows/a.yml:4"]);
    } finally {
        cleanup(root);
    }
});

Deno.test("workflows: a .yaml workflow and an action.yaml are checked too", () => {
    const root = fixture({
        ".agents/knowledge/github-checks.md": CHECKS_DOC_LINE,
        ".github/workflows/x.yaml": "on: pull_request_target\npermissions: {}\njobs:\n  a:\n    runs-on: x\n" +
            "    timeout-minutes: 1\n    permissions: {}\n    steps:\n      - run: echo\n",
        ".github/actions/y/action.yaml": "runs:\n  using: composite\n  steps:\n    - uses: evil/act@main\n",
    });
    try {
        const h = new Harness(root, noDeps);
        checkWorkflows(h);
        assert(h.errors.some((e) => e.startsWith(".github/workflows/x.yaml")), h.errors.join("\n"));
        assert(h.errors.some((e) => e.startsWith(".github/actions/y/action.yaml")), h.errors.join("\n"));
    } finally {
        cleanup(root);
    }
});

Deno.test("tools: a tracked deno.lock, at the root or below", () => {
    const root = gitFixture({ "deno.lock": "{}", "sub/deno.lock": "{}", "justfile": "" });
    try {
        const h = new Harness(root, realDeps);
        checkTools(h);
        assertEquals(h.errors.length, 2);
        assertStringIncludes(h.errors[0], "deno.lock is tracked");
    } finally {
        cleanup(root);
    }
});

Deno.test("tools: a clean tree passes; git grep's no-match status is not a failure", () => {
    const root = gitFixture({ "justfile": "", "a.md": "nothing\n" });
    try {
        const h = new Harness(root, realDeps);
        checkTools(h);
        assertEquals(h.errors, []);
    } finally {
        cleanup(root);
    }
});

Deno.test("tools: git grep failing outside a repository is an interface failure", () => {
    const root = fixture({ "justfile": "" });
    try {
        assertThrows(() => realDeps.grepTracked(root, "x"), Error, "git grep");
    } finally {
        cleanup(root);
    }
});

// --- budget ------------------------------------------------------------------------------------------

function budgetFiles(agentsLines: number, extra: Record<string, string> = {}): Record<string, string> {
    return {
        "AGENTS.md": Array.from({ length: agentsLines }, (_, i) => `line ${i}`).join("\n") + "\n",
        "ARCHITECTURE.md": "# A\n",
        ".agents/knowledge/a.md": "| a | b |\n|---|---|\n",
        ...extra,
    };
}

Deno.test("budget: AGENTS.md at its budget passes, one line over fails", () => {
    assertEquals(findings(budgetFiles(110), checkBudget).errors, []);
    const { errors } = findings(budgetFiles(111), checkBudget);
    assertEquals(errors.length, 1);
    assertStringIncludes(errors[0], "AGENTS.md has 111 lines");
});

Deno.test("budget: a table row over 200 columns, with file and line", () => {
    const row = "| " + "x".repeat(197) + " |"; // 201 columns
    const ok = "| " + "x".repeat(196) + " |"; // 200 columns
    const { errors } = findings(
        budgetFiles(3, {
            "ARCHITECTURE.md": `# A\n\n${ok}\n${row}\n`,
            ".agents/knowledge/b.md": `text ${"y".repeat(300)}\n\`\`\`\n${row}\n\`\`\`\n  ${row}\n`,
            "AGENTS.md": `${row}\n`,
        }),
        checkBudget,
    );
    assertEquals(errors.map((e) => e.split(": ")[0]), [
        "AGENTS.md:1",
        "ARCHITECTURE.md:4",
        ".agents/knowledge/b.md:5",
    ]);
});

Deno.test("budget: main() runs it", () => {
    assert(CHECKS.includes(checkBudget));
});

// --- the existing checks on fixtures -----------------------------------------------------------------

const TEMPLATE = `## Related\nCloses #\n## Changes\n## Validation\n## Checklist\n- [ ] No secrets\n\nSpec: none\n`;

Deno.test("template: passes, then each missing piece is a finding", () => {
    const file = ".github/PULL_REQUEST_TEMPLATE.md";
    assertEquals(findings({ [file]: TEMPLATE }, checkTemplate).errors, []);
    assertStringIncludes(
        findings({ [file]: TEMPLATE.replace("## Validation", "## Other") }, checkTemplate).errors[0],
        "contains 'validation'",
    );
    assertStringIncludes(
        findings({ [file]: TEMPLATE.replace("- [ ] No secrets", "") }, checkTemplate).errors[0],
        "'secrets'",
    );
    assertStringIncludes(findings({ [file]: TEMPLATE.replace("Spec: none", "") }, checkTemplate).errors[0], "Spec:");
    // A heading inside a comment does not count.
    assertEquals(
        findings({ [file]: TEMPLATE.replace("## Validation", "<!-- ## Validation -->") }, checkTemplate).errors.length,
        1,
    );
});

Deno.test("intake: blank issues must be off", () => {
    const file = ".github/ISSUE_TEMPLATE/config.yml";
    assertEquals(findings({ [file]: "blank_issues_enabled: false\n" }, checkIntake).errors, []);
    assertEquals(findings({ [file]: "blank_issues_enabled: true\n" }, checkIntake).errors.length, 1);
    assertEquals(findings({ [file]: "contact_links: []\n" }, checkIntake).errors.length, 1);
});

const FORM = `name: Bug
labels:
  - bug
  - status/needs-triage
body:
  - type: dropdown
    attributes:
      label: Catalog
      options:
        - core
        - repository
  - type: dropdown
    attributes:
      label: Priority
      options:
        - high
        - low
`;
const LABEL_ROWS = [
    { name: "bug", applied_by: "form" },
    { name: "status/needs-triage", applied_by: "form" },
    { name: "catalog/core", applied_by: "triage" },
    { name: "catalog/repository", applied_by: "triage" },
    { name: "priority/high", applied_by: "triage" },
    { name: "priority/low", applied_by: "triage" },
    { name: "wontfix", applied_by: "human" },
];

function labelFiles(rows: unknown[] = LABEL_ROWS, form = FORM): Record<string, string> {
    return {
        ".github/labels.json": JSON.stringify(rows),
        ".github/ISSUE_TEMPLATE/bug.yml": form,
        ".github/ISSUE_TEMPLATE/config.yml": "blank_issues_enabled: false\n",
        ".github/dependabot.yml": 'version: 2\nupdates:\n  - package-ecosystem: x\n    labels: ["wontfix"]\n',
        "skills/core/.keep": "",
    };
}

Deno.test("labels: a consistent taxonomy passes", () => {
    assertEquals(findings(labelFiles(), checkLabels).errors, []);
});

Deno.test("labels: each inconsistency is a finding", () => {
    const missing = findings(labelFiles(LABEL_ROWS.filter((r) => r.name !== "bug")), checkLabels).errors;
    assertEquals(missing, [
        "label 'bug' is referenced by an issue form or dependabot.yml but missing from .github/labels.json; " +
        "GitHub would drop it silently.",
    ]);
    const badApplier = findings(labelFiles([...LABEL_ROWS, { name: "x", applied_by: "bot" }]), checkLabels).errors;
    assertStringIncludes(
        badApplier[0],
        `label 'x' needs "applied_by" set to one of ['form', 'human', 'maintainer', 'triage', 'workflow'].`,
    );
    const workflow = findings(labelFiles([...LABEL_ROWS, { name: "y", applied_by: "workflow" }]), checkLabels).errors;
    assertEquals(workflow.length, 2);
    const catalog = findings(labelFiles([...LABEL_ROWS, { name: "catalog/ml", applied_by: "triage" }]), checkLabels)
        .errors;
    assertStringIncludes(catalog[0], "catalog/* labels in .github/labels.json (['core', 'ml'])");
    const priority = findings(labelFiles(undefined, FORM.replace("        - low\n", "")), checkLabels).errors;
    assertStringIncludes(priority[0], "Priority options ['high'] differ from priority/* labels ['high', 'low']");
    const noDropdown = findings(labelFiles(undefined, FORM.replace("label: Catalog", "label: Area")), checkLabels)
        .errors;
    assertStringIncludes(noDropdown[0], "no dropdown labelled 'Catalog'");
});

Deno.test("labels: a label applied by the maintainer is applied by something", () => {
    assertEquals(
        findings(labelFiles([...LABEL_ROWS, { name: "spec/approved", applied_by: "maintainer" }]), checkLabels)
            .errors,
        [],
    );
});

Deno.test("labels: invalid labels.json stops the run", () => {
    const root = fixture({ ...labelFiles(), ".github/labels.json": "{" });
    try {
        assertThrows(() => checkLabels(new Harness(root, noDeps)), ShapeError, "is not valid JSON");
    } finally {
        cleanup(root);
    }
});

function pointerFiles(agents: string): Record<string, string> {
    return {
        "AGENTS.md": agents,
        "ARCHITECTURE.md": "See `scripts/` and `skills/core/` and `<name>` and `catalog/x`.\n",
        "README.md": "Run `just nope`.\n",
        "justfile": 'v := "1"\ncheck: lint\n    true\nlint:\n    true\nrecipe ARG:\n    true\n',
        ".agents/knowledge/a.md": "",
        "scripts/x.py": "",
        "skills/core/x/SKILL.md": "",
    };
}

Deno.test("pointers: names that exist pass; README misses are warnings", () => {
    const { errors, warnings } = findings(
        pointerFiles("Read `.agents/knowledge/a.md`, `scripts/x.py:3`, `core/x/`; run `just check`.\n"),
        checkPointers,
    );
    assertEquals(errors, []);
    assertEquals(warnings, ["README.md names `just nope`, which is not a justfile recipe."]);
});

Deno.test("pointers: a missing path, an unnamed knowledge file, an unknown recipe", () => {
    const { errors } = findings(pointerFiles("`scripts/gone.py` and `just gone`\n"), checkPointers);
    assertEquals(errors, [
        "AGENTS.md names `scripts/gone.py`, which does not exist.",
        ".agents/knowledge/a.md is not named in AGENTS.md; a knowledge file without a pointer is invisible.",
        "AGENTS.md names `just gone`, which is not a justfile recipe.",
    ]);
});

Deno.test("pointers: helpers", () => {
    assert(pathLike("scripts/x.py"));
    assert(pathLike(".github"));
    assert(!pathLike("catalog/core"));
    assert(!pathLike("README/X"));
    assert(!pathLike("a b/c"));
    assert(!pathLike("plain"));
    assertEquals(
        [...justRecipes('x := "1"\nexport PATH := "a"\ncheck: lint\nskill +PATHS:\nspec-check BASE="x":\n')],
        ["check", "skill"],
    );
});

const SECTION = "## Evidence discipline\n\nShared text.\n\n### Sub\n\nMore.\n\n";

function suiteFiles(member: string): Record<string, string> {
    return {
        "skills/engineering/brownfield-investigation/SKILL.md": `# I\n\n${SECTION}## Next\n`,
        "skills/engineering/brownfield-other/SKILL.md": member,
    };
}

Deno.test("suite: identical sections pass; a drifted or missing one is a finding", () => {
    assertEquals(findings(suiteFiles(`# O\n${SECTION}`), checkSuite).errors, []);
    assertStringIncludes(
        findings(suiteFiles(`# O\n${SECTION.replace("Shared", "Changed")}## Next\n`), checkSuite).errors[0],
        "section differs from",
    );
    assertStringIncludes(findings(suiteFiles("# O\n"), checkSuite).errors[0], "has no `## Evidence discipline`");
    const noSource = findings({ "skills/engineering/brownfield-other/SKILL.md": SECTION }, checkSuite).errors;
    assertStringIncludes(noSource[0], "is missing while other brownfield suite members exist");
});

Deno.test("suite: markdownSection stops at the next level-2 heading or the end", () => {
    assertEquals(markdownSection(`# T\n${SECTION}## Next\nx`, "## Evidence discipline"), SECTION);
    assertEquals(markdownSection(`# T\n${SECTION}`, "## Evidence discipline"), SECTION);
    assertEquals(markdownSection("# T\n", "## Evidence discipline"), null);
});

const STATES = ["spec/pending", "spec/approved", "spec/archived"];

function specDeps(states: string[] = STATES, approval = "spec/approved"): Deps {
    return { ...noDeps, taxonomy: () => ({ states, approval }) };
}

const STATE_ROWS = [
    { name: "spec/pending", applied_by: "workflow" },
    { name: "spec/approved", applied_by: "maintainer" },
    { name: "spec/archived", applied_by: "workflow" },
];

const stateLabelFiles = (rows: unknown[]) => ({ ".github/labels.json": JSON.stringify(rows) });

Deno.test("spec labels: labels.json declares exactly the state labels, the approval label the maintainer's", () => {
    assertEquals(findings(stateLabelFiles(STATE_ROWS), checkSpecLabels, specDeps()).errors, []);
    // A state the script writes and labels.json lacks.
    assertEquals(findings(stateLabelFiles(STATE_ROWS.slice(1)), checkSpecLabels, specDeps()).errors, [
        ".github/labels.json lacks 'spec/pending', which scripts/spec_approval.ts writes; add the row, applied_by " +
        "workflow.",
    ]);
    assertEquals(findings(stateLabelFiles([STATE_ROWS[0], STATE_ROWS[2]]), checkSpecLabels, specDeps()).errors, [
        ".github/labels.json lacks 'spec/approved', which scripts/spec_approval.ts writes; add the row, applied_by " +
        "maintainer.",
    ]);
    // A retired axis label, or any other spec/* label, is not exempt, whoever applies it.
    for (const applier of ["workflow", "maintainer"]) {
        assertEquals(
            findings(
                stateLabelFiles([...STATE_ROWS, { name: "spec/done", applied_by: applier }]),
                checkSpecLabels,
                specDeps(),
            )
                .errors,
            [
                ".github/labels.json: 'spec/done' is not a state label scripts/spec_approval.ts writes; remove it or " +
                "teach the script.",
            ],
        );
    }
    // The approval label is the maintainer's; the others the workflow's.
    const swapped = [
        { name: "spec/pending", applied_by: "human" },
        { name: "spec/approved", applied_by: "workflow" },
        STATE_ROWS[2],
    ];
    assertEquals(findings(stateLabelFiles(swapped), checkSpecLabels, specDeps()).errors, [
        ".github/labels.json: the state label 'spec/pending' must say applied_by workflow.",
        ".github/labels.json: the approval label 'spec/approved' must say applied_by maintainer.",
    ]);
    // An approval label outside the states is a mistake in the script.
    assertStringIncludes(
        findings(stateLabelFiles(STATE_ROWS), checkSpecLabels, specDeps(STATES, "spec/agreed")).errors[0],
        "the approval label 'spec/agreed' is not one of its STATE_LABELS",
    );
});

Deno.test("spec labels: a taxonomy of the wrong shape stops the run; the real one is spec_approval.ts's", () => {
    const root = fixture(stateLabelFiles(STATE_ROWS));
    try {
        assertThrows(
            () => checkSpecLabels(new Harness(root, { ...noDeps, taxonomy: () => ({ managed: [] }) })),
            ShapeError,
            "scripts/spec_approval.ts",
        );
    } finally {
        cleanup(root);
    }
    assertEquals(realDeps.taxonomy(ROOT), { states: [...STATE_LABELS], approval: LABEL });
});

Deno.test("checks doc: job names and the table agree both ways", () => {
    const files = {
        ".github/workflows/a.yml": "jobs:\n  one:\n    name: a / one\n  two:\n    runs-on: x\n",
        ".agents/knowledge/github-checks.md": "| `a / one` | x |\n| `gone` | y |\n",
    };
    assertEquals(findings(files, checkChecksDoc).errors, [
        "workflow job 'two' is not in the table of .agents/knowledge/github-checks.md.",
        ".agents/knowledge/github-checks.md lists 'gone' but no workflow job produces it.",
    ]);
    const root = fixture({ ...files, ".github/workflows/a.yml": "on: push\n" });
    try {
        assertThrows(() => checkChecksDoc(new Harness(root, noDeps)), ShapeError, "no top-level jobs");
    } finally {
        cleanup(root);
    }
});

Deno.test("checks doc: a conditional job name counts as both of its names; any other expression stops the run", () => {
    const carrier = "jobs:\n  carry:\n    name: ${{ needs.v.outputs.ok == 'true' && 'a / done' || 'a / waiting' }}\n";
    const both = "| `a / done` | x |\n| `a / waiting` | y |\n";
    const files = { ".github/workflows/a.yml": carrier, ".agents/knowledge/github-checks.md": both };
    assertEquals(findings(files, checkChecksDoc).errors, []);
    assertEquals(
        findings({ ...files, ".agents/knowledge/github-checks.md": "| `a / done` | x |\n" }, checkChecksDoc).errors,
        ["workflow job 'a / waiting' is not in the table of .agents/knowledge/github-checks.md."],
    );
    assertEquals(jobNames("w", "plain / name"), ["plain / name"]);
    for (const name of ["${{ matrix.os }}", "build ${{ matrix.os }}", "${{ a && 'x' }}"]) {
        assertThrows(() => jobNames("w", name), ShapeError, "is an expression other than");
    }
});

const CORE = [
    "openspec-apply-change",
    "openspec-archive-change",
    "openspec-explore",
    "openspec-propose",
    "openspec-sync-specs",
    "openspec-update-change",
];

function openspecFiles(stamp = "1.2.3"): Record<string, string> {
    const files: Record<string, string> = {
        "justfile": 'openspec_version := "1.2.3"\n',
        ".agents/skills/.openspec-target": "agents\n",
        ".github/workflows/checks.yml": "jobs: {}\n",
        ".devcontainer/devcontainer.json": '// comment\n{ "name": "x" }\n',
        "openspec/config.yaml": "schema: skill-change\n",
        "openspec/schemas/skill-change/schema.yaml": "artifacts:\n  - id: p\n    template: proposal.md\n",
        "openspec/schemas/skill-change/templates/proposal.md": "",
    };
    for (const name of CORE) {
        files[`.agents/skills/${name}/SKILL.md`] = `---\nname: ${name}\nmetadata:\n  generatedBy: "${stamp}"\n---\n`;
    }
    return files;
}

Deno.test("openspec: a consistent setup passes", () => {
    assertEquals(findings(openspecFiles(), checkOpenspec).errors, []);
});

Deno.test("openspec: each drift is a finding", () => {
    assertEquals(findings(openspecFiles("1.0.0"), checkOpenspec).errors.length, 6);
    const extra = { ...openspecFiles(), ".agents/skills/openspec-onboard/SKILL.md": "" };
    assertStringIncludes(findings(extra, checkOpenspec).errors[0], "expected ['openspec-apply-change'");
    const files = {
        ...openspecFiles(),
        ".agents/skills/.openspec-target": "claude\n",
        ".github/workflows/checks.yml": `run: npm i -g ${OPENSPEC_PIN} opentofu\n`,
        "openspec/config.yaml": "schema: spec-driven\nrules: {}\n",
        "openspec/schemas/skill-change/templates/proposal.md": undefined as unknown as string,
    };
    delete (files as Record<string, string | undefined>)["openspec/schemas/skill-change/templates/proposal.md"];
    const errors = findings(files, checkOpenspec).errors;
    assertEquals(errors.length, 6, errors.join("\n"));
    assertStringIncludes(errors.join("\n"), "names template 'proposal.md'");
});

Deno.test("openspec: no pin stops the run; frontmatter is parsed", () => {
    const root = fixture({ ...openspecFiles(), "justfile": "" });
    try {
        assertThrows(() => checkOpenspec(new Harness(root, noDeps)), ShapeError, "no openspec_version");
    } finally {
        cleanup(root);
    }
    assertEquals(frontmatter("---\na: 1\n---\nbody"), { a: 1 });
    assertEquals(frontmatter("no frontmatter"), null);
});

// --- helpers, main, and the real tree ----------------------------------------------------------------

Deno.test("Python-style quoting keeps the original messages", () => {
    assertEquals(quote("bug"), "'bug'");
    assertEquals(quote("it's"), `"it's"`);
    assertEquals(quoteList(["a", "b"]), "['a', 'b']");
    assertEquals(formDropdownOptions({ body: [{ attributes: { label: "Catalog" } }] }, "Catalog"), []);
    assertEquals(formDropdownOptions({ body: [] }, "Catalog"), null);
});

Deno.test("main: --help, a stray argument, and a directory outside git", () => {
    assertEquals(main(["--help"]), 0);
    assertEquals(main(["extra"]), 2);
    assertEquals(main(["--bogus"]), 2);
    const root = Deno.makeTempDirSync();
    try {
        assertEquals(main([], root), 2);
    } finally {
        cleanup(root);
    }
});

Deno.test("main: the real tree passes every check main() runs", () => {
    const h = new Harness(ROOT, realDeps);
    for (const check of CHECKS) check(h);
    assertEquals(h.errors, []);
});
