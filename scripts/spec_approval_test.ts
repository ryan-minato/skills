import { assert, assertEquals, assertMatch, assertRejects, assertThrows } from "@std/assert";
import { parse as parseYaml } from "@std/yaml";
import {
    ARCHIVED,
    archivedCopy,
    changesOf,
    type Event,
    EVENT_VARIABLES,
    type Gh,
    ghApi,
    LABEL,
    main,
    packageOf,
    packageProblems,
    parseInput,
    PENDING,
    RECORD_LINE,
    recordBody,
    RECORDER,
    recordsOf,
    run,
    type Tree,
    type TreeEntry,
} from "./spec_approval.ts";
import { CommandError, exists, REPO, ROOT, UsageError } from "./lib/repo.ts";

const C1 = "a".repeat(40);
const C2 = "b".repeat(40);
const C3 = "c".repeat(40);
const MAINTAINER = "maint";
const NUMBER = 7;
const TEST_REPO = "o/r";

// ---- A tree from paths ---------------------------------------------------------------------------

/** A tree from paths to contents; "link:" makes a symbolic link, "submodule:" a submodule, and directories follow. */
function treeOf(files: Record<string, string>): Tree {
    const entries: TreeEntry[] = [];
    const directories = new Set<string>();
    for (const [path, content] of Object.entries(files)) {
        const parts = path.split("/");
        for (let depth = 1; depth < parts.length; depth++) directories.add(parts.slice(0, depth).join("/"));
        if (content.startsWith("link:")) entries.push({ path, mode: "120000", type: "blob", sha: content });
        else if (content.startsWith("submodule:")) entries.push({ path, mode: "160000", type: "commit", sha: content });
        else entries.push({ path, mode: "100644", type: "blob", sha: `blob-${content}` });
    }
    for (const path of directories) entries.push({ path, mode: "040000", type: "tree", sha: `tree-${path}` });
    return { entries, truncated: false };
}

/** The package of one change in this repository's layout, beside unrelated files and an old archive. */
const CHANGE: Record<string, string> = {
    "openspec/changes/add-x/.openspec.yaml": "m1",
    "openspec/changes/add-x/proposal.md": "p1",
    "openspec/changes/add-x/design.md": "d1",
    "openspec/changes/add-x/tasks.md": "t1",
    "openspec/changes/add-x/specs/core/x/spec.md": "s1",
    "openspec/changes/archive/2026-01-01-old/proposal.md": "po",
    "openspec/specs/core/x/spec.md": "mx1",
    "openspec/specs/core/y/spec.md": "my1",
    "skills/core/x/SKILL.md": "k1",
    "README.md": "r1",
};

const without = (files: Record<string, string>, ...paths: string[]) =>
    Object.fromEntries(Object.entries(files).filter(([path]) => !paths.includes(path)));

/** CHANGE after `just spec-changes archive`: moved under the archive, tasks ticked, main spec rewritten. */
function archivedChange(overrides: Record<string, string> = {}): Record<string, string> {
    const moved: Record<string, string> = {};
    for (const [path, content] of Object.entries(CHANGE)) {
        if (path.startsWith("openspec/changes/add-x/")) {
            moved[path.replace("openspec/changes/add-x/", "openspec/changes/archive/2026-10-20-add-x/")] = content;
        } else moved[path] = content;
    }
    return {
        ...moved,
        "openspec/changes/archive/2026-10-20-add-x/tasks.md": "t-done",
        "openspec/specs/core/x/spec.md": "mx-merged",
        ...overrides,
    };
}

// ---- A stub of gh that answers like the GitHub REST API ------------------------------------------------

interface RawActor {
    login: string;
    type: string;
}

interface RawEvent {
    id: number;
    event: string;
    label?: { name: string };
    actor: RawActor | null;
}

interface RawComment {
    id: number;
    body: string | null;
    user: RawActor | null;
}

interface World {
    pull: { state: "open" | "closed"; head: string; labels: string[] };
    trees: Record<string, Tree>;
    events: RawEvent[];
    comments: RawComment[];
    permissions: Record<string, string>;
    /** The pull request file list; `changedFiles`, when set, is the count the pull request reports instead of its length. */
    files?: { filename: string; status: string }[];
    changedFiles?: number;
    /** Called before a read is answered with its endpoint and how often it was read before; may change the world. */
    before?: (endpoint: string, count: number, world: World) => void;
    /** A call whose arguments include this text fails as gh fails. */
    failing?: string;
    /** Answers that replace the stub's for an endpoint. */
    raw?: Record<string, string>;
}

const person = (login = MAINTAINER): RawActor => ({ login, type: "User" });
const BOT: RawActor = { ...RECORDER };
const labeled = (id: number, label = LABEL, actor = person()): RawEvent => ({
    id,
    event: "labeled",
    label: { name: label },
    actor,
});
const unlabeled = (id: number, label = LABEL, actor = person()): RawEvent => ({
    id,
    event: "unlabeled",
    label: { name: label },
    actor,
});
const record = (id: number, commit: string, eventId: number, withdrawn = false, user: RawActor | null = BOT) => ({
    id,
    body: recordBody(commit, MAINTAINER, eventId, withdrawn),
    user,
});

/** A gh that serves `world` on repository o/r, records every write in `calls`, and applies it to the world. */
function fakeGh(world: World, repo = TEST_REPO) {
    const calls: string[] = [];
    const reads = new Map<string, number>();
    let nextComment = 1000;
    let nextEvent = 5000;
    const prefix = `repos/${repo}/`;
    const fail = (args: string[], message: string): never => {
        throw new CommandError(`gh ${args.join(" ")} exited 1: ${message}`);
    };
    const answer = (args: string[], stdin?: string): string => {
        if (world.failing !== undefined && args.join(" ").includes(world.failing)) {
            fail(args, "gh: Server Error (HTTP 502)");
        }
        assertEquals(args[0], "api");
        if (args[1] === "-X") {
            const [, , method, path] = args;
            const body = stdin === undefined ? undefined : JSON.parse(stdin);
            if (method !== "DELETE") assertEquals(args.slice(4), ["--input", "-"]);
            if (method === "POST" && path === `${prefix}issues/${NUMBER}/labels`) {
                calls.push(`add ${body.labels.join(",")}`);
                for (const name of body.labels) {
                    if (!world.pull.labels.includes(name)) world.pull.labels.push(name);
                    world.events.push(labeled(nextEvent++, name, BOT));
                }
                return "[]";
            }
            if (method === "DELETE" && path.startsWith(`${prefix}issues/${NUMBER}/labels/`)) {
                const name = decodeURIComponent(path.slice(`${prefix}issues/${NUMBER}/labels/`.length));
                calls.push(`remove ${name}`);
                if (!world.pull.labels.includes(name)) fail(args, "gh: Label does not exist (HTTP 404)");
                world.pull.labels = world.pull.labels.filter((label) => label !== name);
                world.events.push(unlabeled(nextEvent++, name, BOT));
                return "[]";
            }
            if (method === "POST" && path === `${prefix}issues/${NUMBER}/comments`) {
                calls.push("create comment");
                world.comments.push({ id: nextComment++, body: body.body, user: { ...BOT } });
                return "{}";
            }
            const patched = new RegExp(`^${prefix}issues/comments/(\\d+)$`).exec(path);
            if (method === "PATCH" && patched) {
                calls.push(`update comment ${patched[1]}`);
                const comment = world.comments.find((item) => item.id === Number(patched[1]));
                if (!comment) fail(args, "gh: Not Found (HTTP 404)");
                comment!.body = body.body;
                return "{}";
            }
            throw new Error(`the stub has no route for ${args.join(" ")}`);
        }
        const paginated = args[1] === "--paginate";
        if (paginated) assertEquals(args.slice(1, 3), ["--paginate", "--slurp"]);
        const path = args.at(-1)!;
        assertEquals(args.length, paginated ? 4 : 2, args.join(" "));
        const endpoint = path.startsWith(prefix) ? path.slice(prefix.length) : path;
        const count = reads.get(endpoint) ?? 0;
        reads.set(endpoint, count + 1);
        world.before?.(endpoint, count, world);
        if (world.raw?.[endpoint] !== undefined) return world.raw[endpoint];
        if (endpoint === `pulls/${NUMBER}`) {
            const { state, head, labels } = world.pull;
            return JSON.stringify({
                state,
                head: { sha: head },
                labels: labels.map((name) => ({ name })),
                changed_files: world.changedFiles ?? world.files?.length ?? 0,
            });
        }
        if (endpoint === `pulls/${NUMBER}/files?per_page=100`) return JSON.stringify([world.files ?? []]);
        const tree = /^git\/trees\/([0-9a-f]{40})\?recursive=1$/.exec(endpoint);
        if (tree) {
            const found = world.trees[tree[1]];
            if (!found) fail(args, "gh: Not Found (HTTP 404)");
            return JSON.stringify({ sha: tree[1], tree: found.entries, truncated: found.truncated });
        }
        if (endpoint === `issues/${NUMBER}/events?per_page=100`) {
            // Two pages, and an event that is no label event, as the API gives them.
            const all = [{ id: 1, event: "closed", actor: person() }, ...world.events];
            return JSON.stringify([all.slice(0, 2), all.slice(2)]);
        }
        if (endpoint === `issues/${NUMBER}/comments?per_page=100`) return JSON.stringify([world.comments]);
        const permission = /^collaborators\/([^/]+)\/permission$/.exec(endpoint);
        if (permission) {
            return JSON.stringify({ permission: world.permissions[decodeURIComponent(permission[1])] ?? "none" });
        }
        throw new Error(`the stub has no route for ${args.join(" ")}`);
    };
    const gh: Gh = (args, stdin) => {
        try {
            return Promise.resolve(answer(args, stdin));
        } catch (error) {
            return Promise.reject(error);
        }
    };
    return { gh, calls, reads };
}

/** An open pull request whose record of C1 answers the maintainer's adding (event 10) and whose head is C2. */
function approvedWorld(head: Record<string, string> = CHANGE): World {
    return {
        pull: { state: "open", head: C2, labels: [LABEL, "documentation"] },
        trees: { [C1]: treeOf(CHANGE), [C2]: treeOf(head) },
        events: [labeled(5, "documentation", BOT), labeled(10)],
        comments: [{ id: 2, body: "Looks good.", user: person() }, record(1, C1, 10)],
        permissions: { [MAINTAINER]: "write" },
    };
}

/** The moment the maintainer added spec/approved at head C2 (event 10), before any record. */
function recordingWorld(): World {
    return {
        pull: { state: "open", head: C2, labels: ["documentation", LABEL] },
        trees: { [C2]: treeOf(CHANGE) },
        events: [labeled(5, "documentation", BOT), labeled(10)],
        comments: [],
        permissions: { [MAINTAINER]: "admin" },
    };
}

const APPROVAL_EVENT: Event = { action: "labeled", label: LABEL, sender: MAINTAINER, head: C2 };
const PUSH_EVENT: Event = { action: "synchronize", label: "", sender: MAINTAINER, head: C2 };

async function ran(world: World, options: { event?: Event; apply?: boolean } = {}) {
    const { gh, calls, reads } = fakeGh(world);
    const printed: string[] = [];
    const notes: string[] = [];
    const status = await run(
        { number: NUMBER, apply: options.apply ?? true, event: options.event },
        ghApi(gh, TEST_REPO),
        { print: (line) => printed.push(line), note: (line) => notes.push(line) },
    );
    const records = recordsOf(world.comments.map((comment) => ({
        id: comment.id,
        body: comment.body ?? "",
        author: comment.user,
    })));
    return { status, printed, notes, calls, reads, world, records };
}

const labelsOf = (world: World) => world.pull.labels.includes(LABEL);

// ---- The approval package --------------------------------------------------------------------------

Deno.test("packageOf lists each unarchived change but its tasks.md, with the main spec of each delta", () => {
    const found = packageOf(treeOf({ ...CHANGE, "openspec/changes/add-z/specs/sdd/z/spec.md": "sz" }));
    assertEquals([...found.changes.keys()], ["add-x", "add-z"]);
    assertEquals([...found.changes.get("add-x")!.entries()].sort(), [
        [".openspec.yaml", "100644 blob-m1"],
        ["design.md", "100644 blob-d1"],
        ["proposal.md", "100644 blob-p1"],
        ["specs/core/x/spec.md", "100644 blob-s1"],
    ]);
    // A delta for a capability without a main spec records its absence.
    assertEquals([...found.specs.entries()].sort(), [
        ["openspec/specs/core/x/spec.md", "100644 blob-mx1"],
        ["openspec/specs/sdd/z/spec.md", "absent"],
    ]);
    assertEquals(found.refused, []);
    // The archive and a plain file directly under openspec/changes/ are no change.
    assertEquals(packageOf(treeOf({ ...archivedChange(), "openspec/changes/README.md": "r" })).changes.size, 0);
});

Deno.test("changesOf names every unarchived change with its kind and skips the archive and plain files", () => {
    const tree = treeOf({
        ...CHANGE,
        "openspec/changes/linked": "link:elsewhere",
        "openspec/changes/sub": "submodule:deadbeef",
        "openspec/changes/README.md": "r",
    });
    assertEquals(changesOf(tree), [
        { name: "add-x", kind: "directory" },
        { name: "linked", kind: "link" },
        { name: "sub", kind: "submodule" },
    ]);
});

Deno.test("an unrelated change, a tasks.md edit, or a main spec outside the deltas is no package change", () => {
    const same: Record<string, Record<string, string>> = {
        "files outside openspec/": {
            ...CHANGE,
            "README.md": "r2",
            "skills/core/x/SKILL.md": "k2",
            "scripts/a.ts": "a",
        },
        "the change's tasks.md": { ...CHANGE, "openspec/changes/add-x/tasks.md": "t9" },
        "a main spec no delta names": { ...CHANGE, "openspec/specs/core/y/spec.md": "my2" },
        "an older archive": { ...CHANGE, "openspec/changes/archive/2026-01-01-old/proposal.md": "po2" },
    };
    for (const [name, head] of Object.entries(same)) {
        assertEquals(packageProblems(treeOf(CHANGE), treeOf(head)), [], name);
    }
});

Deno.test("every path of the change but tasks.md, and the main spec under each delta, is the package", () => {
    const changed: Record<string, Record<string, string>> = {
        "the proposal": { ...CHANGE, "openspec/changes/add-x/proposal.md": "p2" },
        "the design": { ...CHANGE, "openspec/changes/add-x/design.md": "d2" },
        ".openspec.yaml": { ...CHANGE, "openspec/changes/add-x/.openspec.yaml": "m2" },
        "a delta": { ...CHANGE, "openspec/changes/add-x/specs/core/x/spec.md": "s2" },
        "a main spec under a delta": { ...CHANGE, "openspec/specs/core/x/spec.md": "mx2" },
        "a second change added": { ...CHANGE, "openspec/changes/add-y/proposal.md": "p" },
        "a non-design file added": { ...CHANGE, "openspec/changes/add-x/notes.md": "n" },
        "a file named tasks.md deeper in the change": {
            ...CHANGE,
            "openspec/changes/add-x/specs/core/x/tasks.md": "t",
        },
        "a removed file": without(CHANGE, "openspec/changes/add-x/design.md"),
        "the change removed": without(
            CHANGE,
            ...Object.keys(CHANGE).filter((path) => path.startsWith("openspec/changes/add-x/")),
        ),
    };
    for (const [name, head] of Object.entries(changed)) {
        assert(packageProblems(treeOf(CHANGE), treeOf(head)).length > 0, name);
    }
    // The mode counts as well as the blob.
    const executable = treeOf(CHANGE);
    executable.entries.find((entry) => entry.path.endsWith("add-x/design.md"))!.mode = "100755";
    assertMatch(packageProblems(treeOf(CHANGE), executable).join(), /"add-x" differs in "design.md"/);
});

Deno.test("an archive is no package change; an archive that changes the package is", () => {
    assertEquals(packageProblems(treeOf(CHANGE), treeOf(archivedChange())), []);
    assertEquals(archivedCopy(treeOf(archivedChange()), "add-x")?.files.size, 4);
    const edited = archivedChange({ "openspec/changes/archive/2026-10-20-add-x/proposal.md": "p2" });
    assertMatch(packageProblems(treeOf(CHANGE), treeOf(edited)).join(), /archived with "proposal.md" changed/);
    // Only an archive of the same name answers; the latest date wins.
    const renamed = Object.fromEntries(
        Object.entries(archivedChange()).map(([path, content]) => [path.replace("-add-x/", "-add-x2/"), content]),
    );
    assertMatch(packageProblems(treeOf(CHANGE), treeOf(renamed)).join(), /"add-x" is gone/);
    const twice = { ...archivedChange(), "openspec/changes/archive/2026-11-01-add-x/proposal.md": "p3" };
    assert(packageProblems(treeOf(CHANGE), treeOf(twice)).length > 0, "the latest archive is compared");
    const linked = archivedChange({ "openspec/changes/archive/2026-10-20-add-x/design.md": "link:/etc/passwd" });
    assertMatch(packageProblems(treeOf(CHANGE), treeOf(linked)).join(), /link or submodule in the archive/);
});

Deno.test("a link or a submodule anywhere in the package refuses it", () => {
    const unsafe: Record<string, Record<string, string>> = {
        "a linked change": { ...without(CHANGE, ...Object.keys(CHANGE)), "openspec/changes/add-x": "link:../x" },
        "a submodule change": { "openspec/changes/add-x": "submodule:0000" },
        "a link inside the change": { ...CHANGE, "openspec/changes/add-x/design.md": "link:/etc/passwd" },
        "a linked main spec": { ...CHANGE, "openspec/specs/core/x/spec.md": "link:../y/spec.md" },
        "a linked directory above a main spec": {
            ...without(CHANGE, "openspec/specs/core/x/spec.md"),
            "openspec/specs/core/x": "link:y",
        },
    };
    for (const [name, files] of Object.entries(unsafe)) {
        assert(packageOf(treeOf(files)).refused.length > 0, name);
        assert(packageProblems(treeOf(CHANGE), treeOf(files)).length > 0, name);
    }
});

// ---- Recording -------------------------------------------------------------------------------------

Deno.test("the maintainer's account adds the label at the current head, and a record is written", async () => {
    const result = await ran(recordingWorld(), { event: APPROVAL_EVENT });
    assertEquals(result.status, 0);
    assertEquals(result.calls, ["create comment"]);
    assertEquals(result.records, [{ commentId: 1000, commit: C2, login: MAINTAINER, eventId: 10, withdrawn: false }]);
    assert(labelsOf(result.world));
    assertEquals(result.printed, [
        `spec/approved: this run records the approval of ${C2} by "maint" (label event 10)`,
        `write: post the record of ${C2}`,
    ]);
    // The next run, from any event, keeps it and writes nothing; the read-only run says approved.
    const next = await ran(result.world, { event: PUSH_EVENT });
    assertEquals([next.status, next.calls], [0, []]);
    assertMatch(next.printed[0], /^spec\/approved: the approval of b{40} by "maint" \(label event 10\) holds$/);
    const read = await ran(result.world, { apply: false });
    assertEquals([read.status, read.calls], [0, []]);
    // A rerun of the recording run replaces the one record with the same text.
    const rerun = await ran(result.world, { event: APPROVAL_EVENT });
    assertEquals(rerun.calls, ["update comment 1000"]);
    assertEquals(rerun.records.length, 1);
});

Deno.test("the recording run replaces the one withdrawn record and puts back a label a run took off", async () => {
    const world = recordingWorld();
    world.pull.labels = ["documentation"];
    world.events.push(unlabeled(11, LABEL, BOT));
    world.comments = [record(3, C1, 2, true)];
    const result = await ran(world, { event: APPROVAL_EVENT });
    assertEquals(result.calls, ["update comment 3", `add ${LABEL}`]);
    assertEquals(result.records, [{ commentId: 3, commit: C2, login: MAINTAINER, eventId: 10, withdrawn: false }]);
});

Deno.test("recording is refused, and the label withdrawn, when the approval cannot be stood behind", async () => {
    const refusals: Record<string, [(world: World) => void, RegExp]> = {
        "permission below write": [(world) => world.permissions = { [MAINTAINER]: "read" }, /permission is "read"/],
        "no permission at all": [(world) => world.permissions = {}, /permission is "none"/],
        // The same login as the sender, so that only the account's type decides.
        "added by a bot": [
            (world) => world.events = [labeled(10, LABEL, { login: MAINTAINER, type: "Bot" })],
            /not this/,
        ],
        "another person's adding is the latest": [
            (world) => world.events.push(labeled(11, LABEL, person("other"))),
            /not this/,
        ],
        "the head moved between the event and the run": [(world) => {
            world.pull.head = C3;
            world.trees[C3] = treeOf(CHANGE);
        }, /head moved after the label was added/],
        "the head moved during the run": [(world) => {
            world.before = (endpoint, count, now) => {
                if (endpoint === `pulls/${NUMBER}` && count === 1) now.pull.head = C3;
            };
        }, /head moved during the run/],
        "a linked change entry": [
            (world) => world.trees[C2] = treeOf({ ...archivedChange(), "openspec/changes/add-x": "link:/etc" }),
            /link or submodule/,
        ],
        "a submodule change entry": [
            (world) => world.trees[C2] = treeOf({ ...archivedChange(), "openspec/changes/add-x": "submodule:0" }),
            /link or submodule/,
        ],
        "a link inside the change": [
            (world) => world.trees[C2] = treeOf({ ...CHANGE, "openspec/changes/add-x/specs/core/x/spec.md": "link:x" }),
            /link or submodule/,
        ],
        "more than one record comment": [
            (world) => world.comments = [record(1, C1, 2, true), record(2, C1, 3, true)],
            /more than one record comment/,
        ],
    };
    const events: Record<string, Event> = {
        "the event names no usable commit": { ...APPROVAL_EVENT, head: "HEAD" },
        "the event names no usable account": { ...APPROVAL_EVENT, sender: "" },
    };
    for (const name of Object.keys(events)) refusals[name] = [() => {}, /no usable account or commit/];
    for (const [name, [change, why]] of Object.entries(refusals)) {
        const world = recordingWorld();
        change(world);
        const result = await ran(world, { event: events[name] ?? APPROVAL_EVENT });
        assertEquals(result.status, 0, name);
        assert(!result.calls.includes("create comment"), `${name}: no record`);
        assertMatch(result.printed[0], /^spec\/pending: .*; not recorded: /, name);
        assertMatch(result.printed[0], why, name);
        assert(!labelsOf(result.world), `${name}: the label is withdrawn`);
        assert(result.records.every((item) => item.withdrawn), `${name}: no live record`);
    }
});

Deno.test("a person re-added or removed the label during the run: nothing is recorded", async () => {
    // Removed while this run read: the label is gone, so the removal answers 404, which is what it wants.
    const removed = recordingWorld();
    removed.before = (endpoint, count, world) => {
        if (endpoint === `issues/${NUMBER}/events?per_page=100` && count === 1) {
            world.events.push(unlabeled(11));
            world.pull.labels = world.pull.labels.filter((label) => label !== LABEL);
        }
    };
    const gone = await ran(removed, { event: APPROVAL_EVENT });
    assertEquals(gone.status, 0);
    assertEquals(gone.calls, [`add ${PENDING}`, `remove ${LABEL}`]);
    assertMatch(gone.printed[0], /not recorded: a person withdrew spec\/approved during the run/);
    assert(gone.notes.includes(`done: ${LABEL} was already gone`), gone.notes.join("|"));
    // Removed and added again while this run read: this run records nothing; the run of the new adding does.
    const again = recordingWorld();
    again.before = (endpoint, count, world) => {
        if (endpoint === `issues/${NUMBER}/events?per_page=100` && count === 1) {
            world.events.push(unlabeled(11), labeled(12));
        }
    };
    const first = await ran(again, { event: APPROVAL_EVENT });
    assert(!first.calls.includes("create comment"));
    assertMatch(first.printed[0], /spec\/approved was added again during the run/);
    again.before = undefined;
    // The first run's writes are the workflow's own, which withdraw nothing a person did.
    const second = await ran(first.world, { event: APPROVAL_EVENT });
    assertEquals(second.calls, ["create comment", `add ${LABEL}`, `remove ${PENDING}`]);
    assertEquals(second.records.map((item) => item.eventId), [12]);
    // The same events by a bot stop no record, nor does spec/pending added by the workflow.
    const bots = recordingWorld();
    bots.events.push(unlabeled(11, LABEL, BOT), labeled(12, PENDING, BOT));
    assertEquals((await ran(bots, { event: APPROVAL_EVENT })).calls, ["create comment"]);
    // spec/pending added by a person after the adding withdraws it, during the run or before.
    const pendingDuring = recordingWorld();
    pendingDuring.before = (endpoint, count, world) => {
        if (endpoint === `issues/${NUMBER}/events?per_page=100` && count === 1) {
            world.events.push(labeled(11, PENDING));
            world.pull.labels.push(PENDING);
        }
    };
    const during = await ran(pendingDuring, { event: APPROVAL_EVENT });
    assert(!during.calls.includes("create comment"));
    assertMatch(during.printed[0], /not recorded: a person withdrew spec\/approved during the run/);
    const pendingBefore = recordingWorld();
    pendingBefore.events.push(labeled(11, PENDING));
    const before = await ran(pendingBefore, { event: APPROVAL_EVENT });
    assertEquals(before.calls, [`add ${PENDING}`, `remove ${LABEL}`]);
    assertMatch(before.printed[0], /not recorded: a person withdrew spec\/approved after this event/);
});

Deno.test("a closed pull request is never written to by a run that succeeds", async () => {
    const closed = recordingWorld();
    closed.pull.state = "closed";
    const result = await ran(closed, { event: APPROVAL_EVENT });
    assertEquals([result.status, result.calls], [0, []]);
    assertEquals(result.printed.slice(1), ["the pull request is closed: nothing is written"]);
    assertMatch(result.printed[0], /not recorded: the pull request is closed/);
    // Closed while the run read: nothing is recorded or written.
    const closing = recordingWorld();
    closing.before = (endpoint, count, world) => {
        if (endpoint === `pulls/${NUMBER}` && count === 1) world.pull.state = "closed";
    };
    const meanwhile = await ran(closing, { event: APPROVAL_EVENT });
    assertEquals([meanwhile.status, meanwhile.calls], [0, []]);
    assertMatch(meanwhile.printed[0], /closed during the run/);
    // A stale label on a closed pull request stays.
    const stale = approvedWorld({ ...CHANGE, "openspec/changes/add-x/design.md": "d2" });
    stale.pull.state = "closed";
    assertEquals((await ran(stale, { event: PUSH_EVENT })).calls, []);
});

// ---- Every other run -------------------------------------------------------------------------------

Deno.test("after the record: a package change withdraws; an unrelated change or ticked tasks keep it", async () => {
    const changed = await ran(approvedWorld({ ...CHANGE, "openspec/changes/add-x/proposal.md": "p2" }), {
        event: PUSH_EVENT,
    });
    assertEquals(changed.status, 0);
    assertEquals(changed.calls, ["update comment 1", `add ${PENDING}`, `remove ${LABEL}`]);
    assertMatch(changed.printed[0], /^spec\/pending: the change "add-x" differs in "proposal.md"$/);
    assertEquals(changed.records.map((item) => item.withdrawn), [true]);
    assertEquals(changed.printed.slice(1), [
        `write: mark the record of ${C1} withdrawn`,
        `write: add ${PENDING}`,
        `write: remove ${LABEL}`,
    ]);
    assertEquals(changed.world.pull.labels.filter((name) => name.startsWith("spec/")), [PENDING]);
    // The maintainer approves the new package by adding the label again; the next run then records it.
    changed.world.events.push(labeled(30));
    changed.world.pull.labels.push(LABEL);
    const again = await ran(changed.world, { event: APPROVAL_EVENT });
    assertEquals(again.calls, ["update comment 1", `remove ${PENDING}`]);
    assertEquals(again.records.map((item) => [item.commit, item.eventId, item.withdrawn]), [[C2, 30, false]]);
    for (
        const [name, head] of Object.entries({
            "an unrelated change": { ...CHANGE, "README.md": "r2", "openspec/changes/add-x/tasks.md": "t2" },
            "a second commit that only ticks tasks": { ...CHANGE, "openspec/changes/add-x/tasks.md": "t-done" },
        })
    ) {
        const kept = await ran(approvedWorld(head), { event: PUSH_EVENT });
        assertEquals([kept.status, kept.calls], [0, []], name);
        assertMatch(kept.printed[0], /^spec\/approved: /, name);
        assert(kept.notes.includes("done: nothing to write"), name);
    }
});

Deno.test(".openspec.yaml, a main spec under a delta, a second change, or a new file withdraws it", async () => {
    const changes: Record<string, Record<string, string>> = {
        ".openspec.yaml changed": { ...CHANGE, "openspec/changes/add-x/.openspec.yaml": "m2" },
        "a main spec under a delta changed": { ...CHANGE, "openspec/specs/core/x/spec.md": "mx2" },
        "a second change added": { ...CHANGE, "openspec/changes/add-y/proposal.md": "py" },
        "a non-design file added": { ...CHANGE, "openspec/changes/add-x/notes.md": "n" },
    };
    for (const [name, head] of Object.entries(changes)) {
        const result = await ran(approvedWorld(head), { event: PUSH_EVENT });
        assertEquals(result.calls, ["update comment 1", `add ${PENDING}`, `remove ${LABEL}`], name);
        assert(!labelsOf(result.world), name);
    }
});

Deno.test("two records, a record by another account, or a record of an earlier adding withdraws it", async () => {
    const cases: Record<string, [(world: World) => void, string[]]> = {
        "two records": [
            (world) => world.comments.push(record(3, C1, 10)),
            ["update comment 1", "update comment 3", `add ${PENDING}`, `remove ${LABEL}`],
        ],
        "a record comment by a person": [(world) => world.comments = [record(1, C1, 10, false, person())], [
            `add ${PENDING}`,
            `remove ${LABEL}`,
        ]],
        "a record comment by a person with the bot's login": [
            (world) => world.comments = [record(1, C1, 10, false, { login: RECORDER.login, type: "User" })],
            [`add ${PENDING}`, `remove ${LABEL}`],
        ],
        "a record comment by another bot": [
            (world) => world.comments = [record(1, C1, 10, false, { login: "other[bot]", type: "Bot" })],
            [`add ${PENDING}`, `remove ${LABEL}`],
        ],
        "a record answering an earlier adding": [
            (world) => world.events.push(unlabeled(20), labeled(21)),
            ["update comment 1", `add ${PENDING}`, `remove ${LABEL}`],
        ],
        "a record whose first line lost its form": [
            (world) => world.comments = [{ id: 1, body: ` ${recordBody(C1, MAINTAINER, 10, false)}`, user: BOT }],
            [`add ${PENDING}`, `remove ${LABEL}`],
        ],
        "a withdrawn record": [(world) => world.comments = [record(1, C1, 10, true)], [
            `add ${PENDING}`,
            `remove ${LABEL}`,
        ]],
    };
    for (const [name, [change, calls]] of Object.entries(cases)) {
        const world = approvedWorld();
        change(world);
        const result = await ran(world, { event: PUSH_EVENT });
        assertEquals([result.status, result.calls], [0, calls], name);
        assertMatch(result.printed[0], /^spec\/pending: /, name);
    }
});

Deno.test("the label present with no record (the recording run was cancelled) is withdrawn", async () => {
    const world = recordingWorld();
    const result = await ran(world, { event: PUSH_EVENT });
    assertEquals([result.status, result.calls], [0, [`add ${PENDING}`, `remove ${LABEL}`]]);
    assertMatch(result.printed[0], /^spec\/pending: no record comment$/);
    // Another label added by a person is not this event either.
    const other = await ran(recordingWorld(), { event: { ...APPROVAL_EVENT, label: "documentation" } });
    assertEquals(other.calls, [`add ${PENDING}`, `remove ${LABEL}`]);
    // The read-only run reports it and writes nothing.
    const read = await ran(recordingWorld(), { apply: false });
    assertEquals([read.status, read.calls], [1, []]);
    assertEquals(read.printed, ["spec/pending: no record comment", `write: add ${PENDING}`, `write: remove ${LABEL}`]);
});

Deno.test("a person's removal marks the record withdrawn; no label and no record writes nothing", async () => {
    const world = approvedWorld();
    world.events.push(unlabeled(11));
    world.pull.labels = ["documentation"];
    const result = await ran(world, { event: { ...PUSH_EVENT, action: "unlabeled", label: LABEL } });
    assertEquals(result.calls, ["update comment 1", `add ${PENDING}`]);
    assertMatch(result.printed[0], /spec\/approved is not on the pull request; a person withdrew spec\/approved/);
    // An unarchived change with no approval at all is pending; once pending, nothing more is written.
    const fresh = await ran({ ...recordingWorld(), events: [], pull: { state: "open", head: C2, labels: [] } });
    assertEquals([fresh.status, fresh.calls], [0, [`add ${PENDING}`]]);
    const nothing = await ran(fresh.world);
    assertEquals([nothing.status, nothing.calls], [0, []]);
    const read = await ran(fresh.world, { apply: false });
    assertEquals([read.status, read.printed.length], [1, 1]);
});

Deno.test("a link in the package at the head withdraws a recorded approval", async () => {
    const result = await ran(
        approvedWorld({ ...CHANGE, "openspec/changes/add-x/extra.md": "link:/etc/passwd" }),
        { event: PUSH_EVENT },
    );
    assertEquals(result.calls, ["update comment 1", `add ${PENDING}`, `remove ${LABEL}`]);
    assertMatch(
        result.printed[0],
        /link or submodule in the package at the head: "openspec\/changes\/add-x\/extra.md"/,
    );
});

// ---- Failures --------------------------------------------------------------------------------------

Deno.test("a failed run removes the label and exits 2; a read-only run writes nothing", async () => {
    for (const failing of [`pulls/${NUMBER}`, "/events", "/comments", "git/trees", "/permission"]) {
        const world = failing === "/permission" ? recordingWorld() : approvedWorld();
        world.failing = failing;
        const event = failing === "/permission" ? APPROVAL_EVENT : PUSH_EVENT;
        const result = await ran(world, { event });
        assertEquals(result.status, 2, failing);
        // An open pull request gets spec/pending; when the pull request itself could not be read, only the removal.
        const known = failing !== `pulls/${NUMBER}`;
        assertEquals(result.calls, [...known ? [`add ${PENDING}`] : [], `remove ${LABEL}`], failing);
        assert(!labelsOf(result.world), failing);
        assertMatch(result.notes[0], /^error: gh api .*HTTP 502/, failing);
        assertEquals(result.notes.at(-1), `after the error: removed ${LABEL}`, failing);
        // A read-only run reads no permission; every other read fails it without a write.
        if (failing !== "/permission") {
            const read = await ran({ ...approvedWorld(), failing }, { apply: false });
            assertEquals([read.status, read.calls], [2, []], failing);
        }
    }
    // A write that fails: the label is still taken off.
    const write = approvedWorld({ ...CHANGE, "openspec/changes/add-x/design.md": "d2" });
    write.failing = "PATCH";
    const result = await ran(write, { event: PUSH_EVENT });
    assertEquals([result.status, result.calls], [2, [`add ${PENDING}`, `remove ${LABEL}`]]);
    // Taking it off after an error tolerates its absence, and reports a failure of its own without hiding the first.
    const absent = recordingWorld();
    absent.pull.labels = [];
    absent.failing = "/comments";
    assertEquals((await ran(absent)).notes.at(-1), `after the error: ${LABEL} was not on the pull request`);
    const both = approvedWorld();
    both.failing = "issues/7";
    const failed = await ran(both);
    assertEquals(failed.status, 2);
    assertMatch(failed.notes[1], /^after the error: could not add spec\/pending: gh api -X POST/);
    assertMatch(failed.notes[2], /^after the error: could not remove spec\/approved: gh api -X DELETE/);
    // A closed pull request gets no spec/pending after an error.
    const closed = approvedWorld();
    closed.pull.state = "closed";
    closed.failing = "/comments";
    assertEquals((await ran(closed)).calls, [`remove ${LABEL}`]);
});

Deno.test("a truncated tree, an answer that is no JSON, or one of an unexpected shape fails the run", async () => {
    const truncated = approvedWorld();
    truncated.trees[C1] = { ...truncated.trees[C1], truncated: true };
    const cut = await ran(truncated, { event: PUSH_EVENT });
    assertEquals([cut.status, cut.calls], [2, [`add ${PENDING}`, `remove ${LABEL}`]]);
    assertMatch(cut.notes[0], /truncated/);
    const answers: Record<string, string> = {
        [`pulls/${NUMBER}`]: "<html>",
        [`issues/${NUMBER}/comments?per_page=100`]: JSON.stringify({ message: "Not Found" }),
        [`issues/${NUMBER}/events?per_page=100`]: JSON.stringify([[{ id: 3, event: "labeled", actor: null }]]),
    };
    for (const [endpoint, raw] of Object.entries(answers)) {
        const world = approvedWorld();
        world.raw = { [endpoint]: raw };
        const result = await ran(world, { apply: false });
        assertEquals(result.status, 2, endpoint);
        assertMatch(result.notes[0], /did not answer JSON|unexpected shape/, endpoint);
    }
    const headless = approvedWorld();
    headless.raw = { [`pulls/${NUMBER}`]: JSON.stringify({ state: "open", head: {}, labels: [] }) };
    assertMatch(
        (await ran(headless, { apply: false })).notes[0],
        /gh api repos\/o\/r\/pulls\/7 has an unexpected shape/,
    );
});

Deno.test("an error that is no interface failure still takes the label off, then crashes", async () => {
    const world = approvedWorld();
    const { gh, calls } = fakeGh(world);
    let first = true;
    const broken: Gh = (args, stdin) => {
        if (first) {
            first = false;
            return Promise.reject(new TypeError("a bug"));
        }
        return gh(args, stdin);
    };
    const notes: string[] = [];
    await assertRejects(
        () =>
            run({ number: NUMBER, apply: true }, ghApi(broken, TEST_REPO), {
                print: () => {},
                note: (line) => notes.push(line),
            }),
        TypeError,
        "a bug",
    );
    // The pull request was never read, so only the removal.
    assertEquals(calls, [`remove ${LABEL}`]);
    assertEquals(notes, ["error: a bug", `after the error: removed ${LABEL}`]);
});

// ---- The state without an unarchived change --------------------------------------------------------

Deno.test("no unarchived change: archived when the pull request archives one, else none; no record is touched", async () => {
    const archiving = approvedWorld(archivedChange());
    archiving.trees[C2] = treeOf(archivedChange());
    archiving.files = [
        { filename: "openspec/changes/archive/2026-10-20-add-x/proposal.md", status: "renamed" },
        { filename: "openspec/changes/archive/2026-10-20-add-x/tasks.md", status: "added" },
        { filename: "openspec/changes/add-x/tasks.md", status: "removed" },
    ];
    // The freeze: spec/archived replaces spec/approved, and the record stays live as the evidence.
    const frozen = await ran(archiving, { event: PUSH_EVENT });
    assertEquals([frozen.status, frozen.calls], [0, [`add ${ARCHIVED}`, `remove ${LABEL}`]]);
    assertEquals(frozen.printed[0], `${ARCHIVED}: no unarchived change at the head; the pull request archives one`);
    assertEquals(frozen.records.map((item) => item.withdrawn), [false]);
    const again = await ran(frozen.world, { event: PUSH_EVENT });
    assertEquals(again.calls, []);
    assertEquals((await ran(frozen.world, { apply: false })).status, 1);
    // Unrelated to OpenSpec: no state label, and any one present is removed.
    const unrelated: World = {
        ...recordingWorld(),
        trees: { [C2]: treeOf(without(archivedChange(), "openspec/changes/archive/2026-10-20-add-x/proposal.md")) },
        files: [{ filename: "README.md", status: "modified" }],
    };
    unrelated.pull.labels = ["documentation", PENDING, LABEL];
    const none = await ran(unrelated, { event: APPROVAL_EVENT });
    assertEquals(none.calls, [`remove ${PENDING}`, `remove ${LABEL}`]);
    assertEquals(none.printed[0], "none: no unarchived change at the head, and the pull request archives none");
    assertEquals(none.world.pull.labels, ["documentation"]);
    assert(!none.calls.includes("create comment"));
});

Deno.test("no unarchived change and a cut-short file list: spec/archived is left as it is", async () => {
    for (const [name, changedFiles] of [["more files than listed", 5], ["more than the API lists", 3001]] as const) {
        const world: World = {
            ...recordingWorld(),
            trees: { [C2]: treeOf(archivedChange()) },
            files: [{ filename: "README.md", status: "modified" }],
            changedFiles,
        };
        world.pull.labels = [ARCHIVED, PENDING];
        const result = await ran(world, { event: PUSH_EVENT });
        assertEquals(result.calls, [`remove ${PENDING}`], name);
        assertMatch(result.printed[0], /^undecided: .*cut short/, name);
    }
    // An archive found in the part that was listed still decides it.
    const found: World = {
        ...recordingWorld(),
        trees: { [C2]: treeOf(archivedChange()) },
        files: [{ filename: "openspec/changes/archive/2026-10-20-add-x/design.md", status: "added" }],
        changedFiles: 4000,
    };
    assertEquals((await ran(found, { event: PUSH_EVENT })).calls, [`add ${ARCHIVED}`, `remove ${LABEL}`]);
});

// ---- The record comment ----------------------------------------------------------------------------

Deno.test("recordBody checks each field, and the reader accepts no field the writer would refuse", () => {
    assertThrows(() => recordBody("abc", MAINTAINER, 1, false), Error, "commit id");
    assertThrows(() => recordBody(C1, "../x", 1, false), Error, "login");
    assertThrows(() => recordBody(C1, "-x", 1, false), Error, "login");
    assertThrows(() => recordBody(C1, "x".repeat(40), 1, false), Error, "login");
    assertThrows(() => recordBody(C1, MAINTAINER, 0, false), Error, "label event");
    assertThrows(() => recordBody(C1, MAINTAINER, 1.5, false), Error, "label event");
    assertThrows(() => recordBody(C1, MAINTAINER, 2 ** 60, false), Error, "label event");
    const body = recordBody(C1, "a-b", 33636635955, false);
    assert(body.startsWith(`Approval record: ${C1} a-b 33636635955\nStatus: approved\n\n`), body);
    assert(body.includes(`https://github.com/ryan-minato/skills/commit/${C1}`), body);
    assert(!body.includes("@"), "no mention");
    assertEquals(recordsOf([{ id: 4, body: recordBody(C1, "a-b", 12, true), author: { ...RECORDER } }]), [
        { commentId: 4, commit: C1, login: "a-b", eventId: 12, withdrawn: true },
    ]);
    assertEquals(recordsOf([{ id: 4, body, author: null }]), []);
    for (
        const [login, id] of [["-x", "1"], ["x-", "1"], ["a--b", "1"], ["x".repeat(40), "1"], ["ok", "0"], ["ok", "01"]]
    ) {
        const forged = `Approval record: ${C1} ${login} ${id}\nStatus: approved\n`;
        assertEquals(recordsOf([{ id: 4, body: forged, author: { ...RECORDER } }]), [], forged);
    }
    assert(!RECORD_LINE.test(`Approval record: ${C1} ok ${"9".repeat(17)}`), "beyond a safe integer");
});

Deno.test("the record comment holds nothing a pull request named", async () => {
    const hostile = ["@octocat", "<!-- x -->", "::error::boom", "/approve", "#999", "[x](http://evil.example)", "`c`"];
    const files: Record<string, string> = {};
    for (const [index, name] of hostile.entries()) files[`openspec/changes/${name}-${index}/proposal.md`] = "p";
    const world = recordingWorld();
    world.trees[C2] = treeOf(files);
    const result = await ran(world, { event: APPROVAL_EVENT });
    assertEquals(result.calls, ["create comment"]);
    const body = result.world.comments[0].body!;
    for (const [index, name] of hostile.entries()) assert(!body.includes(`${name}-${index}`), name);
    for (const marker of ["@", "<!--", "::", "#9"]) assert(!body.includes(marker), marker);
    // What the script prints carries such names only inside a JSON string, on one line each.
    world.trees[C3] = treeOf({ ...files, "openspec/changes/::warning::x/proposal.md": "q" });
    world.pull.head = C3;
    const later = await ran(world, { event: PUSH_EVENT });
    assertMatch(later.printed[0], /the change "::warning::x" was added/);
    for (const line of [...later.printed, ...later.notes]) assert(!line.includes("\n"), line);
});

// ---- gh, the command line, and the environment ------------------------------------------------------

Deno.test("ghApi passes what a pull request controls only in a body or percent-encoded in a path", async () => {
    const calls: { args: string[]; stdin?: string }[] = [];
    const gh: Gh = (args, stdin) => {
        calls.push({ args, stdin });
        return Promise.resolve(args.includes("--paginate") ? "[[]]" : JSON.stringify({ permission: "read" }));
    };
    const api = ghApi(gh, TEST_REPO);
    await api.permission("../..");
    assertEquals(calls.at(-1)!.args, ["api", "repos/o/r/collaborators/..%2F../permission"]);
    await api.permission("a b@c");
    assertEquals(calls.at(-1)!.args[1], "repos/o/r/collaborators/a%20b%40c/permission");
    await api.removeLabel(5, LABEL);
    assertEquals(calls.at(-1)!.args, ["api", "-X", "DELETE", "repos/o/r/issues/5/labels/spec%2Fapproved"]);
    await api.addLabel(5, PENDING);
    assertEquals(calls.at(-1)!.args, ["api", "-X", "POST", "repos/o/r/issues/5/labels", "--input", "-"]);
    assertEquals(calls.at(-1)!.stdin, '{"labels":["spec/pending"]}');
    await api.files(5);
    assertEquals(calls.at(-1)!.args, ["api", "--paginate", "--slurp", "repos/o/r/pulls/5/files?per_page=100"]);
    // No label but the three states is ever written.
    for (const name of ["documentation", "spec/frozen", "../x"]) {
        await assertRejects(() => api.addLabel(5, name), Error, "not a state label");
        await assertRejects(() => api.removeLabel(5, name), Error, "not a state label");
    }
    await api.createComment(5, "@file\n::error::x");
    assertEquals(calls.at(-1)!.args, ["api", "-X", "POST", "repos/o/r/issues/5/comments", "--input", "-"]);
    assertEquals(calls.at(-1)!.stdin, '{"body":"@file\\n::error::x"}');
    await api.updateComment(9, "b");
    assertEquals(calls.at(-1)!.args, ["api", "-X", "PATCH", "repos/o/r/issues/comments/9", "--input", "-"]);
    await assertRejects(() => api.tree("../main"), CommandError, "40 hexadecimal digits");
    await assertRejects(() => api.tree("main"), CommandError, "40 hexadecimal digits");
    await assertRejects(() => api.pull(1.5), Error, "digits");
    await assertRejects(() => api.labelEvents(-1), Error, "digits");
    await assertRejects(() => api.updateComment(0, "b"), CommandError, "positive integer");
    for (const call of calls) {
        assert(!call.args.includes("-f") && !call.args.includes("-F"), call.args.join(" "));
        for (const arg of call.args) assert(!arg.includes("\n"), arg);
    }
    // A failure other than 404 on the removal is a failure.
    const failing = ghApi(() => Promise.reject(new CommandError("gh api exited 1: gh: Forbidden (HTTP 403)")), "o/r");
    await assertRejects(() => failing.removeLabel(5, LABEL), CommandError, "HTTP 403");
    const missing = ghApi(() =>
        Promise.reject(new CommandError("gh api exited 1: gh: Label does not exist (HTTP 404)"))
    );
    assertEquals(await missing.removeLabel(5, LABEL), false);
});

Deno.test("parseInput: a number for a read, the environment for --apply, and nothing else", () => {
    const env = (values: Record<string, string>) => (name: string) => values[name];
    const untouched = (name: string): string => {
        throw new Error(`a read-only run read ${name}`);
    };
    assertEquals(parseInput(["117"], untouched), { number: 117, apply: false });
    assertEquals(parseInput(["--help"], untouched), "help");
    assertEquals(parseInput(["-h"], untouched), "help");
    for (const bad of ["0", "012", "1e3", "0x10", "12 ", "1.5", "12345678901", "七", "1;2", "-1"]) {
        assertThrows(() => parseInput([bad], untouched), UsageError, "digits", bad);
        assertThrows(
            () => parseInput(["--apply"], env({ PR_NUMBER: bad, EVENT_ACTION: "opened" })),
            UsageError,
            "PR_NUMBER",
        );
    }
    assertThrows(() => parseInput([], untouched), UsageError, "one pull request number");
    assertThrows(() => parseInput(["1", "2"], untouched), UsageError, "one pull request number");
    assertThrows(() => parseInput(["--force", "1"], untouched), UsageError, "unknown option");
    assertThrows(() => parseInput(["--apply", "117"], untouched), UsageError, "takes no number");
    assertThrows(() => parseInput(["--apply"], env({ PR_NUMBER: "9" })), UsageError, "EVENT_ACTION");
    const event = { PR_NUMBER: "9", EVENT_ACTION: "labeled", EVENT_LABEL: LABEL, EVENT_SENDER: "m", EVENT_HEAD: C1 };
    assertEquals(parseInput(["--apply"], env(event)), {
        number: 9,
        apply: true,
        event: { action: "labeled", label: LABEL, sender: "m", head: C1 },
    });
    assertEquals(Object.keys(event).sort(), [...EVENT_VARIABLES].sort());
});

Deno.test("main prints the usage on --help and exits 2 on a usage error", async () => {
    const printed: string[] = [];
    const notes: string[] = [];
    const out = { print: (line: string) => printed.push(line), note: (line: string) => notes.push(line) };
    const unused: Gh = () => Promise.reject(new Error("no gh call expected"));
    assertEquals(await main(["--help"], { gh: unused, out }), 0);
    assertMatch(printed[0], /^usage: scripts\/spec_approval.ts <pull-request-number>/);
    assertEquals(await main(["x"], { gh: unused, out }), 2);
    assertMatch(notes[0], /^error: the pull request number must be digits, got "x"\n\nusage:/);
    const world = approvedWorld();
    const { gh } = fakeGh(world);
    assertEquals(await main(["--apply"], { gh, out, env: () => undefined }), 2);
    // The workflow's run: the event from the environment, the writes made.
    const recording = recordingWorld();
    const fake = fakeGh(recording, REPO);
    const env = {
        PR_NUMBER: "7",
        EVENT_ACTION: "labeled",
        EVENT_LABEL: LABEL,
        EVENT_SENDER: MAINTAINER,
        EVENT_HEAD: C2,
    };
    const apply = await main(["--apply"], { gh: fake.gh, out, env: (name) => env[name as keyof typeof env] });
    assertEquals([apply, fake.calls], [0, ["create comment"]]);
});

// ---- The script's own bounds -----------------------------------------------------------------------

const read = (path: string) => Deno.readTextFileSync(`${ROOT}/${path}`);

Deno.test("the script imports only local modules that import nothing remote", () => {
    const script = read("scripts/spec_approval.ts");
    const imports = [...script.matchAll(/^import .* from "([^"]+)";$/gm)].map((match) => match[1]);
    assertEquals(imports.sort(), ["./lib/args.ts", "./lib/repo.ts"]);
    assert(!/\bimport\(/.test(script), "no dynamic import");
    for (const module of ["scripts/lib/args.ts", "scripts/lib/repo.ts"]) {
        for (const [, from] of read(module).matchAll(/^import .* from "([^"]+)";$/gm)) {
            assert(from.startsWith("./"), `${module} imports ${from}`);
        }
        assert(!/\bimport\(/.test(read(module)), `${module}: no dynamic import`);
    }
});

Deno.test("the shebang lets the script run gh and read the event variables, and grants nothing else", () => {
    const [first] = read("scripts/spec_approval.ts").split("\n");
    assertEquals(
        first,
        `#!/usr/bin/env -S deno run --no-prompt --allow-run=gh --allow-env=${EVENT_VARIABLES.join(",")}`,
    );
    assert((Deno.statSync(`${ROOT}/scripts/spec_approval.ts`).mode ?? 0) & 0o100, "executable");
});

Deno.test("labels.json declares spec/approved as the maintainer's", () => {
    const labels = JSON.parse(read(".github/labels.json")) as { name: string; applied_by: string }[];
    assertEquals(labels.filter((label) => label.name === LABEL).map((label) => label.applied_by), ["maintainer"]);
});

// ---- The bounds of spec-labels.yml -------------------------------------------------------------------

const WORKFLOW = ".github/workflows/spec-labels.yml";

// deno-lint-ignore no-explicit-any
type Yaml = any;

function workflow(): { text: string; doc: Yaml } {
    const text = read(WORKFLOW);
    return { text, doc: parseYaml(text) as Yaml };
}

Deno.test("spec-labels.yml runs the approval within its bounds", () => {
    const { text, doc } = workflow();
    // `on` is a string key in YAML 1.2, as @std/yaml parses it.
    assertEquals(Object.keys(doc.on), ["pull_request_target"]);
    assertEquals(doc.on.pull_request_target, {
        types: ["opened", "synchronize", "reopened", "ready_for_review", "labeled", "unlabeled"],
    });
    // The top level grants nothing, so a job added later starts with no permission; only the job holds one.
    assertEquals(doc.permissions, {});
    assertEquals(doc.env, undefined);
    assertEquals(doc.concurrency, {
        group: "spec-labels-${{ github.event.pull_request.number }}",
        "cancel-in-progress": false,
    });
    assertEquals(Object.keys(doc.jobs), ["labels"]);
    const job = doc.jobs.labels;
    assertEquals(job.name, "spec / labels");
    assertEquals(job.permissions, { contents: "read", "pull-requests": "write" });
    assert(Number.isInteger(job["timeout-minutes"]) && job["timeout-minutes"] > 0, "the job has a time limit");
    for (const key of ["env", "container", "services", "defaults"]) assertEquals(job[key], undefined, key);
    const steps: Yaml[] = job.steps;
    // The checkout, the tools, the one deciding step, and the failure step: nothing else.
    assertEquals(steps.length, 4);
    const [checkout, tools] = steps;
    // The base, without credentials and without a ref.
    assertMatch(checkout.uses, /^actions\/checkout@[0-9a-f]{40}$/);
    assertEquals(checkout.with, { "persist-credentials": false });
    // Deno alone: no tool input of the setup action is given.
    assertEquals(tools.uses, "./.github/actions/setup-tools");
    assertEquals(tools.with, undefined);
    const decide = steps.find((step) => step.run === "./scripts/spec_approval.ts --apply");
    assert(decide, "the deciding step runs the script with --apply");
    assertEquals(steps.indexOf(decide), 2, "the decision comes before every other step");
    // A hung script fails its step, so the failure step runs; a job cancelled by its own limit runs it not.
    assert(decide["timeout-minutes"] > 0 && decide["timeout-minutes"] < job["timeout-minutes"], "the step's limit");
    assertEquals(decide.env, {
        GH_TOKEN: "${{ github.token }}",
        PR_NUMBER: "${{ github.event.pull_request.number }}",
        EVENT_ACTION: "${{ github.event.action }}",
        EVENT_LABEL: "${{ github.event.label.name }}",
        EVENT_SENDER: "${{ github.event.sender.login }}",
        EVENT_HEAD: "${{ github.event.pull_request.head.sha }}",
    });
    assertEquals(Object.keys(decide.env).filter((name) => name !== "GH_TOKEN"), [...EVENT_VARIABLES]);
    const cleanup = steps.at(-1);
    assertEquals(cleanup.if, "failure()");
    assertEquals(cleanup.env, {
        GH_TOKEN: "${{ github.token }}",
        PR_NUMBER: "${{ github.event.pull_request.number }}",
        REPO: "${{ github.repository }}",
    });
    assert(cleanup.run.includes('path="repos/${REPO}/issues/${PR_NUMBER}/labels/spec%2Fapproved"'), cleanup.run);
    assert(cleanup.run.includes('gh api -X DELETE "${path}"'), cleanup.run);
    assertEquals(cleanup.run.match(/\bgh /g)?.length, 1, "one fixed gh api call");
    for (const step of steps) {
        // Event values reach a run: line only through env:.
        if (step.run !== undefined) assert(!step.run.includes("${{"), `an expression in a run: line: ${step.run}`);
        if (step.uses !== undefined) assert(step === checkout || step === tools, `an action: ${step.uses}`);
        for (const key of Object.keys(step.with ?? {})) {
            assert(!["ref", "repository", "path", "token", "submodules", "fetch-depth"].includes(key), key);
        }
        // The token only where gh is called: the script and the cleanup.
        if (step.env?.GH_TOKEN !== undefined) {
            assertEquals(step.env.GH_TOKEN, "${{ github.token }}");
            assert(step === decide || step === cleanup, `a token in ${step.name}`);
        }
        assert(step.if === undefined || step === cleanup, `a condition on ${step.name ?? step.uses}`);
    }
    // No checkout, fetch, or run of the head.
    for (const step of steps) {
        assert(!/\bgit fetch\b|refs\/pull\/|gh pr checkout/.test(step.run ?? ""), "nothing fetches the head");
    }
    assert(!text.includes("secrets."), "no secret");
    assert(!/cache/i.test(JSON.stringify(steps.map((step) => [step.uses ?? "", step.with ?? {}]))), "no cache");
    // Every expression in the file names one of these and nothing else of the event.
    const allowed = new Set([
        "github.event.pull_request.number",
        "github.token",
        "github.event.action",
        "github.event.label.name",
        "github.event.sender.login",
        "github.event.pull_request.head.sha",
        "github.repository",
    ]);
    for (const [, expression] of text.matchAll(/\$\{\{([^}]*)\}\}/g)) {
        assert(allowed.has(expression.trim()), expression);
    }
});

Deno.test({
    name: "the setup action restores and saves no cache",
    // The action lands with the workflow check (task 1.6 of repo-harness-alignment); until then it is not in the tree.
    ignore: !exists(`${ROOT}/.github/actions/setup-tools/action.yml`),
    fn() {
        const action = parseYaml(read(".github/actions/setup-tools/action.yml")) as Yaml;
        assertEquals(Object.keys(action.inputs ?? {}).filter((input) => /cache/i.test(input)), []);
        for (const step of action.runs.steps as Yaml[]) {
            assert(!/cache/i.test(step.uses ?? ""), step.uses);
            assertEquals(Object.keys(step.with ?? {}).filter((input) => /cache/i.test(input)), [], step.uses);
        }
    },
});

Deno.test("the failure step removes spec/approved, tolerates its absence, and fails on any other error", () => {
    const { doc } = workflow();
    const cleanup = doc.jobs.labels.steps.at(-1);
    const dir = Deno.makeTempDirSync();
    try {
        Deno.writeTextFileSync(`${dir}/step.sh`, cleanup.run);
        // A gh that logs its arguments and answers as the mode says.
        Deno.writeTextFileSync(
            `${dir}/gh`,
            '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$LOG"\ncase "$MODE" in\n' +
                '  ok) exit 0 ;;\n  absent) echo "gh: Label does not exist (HTTP 404)" >&2; exit 1 ;;\n' +
                '  broken) echo "gh: Forbidden (HTTP 403)" >&2; exit 1 ;;\nesac\n',
        );
        Deno.chmodSync(`${dir}/gh`, 0o755);
        const step = (mode: string, number = "7") => {
            Deno.writeTextFileSync(`${dir}/log`, "");
            // The runner's default shell for a run: step, bash -e.
            const result = new Deno.Command("bash", {
                args: ["-e", `${dir}/step.sh`],
                env: {
                    PATH: `${dir}:${Deno.env.get("PATH")}`,
                    MODE: mode,
                    LOG: `${dir}/log`,
                    GH_TOKEN: "t",
                    PR_NUMBER: number,
                    REPO: "o/r",
                },
                stdout: "piped",
                stderr: "piped",
            }).outputSync();
            return { code: result.code, log: Deno.readTextFileSync(`${dir}/log`) };
        };
        assertEquals(step("ok"), { code: 0, log: "api -X DELETE repos/o/r/issues/7/labels/spec%2Fapproved\n" });
        assertEquals(step("absent").code, 0);
        assertEquals(step("broken").code, 1);
        assertEquals(step("ok", "7; rm -rf /"), { code: 1, log: "" });
        assertEquals(step("ok", ""), { code: 1, log: "" });
    } finally {
        Deno.removeSync(dir, { recursive: true });
    }
});
