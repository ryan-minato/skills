#!/usr/bin/env -S deno run --no-prompt --allow-run=gh --allow-env=PR_NUMBER,EVENT_ACTION,EVENT_LABEL,EVENT_SENDER,EVENT_HEAD
// Keeps the `spec/approved` label of one pull request: the maintainer's approval of the OpenSpec
// approval package (.agents/knowledge/spec-workflow.md, ## Approval gates).
//
//   scripts/spec_approval.ts <n>        print the state the rules give now and the writes they call for;
//                                       writes nothing (`just spec-approval <n>`)
//   scripts/spec_approval.ts --apply    the `spec / labels` workflow (.github/workflows/spec-labels.yml):
//                                       the pull request and its event come from PR_NUMBER, EVENT_ACTION,
//                                       EVENT_LABEL, EVENT_SENDER, and EVENT_HEAD; makes the writes
//
// The rules. The run started by a person adding `spec/approved` records the approval in one comment
// from github-actions[bot] (the approved commit, the account, the id of the label event) when the pull
// request is open, its head is still the commit the event named, that adding is the latest adding of
// the label by a person and no person removed the label after it, the account has write or admin
// permission, the head holds at least one unarchived change and no link or submodule in the package,
// at most one record comment exists, and all of that still holds when the pull request and its label
// history are read again at the end. Every other run decides from the pull request's present labels,
// label history, and comments, not from its event: the label holds while exactly one record from
// github-actions[bot] answers the latest adding by a person, is not withdrawn, no person removed the
// label after that adding, and the approval package at the head equals the package at the recorded
// commit. Otherwise the label is removed and every live record is marked withdrawn (fail closed). A
// closed pull request is never written to by a run that succeeds.
//
// The approval package of a commit: every path under each unarchived change in openspec/changes/
// except its own tasks.md, by mode and blob, and openspec/specs/<capability>/spec.md (or its absence)
// for each delta openspec/changes/<change>/specs/<capability>/spec.md. A change the head holds only as
// openspec/changes/archive/<date>-<change>/ is compared through that copy, without its main specs,
// which the archive rewrites: an archive is not a package change. A symbolic link or a submodule in a
// package refuses the approval.
//
// Exit codes: 0 the approval holds (read-only) or the writes were made (--apply); 1 the approval does
// not hold (read-only); 2 a usage error, or gh or the API failed (with --apply, after removing the
// label). Any other error is a bug: with --apply the label is removed, then the script crashes with a
// stack trace. Everything is read through `gh api` with the token the environment gives gh; nothing of the
// pull request is checked out or run. A value a pull request controls reaches `gh api` only in a
// request body or percent-encoded in a path, and is printed only escaped, on one line. The script
// imports only local modules that import nothing remote, so no dependency loads beside the job's token
// (.agents/knowledge/management-code.md).
import { parseArgs } from "./lib/args.ts";
import { CommandError, REPO, runAsync, UsageError } from "./lib/repo.ts";

export const LABEL = "spec/approved";

/** The account the workflow's comments carry, with the type the API gives it. */
export const RECORDER = { login: "github-actions[bot]", type: "Bot" } as const;

/** The environment variables --apply reads; the shebang allows exactly these. */
export const EVENT_VARIABLES = ["PR_NUMBER", "EVENT_ACTION", "EVENT_LABEL", "EVENT_SENDER", "EVENT_HEAD"] as const;

export const CHANGES = "openspec/changes/";
export const ARCHIVE = "openspec/changes/archive/";
export const SPECS = "openspec/specs/";
const LINK_MODE = "120000";

export const USAGE = `usage: scripts/spec_approval.ts <pull-request-number>
       scripts/spec_approval.ts --apply

Decides the spec/approved label of one pull request from its labels, label history, comments, and
approval package (.agents/knowledge/spec-workflow.md, ## Approval gates).

  <n>       print the state and the writes the rules call for; write nothing (just spec-approval <n>)
  --apply   the spec / labels workflow only: read the pull request and its event from ${EVENT_VARIABLES.join(", ")},
            then record the approval or withdraw it

Exit codes: 0 approved (read-only) or written (--apply); 1 not approved (read-only); 2 usage error or a
failure of gh or the API.`;

const NUMBER = /^[1-9][0-9]{0,9}$/;
// The three fields of the record, as patterns without anchors; the writer and the reader are built from the same ones.
const COMMIT_PART = "[0-9a-f]{40}";
/** A GitHub login: letters, digits, and single hyphens inside, 39 characters at most. */
const LOGIN_PART = "[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}";
/** An issue event id, kept within the integers a JavaScript number holds exactly. */
const EVENT_ID_PART = "[1-9][0-9]{0,15}";
const COMMIT = new RegExp(`^${COMMIT_PART}$`);
const LOGIN = new RegExp(`^${LOGIN_PART}$`);
const EVENT_ID = new RegExp(`^${EVENT_ID_PART}$`);

/** The first line of the record comment: the approved commit, the account, and the id of the label event. */
export const RECORD_LINE = new RegExp(`^Approval record: (${COMMIT_PART}) (${LOGIN_PART}) (${EVENT_ID_PART})$`);
const STATUS_APPROVED = "Status: approved";
const STATUS_WITHDRAWN = "Status: withdrawn";

export interface Pull {
    state: "open" | "closed";
    /** The head commit id. */
    head: string;
    labels: string[];
}

export interface TreeEntry {
    path: string;
    mode: string;
    /** blob, tree, or commit (a submodule). */
    type: string;
    sha: string;
}

export interface Tree {
    entries: TreeEntry[];
    truncated: boolean;
}

export interface Actor {
    login: string;
    /** User, Bot, Organization, Mannequin. */
    type: string;
}

export interface LabelEvent {
    id: number;
    /** true for an adding (labeled), false for a removing (unlabeled). */
    added: boolean;
    label: string;
    actor: Actor | null;
}

export interface Comment {
    id: number;
    body: string;
    author: Actor | null;
}

/** The reads the decision makes and the writes it calls for; the writes name no label but spec/approved. */
export interface Api {
    pull(number: number): Promise<Pull>;
    tree(commit: string): Promise<Tree>;
    labelEvents(number: number): Promise<LabelEvent[]>;
    comments(number: number): Promise<Comment[]>;
    /** admin, write, read, or none. */
    permission(login: string): Promise<string>;
    addLabel(number: number): Promise<void>;
    /** false when the label was not on the pull request. */
    removeLabel(number: number): Promise<boolean>;
    createComment(number: number, body: string): Promise<void>;
    updateComment(id: number, body: string): Promise<void>;
}

/** What the workflow's event says; absent for a read-only run. */
export interface Event {
    action: string;
    label: string;
    sender: string;
    head: string;
}

export interface Input {
    number: number;
    apply: boolean;
    event?: Event;
}

export interface Change {
    name: string;
    /** Only a directory can be approved. */
    kind: "directory" | "link" | "submodule";
}

export interface ApprovalRecord {
    commentId: number;
    commit: string;
    login: string;
    eventId: number;
    withdrawn: boolean;
}

export interface Decision {
    approved: boolean;
    reason: string;
    /** Whether the pull request was open at the last read; a run that succeeds writes nothing to a closed one. */
    open: boolean;
    /** Whether spec/approved was on the pull request at the last read. */
    labeled: boolean;
    /** The record this run writes; `commentId` when it replaces the one record comment. */
    record?: { body: string; commit: string; commentId?: number };
    /** Live records to mark withdrawn before the label is removed. */
    withdraw: ApprovalRecord[];
}

function errorText(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

/** A value a pull request controls, escaped for one line of output. */
export const shown = (value: string) => JSON.stringify(value);

/** Keeps a message on one line, so that no text that reaches the log can start a workflow command. */
const oneLine = (text: string) => text.replaceAll(/[\r\n]+/g, " ");

const isPerson = (actor: Actor | null): actor is Actor => actor !== null && actor.type !== "Bot";

const list = (values: string[]) => values.map(shown).join(", ");

// ---- The approval package, read from a tree --------------------------------------------------------

const unsafe = (entry: TreeEntry) => entry.type === "commit" || entry.mode === LINK_MODE;

function kindOf(entry: TreeEntry): Change["kind"] | undefined {
    if (entry.type === "tree") return "directory";
    if (entry.type === "commit") return "submodule";
    if (entry.mode === LINK_MODE) return "link";
    return undefined;
}

/** The unarchived changes of a tree: the entries directly under openspec/changes/ but the archive and plain files. */
export function changesOf(tree: Tree): Change[] {
    const changes: Change[] = [];
    for (const entry of tree.entries) {
        if (!entry.path.startsWith(CHANGES)) continue;
        const name = entry.path.slice(CHANGES.length);
        if (name === "" || name === "archive" || name.includes("/")) continue;
        const kind = kindOf(entry);
        if (kind) changes.push({ name, kind });
    }
    return changes.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
}

interface Files {
    /** Path under the change directory to "mode sha"; directories and the change's own tasks.md left out. */
    files: Map<string, string>;
    /** Full paths of the links and submodules among them. */
    unsafe: string[];
}

function filesUnder(tree: Tree, dir: string): Files {
    const files = new Map<string, string>();
    const found: string[] = [];
    for (const entry of tree.entries) {
        if (entry.type === "tree" || !entry.path.startsWith(dir)) continue;
        const rest = entry.path.slice(dir.length);
        if (rest === "tasks.md") continue;
        files.set(rest, `${entry.mode} ${entry.sha}`);
        if (unsafe(entry)) found.push(entry.path);
    }
    return { files, unsafe: found };
}

/** A path's mode and object id, or "absent". */
function lineOf(tree: Tree, path: string): string {
    const entry = tree.entries.find((candidate) => candidate.path === path);
    return entry ? `${entry.mode} ${entry.sha}` : "absent";
}

/** The links and submodules at a path or at a directory above it. */
function unsafeAlong(tree: Tree, path: string): string[] {
    return tree.entries
        .filter((entry) => unsafe(entry) && (entry.path === path || path.startsWith(`${entry.path}/`)))
        .map((entry) => entry.path);
}

export interface Package {
    /** Each unarchived change directory with its files. */
    changes: Map<string, Map<string, string>>;
    /** The main spec of each delta, with its line or "absent". */
    specs: Map<string, string>;
    /** Paths that refuse approval: changes that are no directory, links and submodules inside the package. */
    refused: string[];
}

/**
 * The main spec a delta names: openspec/changes/<change>/specs/<capability>/spec.md gives
 * openspec/specs/<capability>/spec.md.
 */
function mainSpecOf(pathInChange: string): string | undefined {
    const delta = /^specs\/(.+)\/spec\.md$/.exec(pathInChange);
    return delta ? `${SPECS}${delta[1]}/spec.md` : undefined;
}

/** The approval package of a tree. */
export function packageOf(tree: Tree): Package {
    const changes = new Map<string, Map<string, string>>();
    const specs = new Map<string, string>();
    const refused: string[] = [];
    for (const change of changesOf(tree)) {
        if (change.kind !== "directory") {
            refused.push(`${CHANGES}${change.name}`);
            continue;
        }
        const { files, unsafe } = filesUnder(tree, `${CHANGES}${change.name}/`);
        refused.push(...unsafe);
        changes.set(change.name, files);
        for (const path of files.keys()) {
            const spec = mainSpecOf(path);
            if (spec !== undefined) specs.set(spec, lineOf(tree, spec));
        }
    }
    for (const spec of specs.keys()) refused.push(...unsafeAlong(tree, spec));
    return { changes, specs, refused: [...new Set(refused)].sort() };
}

const ARCHIVE_DATE = /^\d{4}-\d{2}-\d{2}-/;

/** The latest archived copy of change `name` a tree holds: openspec/changes/archive/<date>-<name>/. */
export function archivedCopy(tree: Tree, name: string): Files | undefined {
    const dirs = tree.entries
        .filter((entry) => entry.type === "tree" && entry.path.startsWith(ARCHIVE))
        .map((entry) => entry.path.slice(ARCHIVE.length))
        .filter((dir) => !dir.includes("/") && ARCHIVE_DATE.test(dir) && dir.replace(ARCHIVE_DATE, "") === name)
        .sort();
    const latest = dirs.at(-1);
    return latest === undefined ? undefined : filesUnder(tree, `${ARCHIVE}${latest}/`);
}

/** The paths whose line differs between two file maps. */
function differing(a: Map<string, string>, b: Map<string, string>): string[] {
    return [...new Set([...a.keys(), ...b.keys()])].filter((path) => a.get(path) !== b.get(path)).sort();
}

/** How the approval package at `head` differs from the one at `recorded`; empty when it is the same. */
export function packageProblems(recorded: Tree, head: Tree): string[] {
    const before = packageOf(recorded);
    const now = packageOf(head);
    const problems: string[] = [];
    if (before.changes.size === 0) problems.push("the recorded commit holds no unarchived change");
    if (before.refused.length > 0) problems.push(`the recorded package holds ${list(before.refused)}`);
    if (now.refused.length > 0) problems.push(`a link or submodule in the package at the head: ${list(now.refused)}`);
    for (const [name, files] of before.changes) {
        const current = now.changes.get(name);
        if (current !== undefined) {
            const paths = differing(files, current);
            if (paths.length > 0) problems.push(`the change ${shown(name)} differs in ${list(paths)}`);
            continue;
        }
        const archived = archivedCopy(head, name);
        if (archived === undefined) {
            problems.push(`the change ${shown(name)} is gone and not archived`);
            continue;
        }
        if (archived.unsafe.length > 0) problems.push(`a link or submodule in the archive: ${list(archived.unsafe)}`);
        const paths = differing(files, archived.files);
        if (paths.length > 0) problems.push(`the change ${shown(name)} was archived with ${list(paths)} changed`);
    }
    for (const name of now.changes.keys()) {
        if (!before.changes.has(name)) problems.push(`the change ${shown(name)} was added`);
    }
    // The main specs of the changes still unarchived at the head; an archived change's main specs are
    // the archive's to rewrite.
    for (const [spec, line] of now.specs) {
        if (lineOf(recorded, spec) !== line) problems.push(`the main spec ${shown(spec)} differs`);
    }
    return problems;
}

// ---- The record comment ----------------------------------------------------------------------------

/**
 * The record comment, from fixed text and three checked fields. No name of a file, a change, a label
 * the pull request controls, or a branch enters it.
 */
export function recordBody(commit: string, login: string, eventId: number, withdrawn: boolean): string {
    const id = String(eventId);
    if (!COMMIT.test(commit)) throw new Error("the record needs a commit id of 40 hexadecimal digits");
    if (!LOGIN.test(login)) throw new Error("the record needs a GitHub login");
    if (!EVENT_ID.test(id) || !Number.isSafeInteger(eventId)) {
        throw new Error("the record needs the id of a label event");
    }
    const first = `Approval record: ${commit} ${login} ${id}`;
    if (!RECORD_LINE.test(first)) throw new Error("the record line does not have its form");
    const link = `https://github.com/${REPO}/commit/${commit}`;
    const text = withdrawn
        ? `The approval recorded above no longer holds: the approval package changed after that commit, a person ` +
            `removed \`${LABEL}\`, or a run could not stand behind it. The maintainer approves again by adding ` +
            `\`${LABEL}\` again.`
        : `The account named above approved the specification package of this pull request at ${link} by adding ` +
            `\`${LABEL}\`. The \`spec / labels\` workflow keeps the label while the approval package at the head ` +
            "equals the package at that commit (.agents/knowledge/spec-workflow.md, Approval gates).";
    return `${first}\n${withdrawn ? STATUS_WITHDRAWN : STATUS_APPROVED}\n\n${text}\n`;
}

/** The records among the comments: one per comment by the workflow's account whose first line has the form. */
export function recordsOf(comments: Comment[]): ApprovalRecord[] {
    const records: ApprovalRecord[] = [];
    for (const comment of comments) {
        if (comment.author?.login !== RECORDER.login || comment.author.type !== RECORDER.type) continue;
        const [first = "", second = ""] = comment.body.split(/\r?\n/);
        const match = RECORD_LINE.exec(first);
        if (!match || !Number.isSafeInteger(Number(match[3]))) continue;
        records.push({
            commentId: comment.id,
            commit: match[1],
            login: match[2],
            eventId: Number(match[3]),
            withdrawn: second.trim() !== STATUS_APPROVED,
        });
    }
    return records;
}

// ---- The label history -----------------------------------------------------------------------------

/** The latest adding of spec/approved by a person, by event id. */
export function latestApproval(events: LabelEvent[]): LabelEvent | undefined {
    let latest: LabelEvent | undefined;
    for (const event of events) {
        if (!event.added || event.label !== LABEL || !isPerson(event.actor)) continue;
        if (latest === undefined || event.id > latest.id) latest = event;
    }
    return latest;
}

/** Whether a person removed spec/approved in an event with a greater id than `id`. */
export function removedAfter(events: LabelEvent[], id: number): boolean {
    return events.some((event) => event.id > id && !event.added && event.label === LABEL && isPerson(event.actor));
}

// ---- The decision ----------------------------------------------------------------------------------

type Outcome =
    | { refusal: string; pull?: Pull; events?: LabelEvent[] }
    | { pull: Pull; eventId: number; events: LabelEvent[] };

/**
 * Why the run its event started may not record the approval, or, after reading both again, the pull
 * request and the label event it records.
 */
async function recordingRefusal(
    api: Api,
    number: number,
    pull: Pull,
    event: Event,
    events: LabelEvent[],
    records: ApprovalRecord[],
): Promise<Outcome> {
    if (pull.state !== "open") return { refusal: "the pull request is closed" };
    if (!LOGIN.test(event.sender) || !COMMIT.test(event.head)) {
        return { refusal: "the event names no usable account or commit" };
    }
    if (pull.head !== event.head) return { refusal: "the head moved after the label was added" };
    const latest = latestApproval(events);
    if (latest === undefined || latest.actor?.login !== event.sender) {
        return { refusal: `the latest adding of ${LABEL} by a person is not this event` };
    }
    if (removedAfter(events, latest.id)) return { refusal: `a person removed ${LABEL} after this event` };
    if (records.length > 1) {
        return { refusal: "more than one record comment exists; the maintainer deletes all but one" };
    }
    const found = packageOf(await api.tree(pull.head));
    if (found.changes.size === 0 && found.refused.length === 0) {
        return { refusal: "the head holds no unarchived change" };
    }
    if (found.refused.length > 0) {
        return { refusal: `a link or submodule in the package at the head: ${list(found.refused)}` };
    }
    const permission = await api.permission(event.sender);
    if (permission !== "admin" && permission !== "write") {
        return { refusal: `the account's permission is ${shown(permission)}, below write` };
    }
    const again = await api.pull(number);
    if (again.state !== "open") return { refusal: "the pull request was closed during the run", pull: again };
    if (again.head !== event.head) return { refusal: "the head moved during the run", pull: again };
    // The label history once more, after every other read: a removal that landed while this run was
    // reading is not restored, and a later adding belongs to the run it started.
    const now = await api.labelEvents(number);
    if (latestApproval(now)?.id !== latest.id) {
        return { refusal: `${LABEL} was added again during the run`, pull: again, events: now };
    }
    if (removedAfter(now, latest.id)) {
        return { refusal: `a person removed ${LABEL} during the run`, pull: again, events: now };
    }
    return { pull: again, eventId: latest.id, events: now };
}

/** Decides spec/approved for pull request `number`, whose present state `pull` was just read. */
export async function decide(api: Api, number: number, pull: Pull, event: Event | undefined): Promise<Decision> {
    let events = await api.labelEvents(number);
    const records = recordsOf(await api.comments(number));
    const live = records.filter((record) => !record.withdrawn);
    let current = pull;
    let refused = "";
    if (event !== undefined && event.action === "labeled" && event.label === LABEL) {
        const outcome = await recordingRefusal(api, number, pull, event, events, records);
        if ("refusal" in outcome) {
            refused = `; not recorded: ${outcome.refusal}`;
            // The pull request as the refusal last read it, so that a closing meanwhile stops the writes.
            if (outcome.pull) current = outcome.pull;
            if (outcome.events) events = outcome.events;
        } else {
            return {
                approved: true,
                reason: `this run records the approval of ${event.head} by ${shown(event.sender)} ` +
                    `(label event ${outcome.eventId})`,
                open: true,
                labeled: outcome.pull.labels.includes(LABEL),
                record: {
                    body: recordBody(event.head, event.sender, outcome.eventId, false),
                    commit: event.head,
                    commentId: records[0]?.commentId,
                },
                withdraw: [],
            };
        }
    }
    const base = { open: current.state === "open", labeled: current.labels.includes(LABEL) };
    const problems: string[] = [];
    if (!base.labeled) problems.push(`${LABEL} is not on the pull request`);
    const latest = latestApproval(events);
    if (latest === undefined) problems.push(`no person added ${LABEL}`);
    else if (removedAfter(events, latest.id)) problems.push(`a person removed ${LABEL} after adding it`);
    if (records.length === 0) problems.push("no record comment");
    else if (records.length > 1) problems.push("more than one record comment; the maintainer deletes all but one");
    else if (records[0].withdrawn) problems.push("the record is withdrawn");
    else {
        const [record] = records;
        if (latest !== undefined && record.eventId !== latest.id) {
            problems.push(`the record answers an earlier adding of ${LABEL} than the latest`);
        }
        const head = await api.tree(current.head);
        const recorded = record.commit === current.head ? head : await api.tree(record.commit);
        problems.push(...packageProblems(recorded, head));
    }
    if (problems.length === 0) {
        const [record] = records;
        return {
            ...base,
            approved: true,
            reason: `the approval of ${record.commit} by ${shown(record.login)} (label event ${record.eventId}) holds`,
            withdraw: [],
        };
    }
    return { ...base, approved: false, reason: `${problems.join("; ")}${refused}`, withdraw: live };
}

// ---- Writing ---------------------------------------------------------------------------------------

/** One write a decision calls for: what it is, and the call that makes it, which returns what was done. */
export interface Write {
    text: string;
    make(api: Api, number: number): Promise<string>;
}

/** The writes a decision calls for, in the order they are made: the record first, the label last. */
export function writesOf(decision: Decision): Write[] {
    if (!decision.open) return [];
    const writes: Write[] = [];
    const { record } = decision;
    if (record) {
        const { commentId } = record;
        writes.push({
            text: `${commentId === undefined ? "post" : "replace"} the record of ${record.commit}`,
            make: async (api, number) => {
                if (commentId === undefined) await api.createComment(number, record.body);
                else await api.updateComment(commentId, record.body);
                return `recorded the approval of ${record.commit}`;
            },
        });
    }
    for (const live of decision.withdraw) {
        writes.push({
            text: `mark the record of ${live.commit} withdrawn`,
            make: async (api) => {
                await api.updateComment(live.commentId, recordBody(live.commit, live.login, live.eventId, true));
                return `marked the record of ${live.commit} withdrawn`;
            },
        });
    }
    if (decision.approved && !decision.labeled) {
        writes.push({
            text: `add ${LABEL}`,
            make: async (api, number) => {
                await api.addLabel(number);
                return `added ${LABEL}`;
            },
        });
    }
    if (!decision.approved && decision.labeled) {
        writes.push({
            text: `remove ${LABEL}`,
            make: async (api, number) =>
                (await api.removeLabel(number)) ? `removed ${LABEL}` : `${LABEL} was already gone`,
        });
    }
    return writes;
}

export interface Output {
    /** stdout: the state and the writes. */
    print(line: string): void;
    /** stderr: what was done, and errors. */
    note(line: string): void;
}

/** After an error: spec/approved off the pull request, open or closed. The error itself is still reported. */
async function failClosed(api: Api, number: number, out: Output): Promise<void> {
    try {
        const removed = await api.removeLabel(number);
        out.note(`after the error: ${removed ? `removed ${LABEL}` : `${LABEL} was not on the pull request`}`);
    } catch (error) {
        // Deliberate: the original error decides the exit status; the workflow's failure step tries again.
        out.note(`after the error: could not remove ${LABEL}: ${oneLine(errorText(error))}`);
    }
}

/** Decides, prints, and with `apply` writes; returns the exit status. */
export async function run(input: Input, api: Api, out: Output): Promise<number> {
    const { number, apply } = input;
    try {
        const decision = await decide(api, number, await api.pull(number), input.event);
        out.print(`${decision.approved ? "approved" : "not approved"}: ${decision.reason}`);
        if (!decision.open) out.print("the pull request is closed: nothing is written");
        const writes = writesOf(decision);
        for (const planned of writes) out.print(`write: ${planned.text}`);
        if (!apply) return decision.approved ? 0 : 1;
        for (const planned of writes) out.note(`done: ${await planned.make(api, number)}`);
        if (writes.length === 0) out.note("done: nothing to write");
        return 0;
    } catch (error) {
        out.note(`error: ${oneLine(errorText(error))}`);
        if (apply) await failClosed(api, number, out);
        if (error instanceof CommandError) return 2;
        throw error;
    }
}

// ---- The command line and the environment ----------------------------------------------------------

/** Reads the pull request number from the arguments, or with --apply the number and the event from the environment. */
export function parseInput(argv: string[], env: (name: string) => string | undefined): Input | "help" {
    const args = parseArgs(argv, { boolean: ["apply", "help"], alias: { h: "help" } });
    if (args.flags.help) return "help";
    if (args.flags.apply) {
        if (args.positional.length > 0) {
            throw new UsageError(
                "--apply reads the pull request from PR_NUMBER and takes no number; " +
                    "run scripts/spec_approval.ts <n> to read the state",
            );
        }
        const given = env("PR_NUMBER") ?? "";
        if (!NUMBER.test(given)) throw new UsageError(`PR_NUMBER must be a pull request number, got ${shown(given)}`);
        const action = env("EVENT_ACTION") ?? "";
        if (action === "") {
            throw new UsageError("EVENT_ACTION is empty; --apply runs in the spec / labels workflow, which sets it");
        }
        return {
            number: Number(given),
            apply: true,
            event: {
                action,
                label: env("EVENT_LABEL") ?? "",
                sender: env("EVENT_SENDER") ?? "",
                head: env("EVENT_HEAD") ?? "",
            },
        };
    }
    if (args.positional.length !== 1) throw new UsageError("give one pull request number");
    const [given] = args.positional;
    if (!NUMBER.test(given)) throw new UsageError(`the pull request number must be digits, got ${shown(given)}`);
    return { number: Number(given), apply: false };
}

/** Runs `gh` with `args` and `stdin`, and returns its standard output; a failure names the command and its message. */
export type Gh = (args: string[], stdin?: string) => Promise<string>;

export const runGh: Gh = async (args, stdin) => {
    const out = await runAsync("gh", args, { stdin });
    if (out.code !== 0) {
        throw new CommandError(`gh ${args.join(" ")} exited ${out.code}: ${out.stderr.trim() || "no message"}`);
    }
    return out.stdout;
};

type Json = { [key: string]: unknown };

const isObject = (value: unknown): value is Json =>
    typeof value === "object" && value !== null && !Array.isArray(value);

function shapeError(what: string, value: unknown): CommandError {
    const text = oneLine(JSON.stringify(value) ?? String(value));
    return new CommandError(
        `${what} has an unexpected shape: ${text.length > 300 ? `${text.slice(0, 300)}…` : text}; ` +
            "read it with `gh api` and fix scripts/spec_approval.ts if the GitHub REST API changed",
    );
}

function parseJson(text: string, what: string): unknown {
    try {
        return JSON.parse(text);
    } catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
        throw new CommandError(`${what} did not answer JSON: ${oneLine(text.slice(0, 200))}`);
    }
}

/**
 * The REST calls behind Api, on `repo`. A number is checked as digits before it enters a path, a login
 * is percent-encoded there, and a commit id is checked as 40 hexadecimal digits; every other value a
 * pull request controls travels in a request body. Every answer is checked for the fields the
 * decision reads.
 */
export function ghApi(gh: Gh = runGh, repo: string = REPO): Api {
    const issues = `repos/${repo}/issues`;
    const label = encodeURIComponent(LABEL);
    const checked = (number: number) => {
        if (!NUMBER.test(String(number))) throw new Error("the pull request number must be digits");
        return String(number);
    };
    const get = async (path: string) => parseJson(await gh(["api", path]), `gh api ${path}`);
    const all = async (path: string): Promise<unknown[]> => {
        const what = `gh api --paginate --slurp ${path}`;
        const pages = parseJson(await gh(["api", "--paginate", "--slurp", `${path}?per_page=100`]), what);
        if (!Array.isArray(pages) || !pages.every(Array.isArray)) throw shapeError(what, pages);
        return pages.flat();
    };
    const actor = (value: unknown, what: string): Actor | null => {
        if (value === null || value === undefined) return null;
        if (!isObject(value) || typeof value.login !== "string" || typeof value.type !== "string") {
            throw shapeError(what, value);
        }
        return { login: value.login, type: value.type };
    };
    const send = (args: string[], body: Json) => gh(["api", ...args, "--input", "-"], JSON.stringify(body));
    return {
        pull: async (number) => {
            const path = `repos/${repo}/pulls/${checked(number)}`;
            const pull = await get(path);
            if (
                !isObject(pull) || (pull.state !== "open" && pull.state !== "closed") || !isObject(pull.head) ||
                typeof pull.head.sha !== "string" || !COMMIT.test(pull.head.sha) || !Array.isArray(pull.labels) ||
                !pull.labels.every((item) => isObject(item) && typeof item.name === "string")
            ) throw shapeError(`gh api ${path}`, pull);
            return {
                state: pull.state,
                head: pull.head.sha,
                labels: (pull.labels as Json[]).map((item) => item.name as string),
            };
        },
        tree: async (commit) => {
            if (!COMMIT.test(commit)) throw new CommandError("a tree is read for a commit id of 40 hexadecimal digits");
            const path = `repos/${repo}/git/trees/${commit}?recursive=1`;
            const tree = await get(path);
            if (!isObject(tree) || !Array.isArray(tree.tree) || typeof tree.truncated !== "boolean") {
                throw shapeError(`gh api ${path}`, tree);
            }
            if (tree.truncated) {
                throw new CommandError(`the tree of ${commit} is truncated, so its approval package cannot be read`);
            }
            const entries = tree.tree.map((entry) => {
                if (
                    !isObject(entry) || typeof entry.path !== "string" || typeof entry.mode !== "string" ||
                    typeof entry.type !== "string" || typeof entry.sha !== "string"
                ) throw shapeError(`an entry of gh api ${path}`, entry);
                return { path: entry.path, mode: entry.mode, type: entry.type, sha: entry.sha };
            });
            return { entries, truncated: false };
        },
        labelEvents: async (number) => {
            const path = `${issues}/${checked(number)}/events`;
            const events: LabelEvent[] = [];
            for (const event of await all(path)) {
                if (!isObject(event) || typeof event.event !== "string") throw shapeError(`an event of ${path}`, event);
                if (event.event !== "labeled" && event.event !== "unlabeled") continue;
                if (
                    !Number.isSafeInteger(event.id) || !isObject(event.label) || typeof event.label.name !== "string"
                ) throw shapeError(`a label event of ${path}`, event);
                events.push({
                    id: event.id as number,
                    added: event.event === "labeled",
                    label: event.label.name,
                    actor: actor(event.actor, `the actor of a label event of ${path}`),
                });
            }
            return events;
        },
        comments: async (number) => {
            const path = `${issues}/${checked(number)}/comments`;
            return (await all(path)).map((comment) => {
                if (
                    !isObject(comment) || !Number.isSafeInteger(comment.id) ||
                    (typeof comment.body !== "string" && comment.body !== null)
                ) throw shapeError(`a comment of ${path}`, comment);
                return {
                    id: comment.id as number,
                    body: comment.body ?? "",
                    author: actor(comment.user, `the author of a comment of ${path}`),
                };
            });
        },
        permission: async (login) => {
            const path = `repos/${repo}/collaborators/${encodeURIComponent(login)}/permission`;
            const answer = await get(path);
            if (!isObject(answer) || typeof answer.permission !== "string") throw shapeError(`gh api ${path}`, answer);
            return answer.permission;
        },
        addLabel: async (number) =>
            void await send(["-X", "POST", `${issues}/${checked(number)}/labels`], { labels: [LABEL] }),
        removeLabel: async (number) => {
            try {
                await gh(["api", "-X", "DELETE", `${issues}/${checked(number)}/labels/${label}`]);
                return true;
            } catch (error) {
                // A label already gone answers 404, which is what the removal wants; anything else fails.
                if (error instanceof CommandError && error.message.includes("HTTP 404")) return false;
                throw error;
            }
        },
        createComment: async (number, body) =>
            void await send(["-X", "POST", `${issues}/${checked(number)}/comments`], { body }),
        updateComment: async (id, body) => {
            if (!Number.isSafeInteger(id) || id < 1) throw new CommandError("a comment id is a positive integer");
            await send(["-X", "PATCH", `${issues}/comments/${id}`], { body });
        },
    };
}

export interface Io {
    env: (name: string) => string | undefined;
    gh: Gh;
    out: Output;
}

export async function main(argv: string[], io: Partial<Io> = {}): Promise<number> {
    const out = io.out ?? { print: (line) => console.log(line), note: (line) => console.error(line) };
    let input: Input | "help";
    try {
        input = parseInput(argv, io.env ?? ((name) => Deno.env.get(name)));
    } catch (error) {
        if (!(error instanceof UsageError)) throw error;
        out.note(`error: ${error.message}\n\n${USAGE}`);
        return 2;
    }
    if (input === "help") {
        out.print(USAGE);
        return 0;
    }
    return await run(input, ghApi(io.gh ?? runGh), out);
}

if (import.meta.main) Deno.exit(await main(Deno.args));
