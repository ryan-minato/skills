#!/usr/bin/env -S deno run --no-prompt --allow-read --allow-write --allow-run=git,openspec --allow-env=GH_TOKEN,GITHUB_TOKEN,GITHUB_API_URL,OPENSPEC_NO_UPDATE_CHECK --allow-net=api.github.com
// Operate on the OpenSpec changes a pull request touches.
//
// This repository's own management script: `checks / spec` and `spec / labels` call it,
// `just spec-check` and `just spec-changes` wrap it, and the archive executor runs its `archive` command.
// It exports its label taxonomy (MANAGED_LABELS) for `scripts/validate_harness.ts` to import.
//
// Commands (global options --root DIR and --changes-dir DIR go before the command):
//   snapshot --repo OWNER/NAME --pr N [--out FILE] [--api-url URL] [--max-files N] [--max-file-bytes N]
//            [--max-total-bytes N] [--max-calls N]          build a head snapshot from the GitHub REST API
//   status   SOURCE [--change NAME]... [--json]             progress table of the related changes
//   check    SOURCE | --all  [--draft] [--shape combined|split] [--no-validate] [--openspec EXE]
//   archive  --base REF --head REF [--dry-run] [--json] [--openspec EXE]   archive every complete change
//   labels   SOURCE [--current A,B] [--json] | --taxonomy   the archive-axis and progress-axis labels
// SOURCE is `--base REF --head REF` (git plumbing) or `--snapshot FILE` (from the snapshot command).
//
// A request's *related changes* are the directories under the changes directory (default `openspec/changes/`)
// whose files the request touches; a directory under `archive/` counts under its change name with the leading
// `YYYY-MM-DD-` stripped. Each change is `active` (its directory exists at the head), `archived` (an archive
// directory for it exists at the head), or `removed`.
//
// The head is read through one of two sources, never through a checkout: `--base`/`--head` with git plumbing
// (GitSource), or `--snapshot FILE` built from the GitHub REST API by the `snapshot` command (SnapshotSource).
// `archive` is the one command that writes, and it takes the git source only.
//
// Every interface is checked where its data enters — git's exit status and UTF-8 output, the API's status and
// JSON shape, the snapshot document's keys, the OpenSpec CLI's presence and exit status — and fails naming it;
// nothing is reported from a partial read.
//
// Exit codes: 0 success; 1 a failure, a finding, or a refusal; 2 bad arguments, an unknown change name
// included. An unexpected error ends with its stack trace (exit 1).
//
// It runs beside a write token in `spec / labels`, so it imports local modules only.
// Permissions: read and write anywhere (the snapshot file is caller-named, under $RUNNER_TEMP in CI), git and
// the OpenSpec CLI, the two token variables, GITHUB_API_URL, OPENSPEC_NO_UPDATE_CHECK, and api.github.com.
import { parseArgs } from "./lib/args.ts";
import { UsageError } from "./lib/repo.ts";

// --- constants ---------------------------------------------------------------------------------------------

/** Python's `\s` (and `str.isspace`), which the original regular expressions and `strip()` calls used. */
const WS = "[\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
// `(?<![^\n])` and `(?![^\n])` are Python's MULTILINE `^` and `$`: only `\n` ends a line.
const OPEN_TASK = new RegExp(String.raw`(?<![^\n])[ \t]*-[ \t]*\[ \][ \t]*([^\n]*)(?![^\n])`, "g");
const DONE_TASK = new RegExp(String.raw`(?<![^\n])${WS}*-${WS}*\[[xX]\]${WS}`, "g");
const SKIP_SPECS = new RegExp(String.raw`(?<![^\n])${WS}*skip_specs${WS}*:${WS}*true${WS}*(?![^\n])`);
const ARCHIVE_DATE = /^\p{Nd}{4}-\p{Nd}{2}-\p{Nd}{2}-/u;
const WS_RUN = new RegExp(`${WS}+`);
const WS_LEAD = new RegExp(`^${WS}+`);
const WS_TRAIL = new RegExp(`${WS}+$`);

/** The archive axis: one of the two is on every request with a live related change. */
export const ARCHIVED_LABELS = ["spec/unarchived", "spec/archived"] as const;
/** The progress axis. */
export const PROGRESS_LABELS = ["spec/not-started", "spec/in-progress", "spec/done"] as const;
/** Every label `labels` may add or remove; `labels --taxonomy` prints it. */
export const MANAGED_LABELS: readonly string[] = [...ARCHIVED_LABELS, ...PROGRESS_LABELS];

export const SNAPSHOT_SCHEMA = "spec-snapshot/1";
type Kind = "object" | "list" | "string";
// Every key the snapshot readers index, with its JSON type. The base side records directory names only,
// because no base document is ever read.
const SNAPSHOT_KEYS: [string[], Kind][] = [
    [["changed_paths"], "list"],
    [["base", "sha"], "string"],
    [["base", "dirs"], "list"],
    [["head", "sha"], "string"],
    [["head", "dirs"], "list"],
    [["head", "paths"], "list"],
    [["head", "files"], "object"],
];
// A request-authored directory name is read only when it is a plain change name.
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/;
const SAFE_REPO = /^[A-Za-z0-9._-]{1,100}\/[A-Za-z0-9._-]{1,100}$/;
// The documents inside a change directory the commands read; nothing else is fetched.
const CHANGE_DOCS = ["proposal.md", "design.md", "tasks.md", ".openspec.yaml"];
export const MAX_FILES = 3000;
export const MAX_FILE_BYTES = 1_000_000;
export const MAX_TOTAL_BYTES = 8_000_000;
// The platform token's REST budget is shared by every workflow of the repository.
export const MAX_CALLS = 200;
const ATTEMPTS = 3;
const TRANSIENT_STATUSES = [429, 500, 502, 503, 504];
const JSON_KINDS: Record<Kind, string> = { object: "an object", list: "a list", string: "a string" };
const HTTP_HINTS: Record<number, string> = {
    401: "the token is missing or expired",
    403: "the token lacks `contents: read` or `pull-requests: read`, or the rate limit is spent",
    404: "the object is gone, as when the head is force-pushed or deleted while this runs, so rerun on the " +
        "current head; or the token cannot see it",
};
const NO_CHANGE = "No related OpenSpec change: the diff touches nothing under the changes directory.\n";

/** A tool failure, a finding, or a refusal (exit 1). Bad arguments are a UsageError (exit 2). */
export class Failure extends Error {}

// --- text helpers (Python string semantics the output depends on) ------------------------------------------

/** `str.strip()`. */
export function pyStrip(text: string): string {
    return text.replace(WS_LEAD, "").replace(WS_TRAIL, "");
}

/** `str.split()` with no separator. */
function pySplit(text: string): string[] {
    return text.split(WS_RUN).filter((part) => part !== "");
}

/** The length in code points, as Python's `len()` counts a string. */
export function cpLen(text: string): number {
    let count = 0;
    for (let i = 0; i < text.length; i++) {
        const unit = text.charCodeAt(i);
        if (unit >= 0xd800 && unit <= 0xdbff && i + 1 < text.length) {
            const next = text.charCodeAt(i + 1);
            if (next >= 0xdc00 && next <= 0xdfff) i++;
        }
        count++;
    }
    return count;
}

/** The first `count` code points, as Python's `text[:count]`. */
export function cpSlice(text: string, count: number): string {
    let i = 0;
    for (let taken = 0; taken < count && i < text.length; taken++) {
        const unit = text.charCodeAt(i);
        const paired = unit >= 0xd800 && unit <= 0xdbff && i + 1 < text.length &&
            text.charCodeAt(i + 1) >= 0xdc00 && text.charCodeAt(i + 1) <= 0xdfff;
        i += paired ? 2 : 1;
    }
    return text.slice(0, i);
}

/** Orders strings by code point, as Python's `sorted()` and `max()` do (UTF-16 order differs above U+FFFF). */
export function cmpCodePoints(a: string, b: string): number {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
        const x = a.charCodeAt(i);
        const y = b.charCodeAt(i);
        if (x !== y) return rank(x) - rank(y);
    }
    return a.length - b.length;
}

function rank(unit: number): number {
    if (unit >= 0xe000) return unit - 0x800;
    if (unit >= 0xd800) return unit + 0x2000;
    return unit;
}

function maxString(items: string[]): string | null {
    let best: string | null = null;
    for (const item of items) if (best === null || cmpCodePoints(item, best) > 0) best = item;
    return best;
}

/** `json.dumps(value, indent=2)`: ASCII only, every other character escaped as `\uXXXX`. */
export function pyJson(value: unknown): string {
    return JSON.stringify(value, null, 2).replace(
        /[\u007f-\uffff]/g,
        (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"),
    );
}

/** `urllib.parse.quote(text, safe)`. */
export function pyQuote(text: string, safe = "/"): string {
    let out = "";
    for (const byte of new TextEncoder().encode(text)) {
        const char = String.fromCharCode(byte);
        if (/[A-Za-z0-9_.~-]/.test(char) || (byte < 0x80 && safe.includes(char))) out += char;
        else out += "%" + byte.toString(16).toUpperCase().padStart(2, "0");
    }
    return out;
}

/** A value as Python's `repr()` would show a string or a missing value, for messages. */
function repr(value: unknown): string {
    if (value === undefined || value === null) return "None";
    if (typeof value === "string") return `'${value.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'`;
    return JSON.stringify(value);
}

function brief(value: unknown): string {
    return String(JSON.stringify(value)).slice(0, 200);
}

const STRICT_UTF8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/** Bytes another program printed, decoded as Python's text mode did: strict UTF-8, line endings to `\n`. */
function textOutput(bytes: Uint8Array, what: string): string {
    let text: string;
    try {
        text = STRICT_UTF8.decode(bytes);
    } catch {
        throw new Failure(`${what} printed bytes that are not UTF-8; nothing is reported from a partial read.`);
    }
    return text.replace(/\r\n?/g, "\n");
}

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/**
 * `base64.b64decode(text)` without validation: characters outside the alphabet are skipped, decoding stops at
 * the first complete padding, and an incomplete quantum is an error.
 */
export function b64decode(text: string): Uint8Array {
    const out: number[] = [];
    let quad = 0;
    let left = 0;
    let pads = 0;
    if (/[\u0080-\uffff]/.test(text)) throw new Failure("string argument should contain only ASCII characters");
    for (const char of text) {
        if (char === "=") {
            if (quad >= 2 && quad + ++pads >= 4) return new Uint8Array(out);
            continue;
        }
        const value = B64.indexOf(char);
        if (value === -1) continue;
        pads = 0;
        if (quad === 0) {
            left = value;
        } else if (quad === 1) {
            out.push(((left << 2) | (value >> 4)) & 0xff);
            left = value & 0x0f;
        } else if (quad === 2) {
            out.push(((left << 4) | (value >> 2)) & 0xff);
            left = value & 0x03;
        } else {
            out.push(((left << 6) | value) & 0xff);
            left = 0;
        }
        quad = (quad + 1) % 4;
    }
    if (quad === 1) {
        throw new Failure("Invalid base64-encoded string: the data characters are 1 more than a multiple of 4");
    }
    if (quad !== 0) throw new Failure("Incorrect padding");
    return new Uint8Array(out);
}

function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isKind(value: unknown, kind: Kind): boolean {
    if (kind === "object") return isObject(value);
    if (kind === "list") return Array.isArray(value);
    return typeof value === "string";
}

function jsonKind(value: unknown): string {
    if (isObject(value)) return JSON_KINDS.object;
    if (Array.isArray(value)) return JSON_KINDS.list;
    if (typeof value === "string") return JSON_KINDS.string;
    if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
    if (typeof value === "boolean") return "bool";
    return "NoneType";
}

// --- git plumbing ------------------------------------------------------------------------------------------

interface GitOutput {
    code: number;
    stdout: Uint8Array;
    stderr: string;
}

// Not lib/repo.ts `run`: git's output must decode strictly, so a non-UTF-8 path or document fails here
// instead of reaching a label or a reply with replacement characters.
function gitRaw(root: string, args: string[]): GitOutput {
    const result = new Deno.Command("git", {
        args: ["-C", root, ...args],
        stdin: "null",
        stdout: "piped",
        stderr: "piped",
    }).outputSync();
    return { code: result.code, stdout: result.stdout, stderr: new TextDecoder().decode(result.stderr) };
}

function git(root: string, args: string[]): string {
    const result = gitRaw(root, args);
    if (result.code !== 0) {
        throw new Failure(`\`git ${args.join(" ")}\` exited ${result.code}: ${result.stderr.trim()}`);
    }
    return textOutput(result.stdout, `\`git ${args.join(" ")}\``);
}

function resolve(root: string, ref: string, what: string): string {
    const result = gitRaw(root, ["rev-parse", "--verify", "--quiet", `${ref}^{commit}`]);
    if (result.code !== 0) {
        throw new UsageError(
            `${what} ${repr(ref)} does not resolve to a commit in ${root}; fetch it first, or read the head ` +
                "through `snapshot` instead of fetching it into a privileged workflow.",
        );
    }
    return textOutput(result.stdout, "`git rev-parse`").trim();
}

// --- sources -----------------------------------------------------------------------------------------------

export type Side = "base" | "head";

/** A read-only view of the base and the head: directories on both sides, documents at the head. */
export interface Source {
    readonly baseLabel: string;
    readonly headLabel: string;
    changedPaths(): string[];
    existsDir(side: Side, path: string): boolean;
    /** Names directly under a directory; empty when the directory is absent. */
    entries(side: Side, path: string): string[];
    /** A change document's text at the head, or null when the head has no such document. */
    headText(path: string): string | null;
}

/**
 * Reads the two commits out of a local git object store. Use it where the head is already trusted or already
 * present: a developer's clone, or an unprivileged pull-request check that checks the head out the ordinary way.
 */
export class GitSource implements Source {
    readonly shas: Record<Side, string>;
    readonly baseLabel: string;
    readonly headLabel: string;

    constructor(readonly root: string, base: string, head: string) {
        this.shas = { base: resolve(root, base, "--base"), head: resolve(root, head, "--head") };
        this.baseLabel = base;
        this.headLabel = head;
    }

    /** `[type, path]` records of `git ls-tree`, which lists nothing, and exits 0, for an absent path. */
    ls(side: Side, path: string, recursive = false): [string, string][] {
        const out = git(this.root, ["ls-tree", "-z", ...(recursive ? ["-r"] : []), this.shas[side], "--", path]);
        return out.split("\0").filter((record) => record !== "").map((record) => {
            const tab = record.indexOf("\t");
            return [record.slice(0, tab).trim().split(/\s+/)[1], record.slice(tab + 1)];
        });
    }

    changedPaths(): string[] {
        const range = `${this.shas.base}...${this.shas.head}`;
        const result = gitRaw(this.root, ["diff", "-z", "--name-only", "--no-renames", range]);
        if (result.code !== 0) {
            throw new Failure(
                `\`git diff ${this.baseLabel}...${this.headLabel}\` failed (${result.stderr.trim()}); the merge ` +
                    "base must be reachable, so fetch with full history (fetch-depth 0) rather than a shallow clone.",
            );
        }
        return textOutput(result.stdout, "`git diff`").split("\0").filter((path) => path !== "");
    }

    existsDir(side: Side, path: string): boolean {
        const records = this.ls(side, path);
        return records.length === 1 && records[0][0] === "tree" && records[0][1] === path;
    }

    entries(side: Side, path: string): string[] {
        return this.ls(side, `${path}/`).map(([, child]) => child.slice(child.lastIndexOf("/") + 1));
    }

    headText(path: string): string | null {
        const records = this.ls("head", path);
        if (!(records.length === 1 && records[0][0] === "blob" && records[0][1] === path)) return null;
        return git(this.root, ["show", `${this.shas.head}:${path}`]);
    }
}

/** The snapshot document the `snapshot` command writes and SnapshotSource replays. */
export interface Snapshot {
    schema: string;
    repo: string;
    pull_request: number;
    changes_dir: string;
    changed_paths: string[];
    base: { sha: string; dirs: string[] };
    head: { sha: string; dirs: string[]; paths: string[]; files: Record<string, string> };
    bytes: number;
}

/**
 * Reads a snapshot document built from the GitHub REST API. Use it in every privileged workflow
 * (`pull_request_target`): the head's bytes arrive as data to parse, so no object authored by
 * the request ever reaches the runner's git store, and nothing it carries is ever executed.
 */
export class SnapshotSource implements Source {
    readonly baseLabel: string;
    readonly headLabel: string;
    private readonly changed: string[];
    private readonly dirs: Record<Side, string[]>;
    private readonly paths: Record<Side, string[]>;
    private readonly texts: Record<string, string>;

    constructor(path: string, changesDir: string) {
        let doc: unknown;
        try {
            doc = JSON.parse(STRICT_UTF8.decode(Deno.readFileSync(path)));
        } catch (error) {
            throw new Failure(`cannot read the snapshot ${path}: ${(error as Error).message}`);
        }
        const problem = snapshotProblem(doc, changesDir);
        if (problem) {
            throw new Failure(
                `snapshot ${path}: ${problem}; rebuild it with the snapshot command and the same --changes-dir.`,
            );
        }
        const snapshot = doc as Snapshot;
        this.changed = snapshot.changed_paths;
        this.dirs = { base: snapshot.base.dirs, head: snapshot.head.dirs };
        this.paths = { base: [], head: snapshot.head.paths };
        this.texts = snapshot.head.files;
        this.baseLabel = snapshot.base.sha;
        this.headLabel = snapshot.head.sha;
    }

    changedPaths(): string[] {
        return this.changed;
    }

    existsDir(side: Side, path: string): boolean {
        return this.dirs[side].includes(path) || this.paths[side].some((p) => p.startsWith(`${path}/`));
    }

    entries(side: Side, path: string): string[] {
        const prefix = `${path}/`;
        const names: string[] = [];
        for (const item of [...this.dirs[side], ...this.paths[side]]) {
            if (item.startsWith(prefix)) {
                const name = item.slice(prefix.length).split("/")[0];
                if (!names.includes(name)) names.push(name);
            }
        }
        return names;
    }

    headText(path: string): string | null {
        // The snapshot holds every change document the head has, so a missing key is a missing document.
        return Object.hasOwn(this.texts, path) ? this.texts[path] : null;
    }
}

/** What makes a loaded snapshot document unusable for this changes directory, or null. */
export function snapshotProblem(doc: unknown, changesDir: string): string | null {
    if (!isObject(doc) || doc.schema !== SNAPSHOT_SCHEMA) return `it is not a ${SNAPSHOT_SCHEMA} document`;
    if (doc.changes_dir !== changesDir) {
        return `it was built for the changes directory ${repr(doc.changes_dir)}, not ${repr(changesDir)}`;
    }
    for (const [keys, kind] of SNAPSHOT_KEYS) {
        let value: unknown = doc;
        for (const key of keys) value = isObject(value) && Object.hasOwn(value, key) ? value[key] : undefined;
        if (!isKind(value, kind)) return `\`${keys.join(".")}\` is missing or not ${JSON_KINDS[kind]}`;
        const items = Array.isArray(value) ? value : isObject(value) ? Object.values(value) : [];
        if (items.some((item) => typeof item !== "string")) return `\`${keys.join(".")}\` holds a non-string`;
    }
    return null;
}

// --- github rest api ---------------------------------------------------------------------------------------

export type FetchFn = (url: string, init: RequestInit) => Promise<Response>;
type Sleep = (ms: number) => Promise<void>;

/** A tree entry as the git trees endpoint lists it. */
export interface TreeEntry {
    path: string;
    type: string;
    sha: string;
    size?: number;
}

/**
 * The few REST reads the snapshot needs. A fork's head SHA resolves from the base repository, so nothing here
 * needs the fork. `maxCalls` bounds what one request can spend of the repository's shared REST budget.
 */
export class Api {
    readonly apiUrl: string;
    calls = 0;

    constructor(
        readonly repo: string,
        apiUrl: string,
        private readonly auth: string,
        readonly maxCalls: number,
        private readonly fetchFn: FetchFn,
        private readonly sleep: Sleep,
    ) {
        this.apiUrl = apiUrl.replace(/\/+$/, "");
    }

    /**
     * One GET whose body must be JSON of `kind`; anything else fails naming the endpoint. `keys` are the fields
     * the caller reads, of the object or of every item of the list; a missing one fails here.
     */
    async get(endpoint: string, kind: Kind, keys: string[] = []): Promise<unknown> {
        const body = await this.read(endpoint);
        let data: unknown;
        try {
            // json.loads(bytes) read a UTF-8 byte-order mark as the encoding's signature, not as text.
            data = JSON.parse(STRICT_UTF8.decode(body).replace(/^\ufeff/, ""));
        } catch (error) {
            throw new Failure(`GET ${endpoint} did not return JSON: ${(error as Error).message}`);
        }
        if (!isKind(data, kind)) {
            throw new Failure(
                `GET ${endpoint} returned ${jsonKind(data)} where the API documents ${JSON_KINDS[kind]}; ` +
                    "nothing is reported from a partial snapshot.",
            );
        }
        const what = Array.isArray(data) ? "an entry" : "an object";
        for (const item of Array.isArray(data) ? data : [data]) {
            const missing = keys.filter((key) => !isObject(item) || !Object.hasOwn(item, key));
            if (missing.length) {
                throw new Failure(
                    `GET ${endpoint} returned ${what} without \`${missing.join("`, `")}\`: ${brief(item)}; ` +
                        "nothing is reported from a partial snapshot.",
                );
            }
        }
        return data;
    }

    /** The response body; a rate limit, a server error, or a network error is retried, nothing else. */
    private async read(endpoint: string): Promise<Uint8Array> {
        const headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": `Bearer ${this.auth}`,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "spec-changes",
        };
        for (let attempt = 1;; attempt++) {
            if (this.calls >= this.maxCalls) {
                throw new Failure(
                    `the snapshot reached the --max-calls cap (${this.maxCalls} API requests) at GET ${endpoint}; ` +
                        "nothing is reported from a partial snapshot. Split the request, or raise the cap deliberately.",
                );
            }
            this.calls++;
            let response: Response;
            try {
                response = await this.fetchFn(`${this.apiUrl}/${endpoint}`, {
                    headers,
                    signal: AbortSignal.timeout(30_000),
                });
            } catch (error) {
                // fetch rejects with a TypeError on a network failure and a DOMException on the timeout; anything
                // else (a missing --allow-net grant among them) is not transient and propagates.
                if (!(error instanceof TypeError || error instanceof DOMException)) throw error;
                if (attempt === ATTEMPTS) {
                    throw new Failure(`cannot reach ${this.apiUrl} for GET ${endpoint}: ${error.message}`);
                }
                await this.sleep(2 ** attempt * 1000);
                continue;
            }
            if (response.ok) return new Uint8Array(await response.arrayBuffer());
            if (!TRANSIENT_STATUSES.includes(response.status) || attempt === ATTEMPTS) {
                const detail = (await response.text()).slice(0, 200).replaceAll("\n", " ");
                const hint = HTTP_HINTS[response.status] ?? "the API refused the read";
                throw new Failure(
                    `GET ${endpoint} answered HTTP ${response.status}: ${detail}. Likely cause: ${hint}.`,
                );
            }
            await response.body?.cancel();
            await this.sleep(2 ** attempt * 1000);
        }
    }

    /** The base and head SHAs and the count of changed files the request reports. */
    async pullHeads(number: number): Promise<[string, string, number]> {
        const endpoint = `repos/${this.repo}/pulls/${number}`;
        const pull = await this.get(endpoint, "object", ["base", "head", "changed_files"]) as Record<string, unknown>;
        const shas: string[] = [];
        for (const side of ["base", "head"]) {
            const value = pull[side];
            if (!isObject(value) || typeof value.sha !== "string") {
                throw new Failure(
                    `GET ${endpoint} returned a \`${side}\` without a \`sha\`: ${brief(value)}; ` +
                        "nothing is reported from a partial snapshot.",
                );
            }
            shas.push(value.sha);
        }
        const count = pull.changed_files;
        if (typeof count !== "number" || !Number.isInteger(count)) {
            throw new Failure(
                `GET ${endpoint} returned \`changed_files\` ${brief(count)}, not a count; ` +
                    "nothing is reported from a partial snapshot.",
            );
        }
        return [shas[0], shas[1], count];
    }

    /**
     * The request's paths; a listing shorter than the request's `changed_files` fails. The endpoint lists at
     * most 3000 files however it is paged, and a larger request comes back cut short without an error: only the
     * count the pull request reports shows the cut.
     */
    async pullFiles(number: number, maxFiles: number, changedFiles: number): Promise<string[]> {
        const endpoint = `repos/${this.repo}/pulls/${number}/files`;
        const paths: string[] = [];
        let listed = 0;
        for (let page = 1;; page++) {
            const batch = await this.get(`${endpoint}?per_page=100&page=${page}`, "list", ["filename"]) as Record<
                string,
                unknown
            >[];
            listed += batch.length;
            for (const item of batch) {
                paths.push(item.filename as string);
                // `git diff --no-renames` reports a rename as its old path plus its new one, so the git source
                // sees both. The API reports it as one entry carrying `previous_filename`; keeping only
                // `filename` would hide a record moved out of its directory from every privileged job while the
                // unprivileged check still saw it.
                if (Object.hasOwn(item, "previous_filename")) paths.push(item.previous_filename as string);
            }
            if (batch.length < 100 || paths.length > maxFiles) break;
        }
        if (paths.length > maxFiles) {
            throw new Failure(
                `the request touches at least ${paths.length} files, over the --max-files cap ` +
                    `(${maxFiles}); split the request, or raise the cap deliberately.`,
            );
        }
        if (listed < changedFiles) {
            throw new Failure(
                `GET ${endpoint} listed ${listed} of the request's ${changedFiles} changed files (the endpoint ` +
                    "lists at most 3000); nothing is reported from a partial snapshot. Split the request.",
            );
        }
        return paths;
    }

    /** The entries of one tree, named `path` in messages; a truncated listing fails. */
    async tree(sha: string, path: string, recursive = false): Promise<TreeEntry[]> {
        const endpoint = `repos/${this.repo}/git/trees/${pyQuote(sha, "")}`;
        const data = await this.get(endpoint + (recursive ? "?recursive=1" : ""), "object", ["truncated", "tree"]) as {
            truncated: unknown;
            tree: unknown;
        };
        if (!Array.isArray(data.tree)) {
            throw new Failure(
                `GET ${endpoint} returned a \`tree\` that is not a list: ${brief(data.tree)}; ` +
                    "nothing is reported from a partial snapshot.",
            );
        }
        const entries: unknown[] = data.tree;
        const bad = entries.find((e) =>
            !isObject(e) || !["path", "type", "sha"].every((key) => Object.hasOwn(e, key)) ||
            (e.type === "blob" && !Object.hasOwn(e, "size"))
        );
        if (bad !== undefined) {
            throw new Failure(
                `GET ${endpoint} listed an entry of ${path} without its path, type, sha, or a blob's size: ` +
                    `${brief(bad)}; nothing is reported from a partial snapshot.`,
            );
        }
        if (data.truncated) {
            throw new Failure(
                `the tree of ${path} came back truncated (GET ${endpoint}); nothing is reported from a partial ` +
                    "snapshot. Narrow the changes directory or archive older changes.",
            );
        }
        return entries as TreeEntry[];
    }

    async blobText(sha: string, path: string): Promise<string> {
        const endpoint = `repos/${this.repo}/git/blobs/${pyQuote(sha, "")}`;
        const data = await this.get(endpoint, "object", ["encoding", "content"]) as Record<string, unknown>;
        if (data.encoding !== "base64") {
            throw new Failure(`GET ${endpoint} for ${path} returned encoding ${repr(data.encoding)}, not base64.`);
        }
        if (typeof data.content !== "string") {
            throw new Failure(`GET ${endpoint} for ${path} returned content that is not a string.`);
        }
        try {
            return STRICT_UTF8.decode(b64decode(data.content));
        } catch (error) {
            throw new Failure(
                `${path} is not UTF-8 text (${(error as Error).message}); nothing is reported from a partial snapshot.`,
            );
        }
    }
}

/**
 * The entries of a directory at a commit, or null when the directory does not exist there. Each read after the
 * first is of a tree the previous listing named, so a failed read is an error and never an absent directory.
 */
async function listingAt(api: Api, commit: string, path: string): Promise<TreeEntry[] | null> {
    let entries = await api.tree(commit, `the commit ${commit}`);
    const walked: string[] = [];
    for (const segment of path.split("/")) {
        walked.push(segment);
        const entry = entries.find((e) => e.path === segment && e.type === "tree");
        if (entry === undefined) return null;
        entries = await api.tree(entry.sha, walked.join("/"));
    }
    return entries;
}

/** Tree SHAs of the directories under the changes directory and its archive, by path, in listing order. */
async function changeDirs(api: Api, commit: string, changesDir: string): Promise<Map<string, string>> {
    const shas = new Map<string, string>();
    const entries = await listingAt(api, commit, changesDir);
    if (entries === null) return shas; // a project that has no changes directory at this commit
    for (const entry of entries) if (entry.type === "tree") shas.set(`${changesDir}/${entry.path}`, entry.sha);
    const archive = `${changesDir}/archive`;
    const archiveSha = shas.get(archive);
    if (archiveSha !== undefined) {
        for (const entry of await api.tree(archiveSha, archive)) {
            if (entry.type === "tree") shas.set(`${archive}/${entry.path}`, entry.sha);
        }
    }
    return shas;
}

/** Whether a path inside a change directory is a document the commands read. */
function isChangeDoc(relative: string): boolean {
    return CHANGE_DOCS.includes(relative) || (relative.startsWith("specs/") && relative.endsWith("/spec.md"));
}

export interface Caps {
    maxFiles: number;
    maxFileBytes: number;
    maxTotalBytes: number;
}

/**
 * Read the request's related changes into a document the snapshot source replays. Only the documents the
 * commands read are fetched, and a snapshot that would be partial — a cap reached, a truncated tree, an
 * undecodable document, a directory name that is not a plain change name — fails instead of being reported on:
 * a label or a status derived from half the request is worse than no answer.
 */
export async function buildSnapshot(api: Api, number: number, changesDir: string, caps: Caps): Promise<Snapshot> {
    const [baseSha, headSha, changedFiles] = await api.pullHeads(number);
    const changed = await api.pullFiles(number, caps.maxFiles, changedFiles);
    const base = await changeDirs(api, baseSha, changesDir);
    const head = await changeDirs(api, headSha, changesDir);

    const paths: string[] = [];
    const files: Record<string, string> = {};
    let total = 0;
    for (const name of changeNames(changed, changesDir)) {
        if (!SAFE_NAME.test(name)) {
            throw new Failure(
                `the request touches ${changesDir}/${repr(name)}, which is not a plain change name; ` +
                    "nothing is reported from a partial snapshot.",
            );
        }
        const path = headDir([...head.keys()], changesDir, name);
        if (path === null) continue; // removed at the head: nothing to read
        for (const entry of await api.tree(head.get(path)!, path, true)) {
            if (entry.type !== "blob") continue;
            const full = `${path}/${entry.path}`;
            paths.push(full);
            if (!isChangeDoc(entry.path)) continue;
            const size = entry.size as number;
            if (size > caps.maxFileBytes) {
                throw new Failure(
                    `${full} is ${size} bytes, over the --max-file-bytes cap (${caps.maxFileBytes}); ` +
                        "nothing is reported from a partial snapshot.",
                );
            }
            if (total + size > caps.maxTotalBytes) {
                throw new Failure(
                    `the documents reached the --max-total-bytes cap (${caps.maxTotalBytes}) at ${full}; ` +
                        "nothing is reported from a partial snapshot.",
                );
            }
            files[full] = await api.blobText(entry.sha, full);
            total += size;
        }
    }

    return {
        schema: SNAPSHOT_SCHEMA,
        repo: api.repo,
        pull_request: number,
        changes_dir: changesDir,
        changed_paths: changed,
        base: { sha: baseSha, dirs: [...base.keys()] },
        head: { sha: headSha, dirs: [...head.keys()], paths: paths.sort(cmpCodePoints), files },
        bytes: total,
    };
}

// --- related changes ---------------------------------------------------------------------------------------

export function changeNames(paths: string[], changesDir: string): string[] {
    const prefix = `${changesDir}/`;
    const names = new Set<string>();
    for (const path of paths) {
        if (!path.startsWith(prefix)) continue;
        const parts = path.slice(prefix.length).split("/");
        if (parts[0] === "archive" && parts.length >= 3) names.add(parts[1].replace(ARCHIVE_DATE, ""));
        else if (parts[0] !== "archive" && parts.length >= 2) names.add(parts[0]);
    }
    return [...names].sort(cmpCodePoints);
}

/** The directory a change has in a listing: active first, else its latest archive entry. */
export function headDir(dirs: string[], changesDir: string, name: string): string | null {
    if (dirs.includes(`${changesDir}/${name}`)) return `${changesDir}/${name}`;
    return maxString(dirs.filter((d) => d.startsWith(`${changesDir}/archive/`) && archivedName(d) === name));
}

function archivedName(path: string): string {
    return path.slice(path.lastIndexOf("/") + 1).replace(ARCHIVE_DATE, "");
}

export type State = "active" | "archived" | "removed";

function stateAt(source: Source, side: Side, changesDir: string, name: string): [State, string | null] {
    const active = `${changesDir}/${name}`;
    if (source.existsDir(side, active)) return ["active", active];
    const matches = source.entries(side, `${changesDir}/archive`).filter((e) => e.replace(ARCHIVE_DATE, "") === name);
    if (matches.length) return ["archived", `${changesDir}/archive/${maxString(matches)}`];
    return ["removed", null];
}

export interface Tasks {
    done: number;
    open: number;
    open_tasks: string[];
    progress: "not-started" | "in-progress" | "done";
}

/** Progress from a task list; a change without one has not started. */
export function taskCounts(text: string | null): Tasks {
    if (text === null) return { done: 0, open: 0, open_tasks: [], progress: "not-started" };
    const done = [...text.matchAll(DONE_TASK)].length;
    const openTasks = [...text.matchAll(OPEN_TASK)].map((m) => pyStrip(m[1]) || "(untitled task)");
    let progress: Tasks["progress"];
    if (!openTasks.length && done > 0) progress = "done";
    else if (done === 0) progress = "not-started";
    else progress = "in-progress";
    return { done, open: openTasks.length, open_tasks: openTasks, progress };
}

export interface Change {
    name: string;
    state: State;
    path: string | null;
    at_base: State;
    skip_specs: boolean;
    tasks: Tasks;
}

export function relatedChanges(source: Source, changesDir: string): Change[] {
    return changeNames(source.changedPaths(), changesDir).map((name) => {
        const [state, path] = stateAt(source, "head", changesDir, name);
        const [baseState] = stateAt(source, "base", changesDir, name);
        const marker = path ? source.headText(`${path}/.openspec.yaml`) : null;
        return {
            name,
            state,
            path,
            at_base: baseState,
            skip_specs: Boolean(marker && SKIP_SPECS.test(marker)),
            tasks: taskCounts(path ? source.headText(`${path}/tasks.md`) : null),
        };
    });
}

/** The named changes, or all of them when none is named; a name the request does not touch is a bad argument. */
export function select(changes: Change[], wanted: string[]): Change[] {
    const byName = new Map(changes.map((c) => [c.name, c]));
    const unknown = wanted.filter((name) => !byName.has(name));
    if (unknown.length) {
        const related = [...byName.keys()].map((name) => codeSpan(name)).join(", ") || "none";
        throw new UsageError(
            `not a change this request touches: ${unknown.map((name) => codeSpan(name)).join(", ")}. ` +
                `Related changes: ${related}.`,
        );
    }
    return wanted.length ? wanted.map((name) => byName.get(name)!) : changes;
}

// --- rendering ---------------------------------------------------------------------------------------------

function longestRun(text: string): number {
    let longest = 0;
    for (const match of text.matchAll(/`+/g)) longest = Math.max(longest, match[0].length);
    return longest;
}

/** Inline code that request-authored text cannot break out of (no links, mentions, or markup). */
export function codeSpan(text: string, inTable = false): string {
    text = pySplit(text).join(" ");
    if (inTable) text = text.replaceAll("|", "\\|"); // a table splits cells before it parses code spans
    const ticks = "`".repeat(longestRun(text) + 1);
    const pad = !text || text.startsWith("`") || text.endsWith("`") ? " " : "";
    return `${ticks}${pad}${text}${pad}${ticks}`;
}

export function statusMarkdown(changes: Change[]): string {
    if (!changes.length) return NO_CHANGE;
    const lines = ["| Change | State | Tasks | Progress |", "|---|---|---|---|"];
    for (const c of changes) {
        const t = c.tasks;
        lines.push(`| ${codeSpan(c.name, true)} | ${c.state} | ${t.done} done, ${t.open} open | ${t.progress} |`);
    }
    for (const c of changes) {
        if (c.tasks.open_tasks.length) {
            lines.push("");
            lines.push(`Open tasks of ${codeSpan(c.name)} (first ten):`);
            for (const task of c.tasks.open_tasks.slice(0, 10)) {
                lines.push(`- ${codeSpan(cpLen(task) > 120 ? cpSlice(task, 117) + "..." : task)}`);
            }
        }
    }
    return lines.join("\n") + "\n";
}

export function desiredLabels(changes: Change[]): string[] {
    const live = changes.filter((c) => c.state !== "removed");
    if (!live.length) return [];
    const archived = live.every((c) => c.state === "archived") ? ARCHIVED_LABELS[1] : ARCHIVED_LABELS[0];
    const progressSet = new Set(live.map((c) => c.tasks.progress));
    let progress: string = PROGRESS_LABELS[1];
    if (progressSet.size === 1 && progressSet.has("done")) progress = PROGRESS_LABELS[2];
    else if (progressSet.size === 1 && progressSet.has("not-started")) progress = PROGRESS_LABELS[0];
    return [archived, progress];
}

// --- commands ----------------------------------------------------------------------------------------------

/** Where the commands write and what they read from the environment; tests replace it. */
export interface Io {
    out(text: string): void;
    err(text: string): void;
    env(name: string): string | undefined;
    fetch: FetchFn;
    sleep: Sleep;
}

function writeAll(stream: { writeSync(p: Uint8Array): number }, text: string): void {
    let bytes = new TextEncoder().encode(text);
    while (bytes.length > 0) bytes = bytes.subarray(stream.writeSync(bytes));
}

export const realIo: Io = {
    out: (text) => writeAll(Deno.stdout, text),
    err: (text) => writeAll(Deno.stderr, text),
    env: (name) => Deno.env.get(name),
    fetch: (url, init) => fetch(url, init),
    sleep: (ms) => new Promise((done) => setTimeout(done, ms)),
};

const COMMANDS = ["snapshot", "status", "check", "archive", "labels"] as const;
type CommandName = typeof COMMANDS[number];

/** The parsed command line, with the original's defaults. */
export interface Options {
    command: CommandName;
    root: string;
    changesDir: string;
    base?: string;
    head?: string;
    snapshot?: string;
    change: string[];
    json: boolean;
    all: boolean;
    draft: boolean;
    shape: string;
    noValidate: boolean;
    openspec: string;
    dryRun: boolean;
    current: string;
    taxonomy: boolean;
    repo: string;
    pr: number;
    out?: string;
    apiUrl: string;
    maxFiles: number;
    maxFileBytes: number;
    maxTotalBytes: number;
    maxCalls: number;
}

/** The head source the options name, or null when they name none. */
function openSource(opts: Options, root: string, changesDir: string): Source | null {
    if (opts.snapshot && (opts.base || opts.head)) {
        throw new UsageError("--snapshot reads the head on its own; do not pass --base/--head with it.");
    }
    if (opts.snapshot) return new SnapshotSource(opts.snapshot, changesDir);
    if (opts.base && opts.head) return new GitSource(root, opts.base, opts.head);
    if (opts.base || opts.head) {
        throw new UsageError("--base and --head go together; pass both, or pass --snapshot instead.");
    }
    return null;
}

function needSource(opts: Options, root: string, changesDir: string): Source {
    const source = openSource(opts, root, changesDir);
    if (source === null) {
        throw new UsageError(
            `${opts.command} needs a head source: --base with --head, or --snapshot from the snapshot command.`,
        );
    }
    return source;
}

function runOpenspec(root: string, executable: string, args: string[], io: Io): void {
    let result: Deno.CommandOutput;
    try {
        // The CLI's progress output is relayed to stderr so that `--json` on stdout stays parseable.
        result = new Deno.Command(executable, {
            args,
            cwd: root,
            env: { OPENSPEC_NO_UPDATE_CHECK: "1" },
            stdin: "inherit",
            stdout: "piped",
            stderr: "inherit",
        }).outputSync();
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) {
            throw new Failure(
                `\`${executable}\` is not installed; install the OpenSpec CLI pinned in the justfile (\`just setup\`).`,
            );
        }
        if (error instanceof Deno.errors.NotCapable) {
            throw new Failure(
                `\`${executable}\` is not a command this script may run (its shebang grants --allow-run=git,openspec); ` +
                    `run it as \`deno run --allow-run=git,${executable} ...\` with the shebang's other grants.`,
            );
        }
        throw error;
    }
    io.err(new TextDecoder().decode(result.stdout).replace(/\r\n?/g, "\n"));
    if (result.code !== 0) throw new Failure(`\`${executable} ${args.join(" ")}\` exited ${result.code}.`);
}

async function cmdSnapshot(opts: Options, _root: string, changesDir: string, io: Io): Promise<number> {
    const auth = io.env("GH_TOKEN") || io.env("GITHUB_TOKEN");
    if (!auth) throw new Failure("snapshot needs a GitHub token in GH_TOKEN or GITHUB_TOKEN.");
    const api = new Api(opts.repo, opts.apiUrl, auth, opts.maxCalls, io.fetch, io.sleep);
    const caps = { maxFiles: opts.maxFiles, maxFileBytes: opts.maxFileBytes, maxTotalBytes: opts.maxTotalBytes };
    const doc = await buildSnapshot(api, opts.pr, changesDir, caps);
    const text = pyJson(doc);
    if (!opts.out) {
        io.out(text + "\n");
        return 0;
    }
    Deno.writeTextFileSync(opts.out, text + "\n");
    io.err(
        `snapshot of ${api.repo}#${opts.pr} at ${doc.head.sha.slice(0, 7)}: ` +
            `${
                Object.keys(doc.head.files).length
            } file(s), ${doc.bytes} byte(s), ${api.calls} API call(s) → ${opts.out}\n`,
    );
    return 0;
}

function cmdStatus(opts: Options, root: string, changesDir: string, io: Io): number {
    const source = needSource(opts, root, changesDir);
    const changes = select(relatedChanges(source, changesDir), opts.change);
    if (opts.json) {
        io.out(pyJson({ base: source.baseLabel, head: source.headLabel, changes }) + "\n");
    } else {
        io.out(statusMarkdown(changes));
    }
    return 0;
}

function isDir(path: string): boolean {
    try {
        return Deno.statSync(path).isDirectory;
    } catch (error) {
        // Path.is_dir() read these as "not a directory" too.
        if (
            error instanceof Deno.errors.NotFound || error instanceof Deno.errors.NotADirectory ||
            error instanceof Deno.errors.FilesystemLoop
        ) return false;
        throw error;
    }
}

function cmdCheck(opts: Options, root: string, changesDir: string, io: Io): number {
    const source = openSource(opts, root, changesDir);
    if (opts.all && source !== null) {
        throw new UsageError("check --all reads the working tree; do not pass a head source with it.");
    }
    if (!opts.all && source === null) {
        throw new UsageError("check needs a head source (--base with --head, or --snapshot), or --all.");
    }
    if (!opts.noValidate) runOpenspec(root, opts.openspec, ["validate", "--all", "--strict", "--no-interactive"], io);
    const split = opts.shape === "split";
    const findings: string[] = [];
    if (source === null) {
        const live = `${root}/${changesDir}`;
        const unarchived = isDir(live)
            ? [...Deno.readDirSync(live)].map((e) => e.name).filter((name) =>
                name !== "archive" && isDir(`${live}/${name}`)
            ).sort(cmpCodePoints)
            : [];
        for (const name of unarchived) {
            // Under the combined shape the integration branch never holds an unarchived change. Under split it
            // holds every approved record whose implementation has not landed yet, by design.
            if (split) io.out(`warning: unarchived change ${name} on the integration branch (expected under split).\n`);
            else findings.push(`unarchived change ${name}: the integration branch holds only archived changes.`);
        }
    } else {
        for (const c of relatedChanges(source, changesDir)) {
            if (c.state === "active") {
                const message = `unarchived change ${c.name}: ${c.tasks.open} open task(s) — archive it in this ` +
                    "pull request once the deliberation on the finished implementation closes; " +
                    "the request stays red until then.";
                // The one request that may merge with an unarchived record is the split shape's specification
                // request: it carries the record and no implementation, and the implementation requests that
                // follow archive it. A request that implemented something is not that request, whatever its shape.
                if (opts.draft) {
                    io.out(`warning: ${message}\n`);
                } else if (split && c.tasks.done === 0) {
                    io.out(
                        `warning: unarchived change ${c.name} carries no implemented task; ` +
                            "allowed for a specification request under the split shape.\n",
                    );
                } else {
                    findings.push(message);
                }
            } else if (c.state === "archived" && c.tasks.open) {
                io.out(`warning: archived change ${c.name} still has ${c.tasks.open} open task(s).\n`);
            }
        }
    }
    for (const finding of findings) io.out(finding + "\n");
    return findings.length ? 1 : 0;
}

/** `str(Path(changesDir).parent)` for a relative path with a separator. */
function parentDir(changesDir: string): string {
    const parts = changesDir.split("/").filter((part) => part !== "" && part !== ".");
    return parts.slice(0, -1).join("/") || ".";
}

/**
 * Archive every complete related change, or none of them. This is the one command that writes, through the
 * OpenSpec CLI (`openspec archive <name> --yes`, plus `--skip-specs` for a change whose `.openspec.yaml` sets
 * `skip_specs: true`; flags verified against OpenSpec 1.12.0 on 2026-09-17). It must only run where the head is
 * already trusted, which is why it refuses any source but git. The CLI's own output goes to stderr, so `--json`
 * on stdout stays machine-readable.
 */
function cmdArchive(opts: Options, root: string, changesDir: string, io: Io): number {
    const source = needSource(opts, root, changesDir);
    if (!(source instanceof GitSource)) {
        throw new UsageError("archive edits the working tree, so it needs --base and --head, not a snapshot.");
    }
    if (git(root, ["rev-parse", "HEAD"]).trim() !== source.shas.head) {
        throw new UsageError(
            "archive edits the working tree, so --head must be the checked-out HEAD; check it out first.",
        );
    }
    // The CLI rewrites the main specs as well as the change records, and the documented follow-up stages the
    // whole tree the tool owns.
    const owned = changesDir.includes("/") ? parentDir(changesDir) : changesDir;
    if (git(root, ["status", "--porcelain", "--", owned]).trim()) {
        throw new Failure(`${owned} has uncommitted changes; commit or stash them before archiving.`);
    }
    const plan = {
        archived: [] as string[],
        skipped: [] as { name: string; reason: State }[],
        refused: [] as { name: string; open: number; tasks: string[] }[],
        validated: false,
    };
    const todo: Change[] = [];
    for (const c of relatedChanges(source, changesDir)) {
        if (c.state !== "active") {
            plan.skipped.push({ name: c.name, reason: c.state });
        } else if (c.tasks.open) {
            plan.refused.push({ name: c.name, open: c.tasks.open, tasks: c.tasks.open_tasks.slice(0, 5) });
        } else if (c.tasks.done === 0) {
            plan.refused.push({ name: c.name, open: 0, tasks: ["no ticked task in tasks.md"] });
        } else {
            todo.push(c);
        }
    }
    if (plan.refused.length) {
        if (opts.json) io.out(pyJson(plan) + "\n");
        for (const r of plan.refused) io.err(`refused ${r.name}: ${r.open} open task(s) — ${r.tasks.join("; ")}\n`);
        io.err("nothing archived: every related change must be complete first.\n");
        return 1;
    }
    for (const c of todo) {
        const cmd = ["archive", c.name, "--yes", ...(c.skip_specs ? ["--skip-specs"] : [])];
        if (opts.dryRun) {
            io.out(`would run: ${opts.openspec} ${cmd.join(" ")}\n`);
            continue;
        }
        io.err(`archiving ${c.name}\n`);
        runOpenspec(root, opts.openspec, cmd, io);
        plan.archived.push(c.name);
    }
    if (todo.length && !opts.dryRun) {
        runOpenspec(root, opts.openspec, ["validate", "--all", "--strict", "--no-interactive"], io);
        plan.validated = true;
    }
    if (opts.json) io.out(pyJson(plan) + "\n");
    else if (!todo.length) io.out("nothing to archive: every related change is already archived or removed.\n");
    return 0;
}

function cmdLabels(opts: Options, root: string, changesDir: string, io: Io): number {
    if (opts.taxonomy) {
        io.out(pyJson({ managed: MANAGED_LABELS }) + "\n");
        return 0;
    }
    const desired = desiredLabels(relatedChanges(needSource(opts, root, changesDir), changesDir));
    const current = opts.current.split(",").map(pyStrip).filter((label) => label !== "");
    const add = desired.filter((label) => !current.includes(label));
    const remove = current.filter((label) => MANAGED_LABELS.includes(label) && !desired.includes(label));
    if (opts.json) {
        io.out(pyJson({ desired, add, remove }) + "\n");
    } else {
        io.out(`desired: ${desired.join(" ") || "(none)"}\n`);
        io.out(`add: ${add.join(" ") || "(none)"}\n`);
        io.out(`remove: ${remove.join(" ") || "(none)"}\n`);
    }
    return 0;
}

// --- argument parsing --------------------------------------------------------------------------------------

const SOURCE_HELP = `head source (one of):
  --base REF            base ref, read with git plumbing (with --head)
  --head REF            head ref, read with git plumbing (with --base)
  --snapshot FILE       snapshot file from the snapshot command; the only source a privileged workflow may use
`;

const HELP: Record<CommandName | "main", string> = {
    main: `usage: spec_changes.ts [-h] [--root ROOT] [--changes-dir CHANGES_DIR] <command> ...

Resolve, check, archive, and label the OpenSpec changes a pull request touches.

commands:
  snapshot              build a head snapshot from the GitHub REST API (no checkout, no fetch)
  status                print a progress table (markdown) for the related changes
  check                 strict validation plus the unarchived-change rule
  archive               archive every complete related change through the OpenSpec CLI
  labels                compute the archive-axis and progress-axis labels

options (before the command):
  -h, --help            show this help message and exit
  --root ROOT           git work tree holding the changes directory (default: .)
  --changes-dir CHANGES_DIR
                        changes directory relative to --root (default: openspec/changes)

Run \`spec_changes.ts <command> --help\` for a command's options.
Examples: scripts/spec_changes.ts status --base origin/main --head HEAD; scripts/spec_changes.ts snapshot --repo
owner/name --pr 12 --out snapshot.json; scripts/spec_changes.ts labels --snapshot snapshot.json --current
'spec/unarchived'
Exit codes: 0 success; 1 a failure, a finding, or a refusal; 2 bad arguments.
`,
    snapshot: `usage: spec_changes.ts snapshot [-h] --repo REPO --pr PR [--out OUT] [--api-url API_URL]
                                [--max-files N] [--max-file-bytes N] [--max-total-bytes N] [--max-calls N]

Build a head snapshot from the GitHub REST API (no checkout, no fetch). Needs GH_TOKEN or GITHUB_TOKEN.

options:
  -h, --help            show this help message and exit
  --repo REPO           OWNER/NAME of the repository the request targets
  --pr PR               pull request number
  --out OUT             write the snapshot here (default: stdout)
  --api-url API_URL     REST API root (default: $GITHUB_API_URL or https://api.github.com)
  --max-files N         cap on touched files (default ${MAX_FILES})
  --max-file-bytes N    per-file byte cap (default ${MAX_FILE_BYTES})
  --max-total-bytes N   total byte cap (default ${MAX_TOTAL_BYTES})
  --max-calls N         cap on API requests (default ${MAX_CALLS})
`,
    status: `usage: spec_changes.ts status [-h] [--base BASE] [--head HEAD] [--snapshot SNAPSHOT] [--change CHANGE]
                              [--json]

Print a progress table (markdown) for the related changes.

options:
  -h, --help            show this help message and exit
  --change CHANGE       limit to this change (repeatable)
  --json                print the changes as JSON

${SOURCE_HELP}`,
    check: `usage: spec_changes.ts check [-h] [--base BASE] [--head HEAD] [--snapshot SNAPSHOT] [--all] [--draft]
                             [--shape {combined,split}] [--no-validate] [--openspec OPENSPEC]

Strict validation plus the unarchived-change rule.

options:
  -h, --help            show this help message and exit
  --all                 fail on any change outside archive/ (integration branch)
  --draft               report unarchived related changes as warnings
  --shape {combined,split}
                        the change request shape the project's contract records (default: combined). Under
                        split, a related change with no implemented task is allowed unarchived: that is the
                        specification request, which the implementation requests archive later.
  --no-validate         skip the OpenSpec validator
  --openspec OPENSPEC   OpenSpec CLI executable (default: openspec)

${SOURCE_HELP}`,
    archive: `usage: spec_changes.ts archive [-h] [--base BASE] [--head HEAD] [--snapshot SNAPSHOT] [--dry-run]
                               [--openspec OPENSPEC] [--json]

Archive every complete related change through the OpenSpec CLI, or none of them. Needs --base and --head,
with --head checked out.

options:
  -h, --help            show this help message and exit
  --dry-run             print the plan; change nothing
  --openspec OPENSPEC   OpenSpec CLI executable (default: openspec)
  --json                print the plan as JSON

${SOURCE_HELP}`,
    labels: `usage: spec_changes.ts labels [-h] [--base BASE] [--head HEAD] [--snapshot SNAPSHOT] [--current CURRENT]
                              [--taxonomy] [--json]

Compute the archive-axis and progress-axis labels.

options:
  -h, --help            show this help message and exit
  --current CURRENT     comma-separated labels currently on the request
  --taxonomy            print the managed label names
  --json                print the plan as JSON

${SOURCE_HELP}`,
};

const SOURCE_OPTIONS = ["base", "head", "snapshot"];
const SPECS: Record<CommandName, { string: string[]; boolean: string[]; multiple?: string[] }> = {
    snapshot: {
        string: ["repo", "pr", "out", "api-url", "max-files", "max-file-bytes", "max-total-bytes", "max-calls"],
        boolean: [],
    },
    status: { string: SOURCE_OPTIONS, boolean: ["json"], multiple: ["change"] },
    check: { string: [...SOURCE_OPTIONS, "shape", "openspec"], boolean: ["all", "draft", "no-validate"] },
    archive: { string: [...SOURCE_OPTIONS, "openspec"], boolean: ["dry-run", "json"] },
    labels: { string: [...SOURCE_OPTIONS, "current"], boolean: ["taxonomy", "json"] },
};

function positive(name: string, value: string): number {
    const text = pyStrip(value);
    if (!/^[+-]?\d+(?:_\d+)*$/.test(text)) {
        throw new UsageError(`argument --${name}: invalid positive value: ${repr(value)}`);
    }
    const number = Number(text.replaceAll("_", ""));
    if (number <= 0) throw new UsageError(`argument --${name}: ${value} is not a positive number`);
    return number;
}

function choice(name: string, value: string, choices: readonly string[]): string {
    if (!choices.includes(value)) {
        throw new UsageError(
            `argument --${name}: invalid choice: ${repr(value)} (choose from ${choices.map(repr).join(", ")})`,
        );
    }
    return value;
}

/** Parses the command line into options, or returns the help text to print. */
export function parseCommandLine(argv: string[], env: (name: string) => string | undefined): Options | string {
    let root = ".";
    let changesDir = "openspec/changes";
    let i = 0;
    // The global options go before the command, as the original's parser required.
    for (; i < argv.length; i++) {
        const arg = argv[i];
        if (arg === "-h" || arg === "--help") return HELP.main;
        const match = /^--(root|changes-dir)(?:=([\s\S]*))?$/.exec(arg);
        if (match) {
            const value = match[2] ?? argv[++i];
            if (value === undefined) throw new UsageError(`--${match[1]} needs a value`);
            if (match[1] === "root") root = value;
            else changesDir = value;
            continue;
        }
        if (arg.startsWith("-")) throw new UsageError(`unrecognized arguments: ${arg}`);
        break;
    }
    const command = argv[i];
    if (command === undefined) {
        throw new UsageError("a command is required: snapshot, status, check, archive, or labels");
    }
    if (!(COMMANDS as readonly string[]).includes(command)) {
        throw new UsageError(
            `argument <command>: invalid choice: ${repr(command)} (choose from ${COMMANDS.map(repr).join(", ")})`,
        );
    }
    const name = command as CommandName;
    const spec = SPECS[name];
    const parsed = parseArgs(argv.slice(i + 1), {
        string: spec.string,
        boolean: [...spec.boolean, "help"],
        multiple: spec.multiple,
        alias: { h: "help" },
    });
    if (parsed.flags.help) return HELP[name];
    if (parsed.positional.length) throw new UsageError(`unrecognized arguments: ${parsed.positional.join(" ")}`);
    const v = parsed.values;
    const flag = (key: string) => parsed.flags[key] ?? false;
    if (name === "snapshot") {
        const missing = ["repo", "pr"].filter((key) => v[key] === undefined).map((key) => `--${key}`);
        if (missing.length) throw new UsageError(`the following arguments are required: ${missing.join(", ")}`);
        if (!SAFE_REPO.test(v.repo!)) {
            throw new UsageError(`argument --repo: ${repr(v.repo)} is not an OWNER/NAME pair`);
        }
    }
    const number = (key: string, fallback: number) => v[key] === undefined ? fallback : positive(key, v[key]!);
    return {
        command: name,
        root,
        changesDir,
        base: v.base,
        head: v.head,
        snapshot: v.snapshot,
        change: name === "status" ? parsed.lists.change : [],
        json: flag("json"),
        all: flag("all"),
        draft: flag("draft"),
        shape: v.shape === undefined ? "combined" : choice("shape", v.shape, ["combined", "split"]),
        noValidate: flag("no-validate"),
        openspec: v.openspec ?? "openspec",
        dryRun: flag("dry-run"),
        current: v.current ?? "",
        taxonomy: flag("taxonomy"),
        repo: v.repo ?? "",
        pr: number("pr", 0),
        out: v.out,
        apiUrl: v["api-url"] ?? (name === "snapshot" ? env("GITHUB_API_URL") ?? "https://api.github.com" : ""),
        maxFiles: number("max-files", MAX_FILES),
        maxFileBytes: number("max-file-bytes", MAX_FILE_BYTES),
        maxTotalBytes: number("max-total-bytes", MAX_TOTAL_BYTES),
        maxCalls: number("max-calls", MAX_CALLS),
    };
}

/** `Path(root).resolve()`: absolute, symlinks resolved where the path exists. */
function resolveRoot(root: string): string {
    const absolute = root.startsWith("/") ? root : `${Deno.cwd()}/${root}`;
    try {
        return Deno.realPathSync(absolute);
    } catch (error) {
        if (!(error instanceof Deno.errors.NotFound)) throw error;
    }
    const parts: string[] = [];
    for (const part of absolute.split("/")) {
        if (part === "" || part === ".") continue;
        if (part === "..") parts.pop();
        else parts.push(part);
    }
    return "/" + parts.join("/");
}

type Handler = (opts: Options, root: string, changesDir: string, io: Io) => number | Promise<number>;
const HANDLERS: Record<CommandName, Handler> = {
    snapshot: cmdSnapshot,
    status: cmdStatus,
    check: cmdCheck,
    archive: cmdArchive,
    labels: cmdLabels,
};

export async function main(argv: string[], io: Io = realIo): Promise<number> {
    let opts: Options;
    try {
        const parsed = parseCommandLine(argv, io.env);
        if (typeof parsed === "string") {
            io.out(parsed);
            return 0;
        }
        opts = parsed;
    } catch (error) {
        if (!(error instanceof UsageError)) throw error;
        io.err(`spec_changes.ts: error: ${error.message}\nRun \`spec_changes.ts --help\` for usage.\n`);
        return 2;
    }
    const root = resolveRoot(opts.root);
    const changesDir = opts.changesDir.replace(/^\/+|\/+$/g, "");
    try {
        return await HANDLERS[opts.command](opts, root, changesDir, io);
    } catch (error) {
        if (error instanceof UsageError) {
            io.err(`error: ${error.message}\n`);
            return 2;
        }
        if (error instanceof Failure) {
            io.err(`error: ${error.message}\n`);
            return 1;
        }
        throw error;
    }
}

if (import.meta.main) Deno.exit(await main(Deno.args));
