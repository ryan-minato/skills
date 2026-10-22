#!/usr/bin/env -S deno run --no-prompt --allow-read --allow-run=git,gh
// Checks a pull request against the PR template and the commit convention.
//
// The template (.github/PULL_REQUEST_TEMPLATE.md) is the single source of the required sections: its `## `
// headings are parsed here, never restated. `pr / policy` (.github/workflows/pr-policy.yml) runs this file
// from a checkout of the BASE commit, so a pull request cannot weaken the template or the script it is
// checked against. It runs beside the workflow's token, so it imports local modules only.
//
// Checks, in order:
//   1. every template heading appears verbatim in the body;
//   2. the related-work section carries `Closes #N` (or Fixes/Resolves) or `N/A — <reason>`;
//   3. the checklist line containing the security keyword ("secrets") exists;
//   4. commit subjects follow Conventional Commits — over BASE..HEAD for a branch in this repository,
//      over the pull request title for a fork;
//   5. for a ready (non-draft) pull request: the changes and validation sections are filled in (not
//      empty, not the template's reserved placeholder), the `Phase:` line reads `implementation`, every
//      checklist box is ticked, and a `Spec:` line names an OpenSpec change as a link
//      (`[openspec/changes/<slug>](https://github.com/…/openspec/changes/<slug>)` or the bare URL) or
//      reads `none — <reason>`.
// Pull requests opened by Dependabot skip checks 1–3 and 5; their commit subjects are still checked.
// Sections beyond the template's are allowed; only the template's headings are required.
//
// Usage:
//   ./scripts/check_pr_policy.ts --event "$GITHUB_EVENT_PATH"   # pr / policy
//   ./scripts/check_pr_policy.ts --pr 123                       # local dry run via gh
//   options: --template <path> (default .github/PULL_REQUEST_TEMPLATE.md)
//
// Findings go to stdout, one per line, each ending with the file to fix.
// Exit codes: 0 = compliant, 1 = findings, 2 = bad input, unreadable template, or a failed gh or git
// call (a shallow checkout cannot list the range).
import { parseArgs } from "./lib/args.ts";
import { CommandError, eprint, join, type Output, readText, rel, REPO, ROOT, run, UsageError } from "./lib/repo.ts";

const USAGE = `usage: check_pr_policy.ts (--event PATH | --pr NUMBER) [--template PATH]

Checks a pull request against the PR template and the commit convention.

  --event PATH      the pull_request event payload (GITHUB_EVENT_PATH)
  --pr NUMBER       pull request number to fetch with gh (local dry run)
  --template PATH   PR template path (default .github/PULL_REQUEST_TEMPLATE.md)
  -h, --help        show this help

Exit codes: 0 compliant, 1 findings, 2 bad input or a failed interface.`;

export const TEMPLATE = join(ROOT, ".github/PULL_REQUEST_TEMPLATE.md");
const SECURITY_KEYWORD = "secrets";
export const COMMIT_TYPES = "feat|fix|docs|refactor|chore|test|ci|perf|build|style|revert";

// The patterns below keep the Python original's `re` semantics on `str`, so the same body yields the
// same findings. ECMAScript differs in each piece, so each is spelled out (all patterns take the `u` flag):
//   \s, \S, strip()  Python's str.isspace() set: adds \x1c-\x1f and \x85, lacks U+FEFF;
//   \d               any Unicode decimal digit (\p{Nd}), not only 0-9;
//   \w, \b           Unicode letters, numbers, and `_`; \b is written as a lookaround over that class;
//   .                anything but \n (ECMAScript also stops at \r, U+2028, U+2029);
//   ^ $ (MULTILINE)  only \n starts or ends a line (ECMAScript's `m` also splits at \r, U+2028, U+2029);
//   $ (plain)        the end, or before a final \n;
//   IGNORECASE       Python also folds i to U+0130 and U+0131; `iu` folds U+017F (long s) to s as Python does.
// The classes follow the runtime's Unicode version; they differ from Python 3.12 (Unicode 15.0) only on
// code points that version leaves unassigned.
const WS = String.raw`\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000`;
const S = `[${WS}]`;
const NOT_S = `[^${WS}]`;
const WORD = String.raw`[\p{L}\p{N}_]`;
const DIGIT = String.raw`\p{Nd}`;
const DOT = String.raw`[^\n]`;
const LINE_START = String.raw`(?<=^|\n)`;
const LINE_END = String.raw`(?=\n|$)`;
const END = String.raw`(?=\n?$)`;
const pyRe = (source: string, flags = "") => new RegExp(source, `u${flags}`);

const SUBJECT_RE = pyRe(String.raw`^(${COMMIT_TYPES})(\([a-z0-9][a-z0-9, ._-]*\))?!?: [a-z0-9]${DOT}+${END}`);
const EXEMPT_SUBJECT_RE = pyRe(String.raw`^(Merge |Revert |fixup!|squash!)`);
const CLOSING_RE = pyRe(
    String.raw`(?<!${WORD})(close[sd]?|f[i\u0130\u0131]x(e[sd])?|resolve[sd]?)${S}+#${DIGIT}+(?!${WORD})`,
    "i",
);
const NO_ISSUE_RE = pyRe(String.raw`${LINE_START}N/A${S}+[—-]${S}+${NOT_S}`, "i");
// A change record is named as a clickable link to its directory on the branch
// (`[openspec/changes/<slug>](https://github.com/<owner>/<repo>/tree/<ref>/openspec/changes/<slug>)`)
// or as the bare URL. A bare path is refused: it is not clickable.
const SPEC_PATH = String.raw`openspec/changes/(archive/${DIGIT}{4}-${DIGIT}{2}-${DIGIT}{2}-)?[a-z0-9][a-z0-9-]*/?`;
const SPEC_URL = String.raw`https://github\.com/[\p{L}\p{N}_.-]+/[\p{L}\p{N}_.-]+/(tree|blob)/[^${WS})]+?/${SPEC_PATH}`;
const SPEC_RE = pyRe(
    String
        .raw`${LINE_START}Spec:${S}*(\[${SPEC_PATH}\]\(${SPEC_URL}\)|${SPEC_URL}|none${S}+[—-]${S}+${NOT_S}${DOT}*)${S}*${LINE_END}`,
);
const PHASE_RE = pyRe(`${LINE_START}Phase:${S}*(specification|implementation)${S}*${LINE_END}`);
// The template reserves the changes and validation sections until ready with a line that is entirely
// this placeholder.
const RESERVED_RE = pyRe(`${LINE_START}${S}*_Reserved:${DOT}*_${S}*${LINE_END}`);
const DASH_LINE_RE = pyRe(`${LINE_START}${S}*-${S}*${LINE_END}`, "g");
const COMMENT_RE = pyRe(String.raw`<!--[\s\S]*?-->`, "g");
const TEMPLATE_HEADING_RE = pyRe(`${LINE_START}## (${DOT}+?)${S}*${LINE_END}`, "g");
const HEADING_RE = pyRe(`^## (${DOT}+?)${S}*${END}`);
const CHECKBOX_RE = pyRe(String.raw`^${S}*- \[( |x|X)\]${S}*(${DOT}*)${END}`);
const STRIP_RE = pyRe(`^${S}+|${S}+$`, "g");
const BOT_AUTHORS = new Set(["dependabot[bot]"]);
// Headings the template must carry, matched by a keyword in the heading text.
const ROLES = ["related", "changes", "validation", "checklist"] as const;
type Role = typeof ROLES[number];

/** The pull request fields the checks read, from either source. */
export interface PullRequest {
    number: number;
    title: string;
    body: string;
    draft: boolean;
    fork: boolean;
    baseSha: string;
    headSha: string;
    author: string;
}

/** The event payload, the template, or gh output is unusable; main() reports it and exits 2. */
export class InputError extends Error {}

/** Runs gh with the given arguments; injectable so tests never reach the network. */
export type GhRunner = (args: string[]) => Output;
const defaultGh: GhRunner = (args) => run("gh", args);

/** Splits like Python's str.splitlines(): every line boundary it knows, no trailing empty line. */
export function splitLines(text: string): string[] {
    // deno-lint-ignore no-control-regex -- \x1c-\x1e are line boundaries to str.splitlines()
    const lines = text.split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]/);
    if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
    return lines;
}

/** Quotes a string the way Python's repr() does, so finding messages read as they always have. */
export function pyRepr(text: string): string {
    const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
    let out = quote;
    for (const ch of text) {
        const code = ch.codePointAt(0)!;
        if (ch === quote || ch === "\\") out += "\\" + ch;
        else if (ch === "\t") out += "\\t";
        else if (ch === "\n") out += "\\n";
        else if (ch === "\r") out += "\\r";
        else if (ch !== " " && /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u.test(ch)) {
            if (code < 0x100) out += "\\x" + code.toString(16).padStart(2, "0");
            else if (code < 0x10000) out += "\\u" + code.toString(16).padStart(4, "0");
            else out += "\\U" + code.toString(16).padStart(8, "0");
        } else out += ch;
    }
    return out + quote;
}

/** Strips like Python's str.strip(): the str.isspace() set, which differs from String.prototype.trim(). */
export function pyStrip(text: string): string {
    return text.replace(STRIP_RE, "");
}

export function stripComments(text: string): string {
    return text.replace(COMMENT_RE, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function field<T>(source: string, value: unknown, name: string, ok: (v: unknown) => v is T, want: string): T {
    if (!ok(value)) {
        throw new InputError(`${source}: ${name} is ${JSON.stringify(value) ?? "missing"}, expected ${want}`);
    }
    return value;
}

const isString = (v: unknown): v is string => typeof v === "string";
const isBool = (v: unknown): v is boolean => typeof v === "boolean";
const isInt = (v: unknown): v is number => Number.isInteger(v);
const isNullableString = (v: unknown): v is string | null => v === null || typeof v === "string";

/** Reads the fields the checks need from a pull_request event payload, checking their shape. */
export function parseEvent(payload: unknown, source: string): PullRequest {
    const pr = isRecord(payload) ? payload.pull_request : undefined;
    if (!isRecord(pr)) throw new InputError(`${source} is not a pull_request event payload`);
    const head = field(source, pr.head, "pull_request.head", isRecord, "an object");
    const base = field(source, pr.base, "pull_request.base", isRecord, "an object");
    // head.repo is null when the fork it came from has been deleted.
    const headRepo = head.repo === null
        ? null
        : field(source, head.repo, "pull_request.head.repo", isRecord, "an object");
    const user = pr.user === null ? null : field(source, pr.user, "pull_request.user", isRecord, "an object");
    return {
        number: field(source, pr.number, "pull_request.number", isInt, "an integer"),
        title: field(source, pr.title, "pull_request.title", isString, "a string"),
        body: field(source, pr.body, "pull_request.body", isNullableString, "a string or null") ?? "",
        draft: field(source, pr.draft, "pull_request.draft", isBool, "a boolean"),
        fork: headRepo === null
            ? false
            : field(source, headRepo.fork, "pull_request.head.repo.fork", isBool, "a boolean"),
        baseSha: field(source, base.sha, "pull_request.base.sha", isString, "a string"),
        headSha: field(source, head.sha, "pull_request.head.sha", isString, "a string"),
        author: user === null ? "" : field(source, user.login, "pull_request.user.login", isString, "a string"),
    };
}

export function loadEvent(path: string): PullRequest {
    let payload: unknown;
    try {
        payload = JSON.parse(readText(path));
    } catch (error) {
        if (
            error instanceof Deno.errors.NotFound || error instanceof Deno.errors.PermissionDenied ||
            error instanceof Deno.errors.IsADirectory || error instanceof SyntaxError
        ) {
            throw new InputError(`cannot read event payload ${path}: ${error.message}`);
        }
        throw error;
    }
    return parseEvent(payload, path);
}

const GH_FIELDS = "number,title,body,isDraft,isCrossRepository,baseRefOid,headRefOid,author";

/** Fetches the pull request with `gh pr view`, checking the exit status and the JSON shape. */
export function loadPrViaGh(number: number, gh: GhRunner = defaultGh): PullRequest {
    const args = ["pr", "view", String(number), "--repo", REPO, "--json", GH_FIELDS];
    const out = gh(args);
    const command = `gh ${args.join(" ")}`;
    if (out.code !== 0) throw new CommandError(`\`${command}\` exited ${out.code}: ${out.stderr.trim()}`);
    let data: unknown;
    try {
        data = JSON.parse(out.stdout);
    } catch (error) {
        throw new CommandError(
            `\`${command}\` printed no JSON (${(error as Error).message}): ${out.stdout.slice(0, 200)}`,
        );
    }
    if (!isRecord(data)) {
        throw new CommandError(`\`${command}\` printed ${out.stdout.slice(0, 200)}, expected an object`);
    }
    const source = `gh pr view ${number}`;
    const author = data.author === null ? null : field(source, data.author, "author", isRecord, "an object");
    return {
        number: field(source, data.number, "number", isInt, "an integer"),
        title: field(source, data.title, "title", isString, "a string"),
        body: field(source, data.body, "body", isString, "a string"),
        draft: field(source, data.isDraft, "isDraft", isBool, "a boolean"),
        fork: field(source, data.isCrossRepository, "isCrossRepository", isBool, "a boolean"),
        baseSha: field(source, data.baseRefOid, "baseRefOid", isString, "a string"),
        headSha: field(source, data.headRefOid, "headRefOid", isString, "a string"),
        author: author === null ? "" : field(source, author.login, "author.login", isString, "a string"),
    };
}

/** The template's `## ` headings and the heading that plays each role. */
export function templateHeadings(path: string): { headings: string[]; roles: Record<Role, string> } {
    let text: string;
    try {
        text = stripComments(readText(path));
    } catch (error) {
        if (
            error instanceof Deno.errors.NotFound || error instanceof Deno.errors.PermissionDenied ||
            error instanceof Deno.errors.IsADirectory
        ) {
            throw new InputError(`cannot read ${rel(path)}: ${error.message}`);
        }
        throw error;
    }
    const headings = [...text.matchAll(TEMPLATE_HEADING_RE)].map((m) => m[1]);
    if (headings.length === 0) throw new InputError(`${rel(path)} has no '## ' headings to enforce; fix the template`);
    const roles: Partial<Record<Role, string>> = {};
    for (const heading of headings) {
        const lower = heading.toLowerCase();
        for (const role of ROLES) {
            if (lower.includes(role) && !(role in roles)) roles[role] = heading;
        }
    }
    const missing = ROLES.filter((role) => !(role in roles));
    if (missing.length > 0) {
        throw new InputError(`${rel(path)} lacks a heading for: ${missing.join(", ")}; fix the template`);
    }
    return { headings, roles: roles as Record<Role, string> };
}

/** The body's `## ` sections; a repeated heading starts its section over, as the Python original did. */
export function bodySections(body: string): Map<string, string> {
    const sections = new Map<string, string>();
    let current: string | undefined;
    for (const line of splitLines(body)) {
        const match = HEADING_RE.exec(line);
        if (match) {
            current = match[1];
            sections.set(current, "");
        } else if (current !== undefined) {
            sections.set(current, sections.get(current) + line + "\n");
        }
    }
    return sections;
}

/** The subjects of BASE..HEAD, from the repository at `cwd` (the checkout the script runs in). */
export function commitSubjects(base: string, head: string, cwd: string = ROOT): string[] {
    const out = run("git", ["log", "--format=%s", `${base}..${head}`], { cwd });
    if (out.code !== 0) {
        throw new CommandError(
            `git log ${base}..${head} failed (shallow checkout or missing fetch?): exit ${out.code}: ${out.stderr.trim()}`,
        );
    }
    return splitLines(out.stdout).filter((line) => pyStrip(line) !== "");
}

export function checkSubject(subject: string, label: string, findings: string[]): void {
    if (EXEMPT_SUBJECT_RE.test(subject)) return;
    if (!SUBJECT_RE.test(subject)) {
        findings.push(
            `${label} ${pyRepr(subject)} is not a Conventional Commit subject ` +
                `(\`type(scope): subject\`, types ${COMMIT_TYPES}). Fix: rewrite it; see AGENTS.md Commits.`,
        );
    }
}

/** Every finding for the pull request; `subjects` lists BASE..HEAD (injectable for tests). */
export function check(
    pr: PullRequest,
    template: string,
    subjects: (base: string, head: string) => string[] = (base, head) => commitSubjects(base, head),
): string[] {
    const findings: string[] = [];
    const { headings, roles } = templateHeadings(template);
    const shown = rel(template);
    const body = stripComments(pr.body);
    const ready = !pr.draft;

    if (!BOT_AUTHORS.has(pr.author)) {
        const sections = bodySections(body);
        for (const heading of headings) {
            if (!sections.has(heading)) {
                findings.push(`PR body is missing the section '## ${heading}'. Fix: restore it from ${shown}.`);
            }
        }
        const related = sections.get(roles.related) ?? "";
        if (!(CLOSING_RE.test(related) || NO_ISSUE_RE.test(related))) {
            findings.push(
                `'## ${roles.related}' needs \`Closes #N\` (or Fixes/Resolves) or \`N/A — <reason>\`. ` +
                    "Fix: the PR description.",
            );
        }
        const checklist = splitLines(sections.get(roles.checklist) ?? "")
            .map((line) => CHECKBOX_RE.exec(line))
            .filter((m): m is RegExpExecArray => m !== null);
        if (!checklist.some((m) => m[2].toLowerCase().includes(SECURITY_KEYWORD))) {
            findings.push(
                `The checklist line containing '${SECURITY_KEYWORD}' is missing. ` +
                    `Fix: restore it from ${shown} and tick it.`,
            );
        }
        if (ready) {
            const filled: [Role, string][] = [
                ["changes", "list every touched file as a permalink to its commit"],
                ["validation", "record what you ran and each scenario's result"],
            ];
            for (const [role, hint] of filled) {
                const section = sections.get(roles[role]) ?? "";
                if (RESERVED_RE.test(section)) {
                    findings.push(
                        `'## ${roles[role]}' still holds the template's reserved line on a ready pull request. ` +
                            `Fix: remove it and ${hint} in the PR description.`,
                    );
                } else if (!pyStrip(section.replace(DASH_LINE_RE, ""))) {
                    findings.push(
                        `'## ${roles[role]}' is empty on a ready pull request. Fix: ${hint} in the PR description.`,
                    );
                }
            }
            const phase = PHASE_RE.exec(body);
            if (!phase || phase[1] !== "implementation") {
                findings.push(
                    "A ready pull request needs a `Phase: implementation` line. " +
                        "Fix: get the specification approved on the draft, then set the line.",
                );
            }
            for (const m of checklist) {
                if (m[1] === " ") {
                    findings.push(
                        `Checklist item not ticked on a ready pull request: ${pyRepr(m[2])}. ` +
                            "Fix: complete it and tick it.",
                    );
                }
            }
            if (!SPEC_RE.test(body)) {
                findings.push(
                    "A ready pull request needs a `Spec: [openspec/changes/<slug>](<its URL on the branch>)` line " +
                        "or `Spec: none — <reason>`. Fix: the PR description.",
                );
            }
        }
    }

    if (pr.fork) {
        checkSubject(pr.title, "Pull request title (fork PRs are squash-merged)", findings);
    } else {
        if (!(pr.baseSha && pr.headSha)) {
            throw new InputError("base and head SHAs are required for the commit-range check");
        }
        for (const subject of subjects(pr.baseSha, pr.headSha)) checkSubject(subject, "Commit subject", findings);
    }
    return findings;
}

export function main(argv: string[], gh: GhRunner = defaultGh): number {
    try {
        const args = parseArgs(argv, { string: ["event", "pr", "template"], boolean: ["help"], alias: { h: "help" } });
        if (args.flags.help) {
            console.log(USAGE);
            return 0;
        }
        if (args.positional.length > 0) throw new UsageError(`unexpected argument ${args.positional[0]}`);
        const { event, pr: prArg } = args.values;
        if ((event === undefined) === (prArg === undefined)) {
            throw new UsageError("exactly one of --event and --pr is required");
        }
        if (prArg !== undefined && !/^\d+$/.test(prArg)) {
            throw new UsageError(`--pr takes a pull request number, got ${JSON.stringify(prArg)}`);
        }
        const pr = event !== undefined ? loadEvent(event) : loadPrViaGh(Number(prArg), gh);
        const findings = check(pr, args.values.template ?? TEMPLATE);
        for (const finding of findings) console.log(finding);
        if (findings.length > 0) {
            eprint(`${findings.length} finding(s) for pull request #${pr.number}.`);
            return 1;
        }
        console.log(`Pull request #${pr.number} satisfies the template and commit policy.`);
        return 0;
    } catch (error) {
        if (error instanceof UsageError) {
            eprint(`${USAGE.split("\n")[0]}\ncheck_pr_policy: error: ${error.message}`);
            return 2;
        }
        if (error instanceof InputError || error instanceof CommandError) {
            eprint(`check_pr_policy: error: ${error.message}`);
            return 2;
        }
        throw error;
    }
}

if (import.meta.main) Deno.exit(main(Deno.args));
