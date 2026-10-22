#!/usr/bin/env -S deno run --no-prompt --allow-run=git
// Commit safety gate for this repository.
//
// Staged mode (the default; the first pre-commit hook and `just commit-gate`) checks the index of the
// repository the process runs in: there is staged content, `git config user.email` is a GitHub or GitLab
// noreply address, and the staged diff adds no private key marker, no credential-like `key = value`
// assignment, and no email address outside the noreply and RFC 2606 example domains.
//
// Range mode (`--range <base>..<head>`; `pr / policy` runs it from the base's copy) checks every commit
// in `git rev-list <base>..<head>` on its own: the lines that commit adds (not the net diff, because a
// rebase merge lands every commit on `main`) against the same patterns; its author email against the
// noreply domains unless `--fork` is given (a fork's commits are squashed); and its message for
// agent-attribution trailers — a Co-Authored-By line naming an AI model or tool or a model vendor's
// noreply address, and a "Generated with" line naming an AI tool. Human co-authors, the co-author GitHub
// adds for an applied review suggestion (Copilot Autofix), and other trailers pass.
//
// Usage:
//   scripts/check_commit_safety.ts                          staged mode
//   scripts/check_commit_safety.ts --range <base>..<head> [--fork]
//
// Exit status: 0 passed, 1 findings, 2 usage error or a git failure.
import { parseArgs } from "./lib/args.ts";
import { CommandError, eprint, run, runChecked, UsageError } from "./lib/repo.ts";

const USAGE = `usage: check_commit_safety.ts [--range <base>..<head> [--fork]]

Without --range, checks the staged change: staged content exists, git config user.email is a
GitHub/GitLab noreply address, and the added lines hold no private key, credential, or personal email.

  --range <base>..<head>  check each commit in the range: its own added lines, its author email,
                          and agent-attribution trailers in its message
  --fork                  with --range: skip the author email check (fork commits are squashed)
  -h, --help              show this help`;

const ANONYMOUS_EMAILS = [
    /^[^@\s]+@users\.noreply\.github\.com$/i,
    /^[^@\s]+@noreply\.gitlab\.com$/i,
    /^[^@\s]+@users\.noreply\.gitlab\.com$/i,
];

// Python's `\b` is Unicode-aware; JavaScript's is ASCII-only even with the `u` flag. This lookaround pair
// is the boundary the original patterns were written against: a word character on exactly one side.
const W = String.raw`[\p{L}\p{N}_]`;
const B = String.raw`(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`;

const EMAIL_RE = new RegExp(String.raw`${B}[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}${B}`, "giu");
const ENV_VAR_RE = /^\$[A-Za-z_][A-Za-z0-9_]*$/;
const BRACED_ENV_VAR_RE = /^\$\{[A-Za-z_][A-Za-z0-9_]*(?:\})?$/;
const PRIVATE_KEY_RE = /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----/;
const CREDENTIAL_RE = new RegExp( // pragma: allowlist secret
    String.raw`${B}(api[_-]?key|apikey|secret|token|passw(?:or)?d|passwd|credential)${B}` + // pragma: allowlist secret
        String.raw`\s*[:=]\s*["']?([^"'\s,}]+)`,
    "iu",
);
const SUPPRESSION_RE = new RegExp(
    String.raw`pragma: allowlist secret|gitleaks:allow|detect-secrets:disable|#\s*nosec${B}`,
    "u",
);

const ALLOWED_EMAIL_DOMAINS = new Set([
    "example.com",
    "example.org",
    "example.net",
    "users.noreply.github.com",
    "noreply.gitlab.com",
    "users.noreply.gitlab.com",
]);

const PLACEHOLDERS = new Set(["...", "REDACTED", "redacted", "changeme", "example"]);

// Agent attribution. A co-author name is an AI model or tool when it names one of these. A name starting
// with "Claude" counts when it is the whole name, continues with a separator (`claude[bot]`, `claude-code`,
// `Claude.ai`), or is followed by a tool or model word (`Claude Code`, `Claude AI`, `Claude Opus 5`), so a
// human named Claude (`Claude Monet`) passes.
const AI_COAUTHOR_RE =
    /\b(?:(?:chat)?gpt(?:[-\d]|\b)|(?:openai|codex|gemini|copilot|opencode|anthropic|cursor agent)\b)|^claude(?:$|[-_.[]|\s+(?:(?:bot|code|agent|ai|opus|sonnet|haiku|instant)\b|\d))/i;
// Co-authors GitHub itself adds when a review suggestion is applied in its interface; they stay allowed.
const GITHUB_FEATURE_COAUTHOR_RE = /^copilot autofix\b/i;
// Domains whose no-reply address only a model or tool uses as a co-author.
const VENDOR_DOMAINS = new Set(["anthropic.com", "openai.com", "google.com", "cursor.com"]);
const NOREPLY_LOCAL_RE = /^no-?reply$/i;
const COAUTHOR_RE = /^\s*co-authored-by:\s*(.*)$/i;
const GENERATED_RE = /^\W*generated (?:with|by)\b/iu;
const AI_TOOL_RE =
    /\b(?:(?:chat)?gpt(?:[-\d]|\b)|(?:claude|anthropic|openai|codex|gemini|copilot|opencode|cursor|aider)\b)|\u{1F916}/iu;

export interface Finding {
    message: string;
}

/** Whether an email is a GitHub or GitLab noreply address. */
export function isAnonymousEmail(email: string): boolean {
    return ANONYMOUS_EMAILS.some((pattern) => pattern.test(email));
}

/** Whether a credential value is a placeholder rather than a secret. */
export function looksLikePlaceholder(value: string): boolean {
    const v = value.trim().replace(/^["']+|["']+$/g, "");
    return v === "" || ENV_VAR_RE.test(v) || BRACED_ENV_VAR_RE.test(v) || v.startsWith("<") || PLACEHOLDERS.has(v);
}

/** Findings for the added lines of a unified diff (`--unified=0`, `b/` destination prefix). */
export function scanDiff(diff: string): string[] {
    const findings: string[] = [];
    let currentFile = "";
    for (const raw of diff.split("\n")) {
        const line = raw.endsWith("\r") ? raw.slice(0, -1) : raw;
        if (line.startsWith("+++ b/")) {
            currentFile = line.slice("+++ b/".length);
            continue;
        }
        if (!line.startsWith("+") || line.startsWith("+++")) continue;
        const content = line.slice(1);
        if (SUPPRESSION_RE.test(content)) continue;

        if (PRIVATE_KEY_RE.test(content)) findings.push(`${currentFile}: added private key marker`);

        const credential = CREDENTIAL_RE.exec(content); // pragma: allowlist secret
        if (credential && !looksLikePlaceholder(credential[2])) {
            findings.push(`${currentFile}: added credential-like assignment for \`${credential[1]}\``);
        }

        for (const match of content.matchAll(EMAIL_RE)) {
            const email = match[0];
            const domain = email.slice(email.lastIndexOf("@") + 1).toLowerCase();
            if (!ALLOWED_EMAIL_DOMAINS.has(domain)) findings.push(`${currentFile}: added personal email \`${email}\``);
        }
    }
    return findings;
}

/** The agent-attribution lines of a commit message. */
export function attributionLines(message: string): string[] {
    const found: string[] = [];
    for (const raw of message.split("\n")) {
        const line = raw.trim();
        const coauthor = COAUTHOR_RE.exec(line);
        if (coauthor) {
            const value = coauthor[1].trim();
            const lt = value.indexOf("<");
            const name = (lt === -1 ? value : value.slice(0, lt)).trim();
            const address = lt === -1 ? "" : value.slice(lt + 1).replace(/>.*$/, "").trim();
            const at = address.lastIndexOf("@");
            const vendorNoreply = at > 0 && NOREPLY_LOCAL_RE.test(address.slice(0, at)) &&
                VENDOR_DOMAINS.has(address.slice(at + 1).toLowerCase());
            if (GITHUB_FEATURE_COAUTHOR_RE.test(name)) continue;
            if (AI_COAUTHOR_RE.test(name) || vendorNoreply) found.push(line);
            continue;
        }
        if (GENERATED_RE.test(line) && AI_TOOL_RE.test(line)) found.push(line);
    }
    return found;
}

/** Splits `<base>..<head>`; a three-dot range, an empty side, or an option-like side is a usage error. */
export function parseRange(range: string): [string, string] {
    const parts = range.split("..");
    if (parts.length !== 2 || parts[0] === "" || parts[1] === "" || parts[1].startsWith(".")) {
        throw new UsageError(`--range must be <base>..<head>, got \`${range}\``);
    }
    if (parts[0].startsWith("-") || parts[1].startsWith("-")) {
        throw new UsageError(`--range sides must be revisions, got \`${range}\``);
    }
    return [parts[0], parts[1]];
}

const DIFF_FLAGS = [
    "--unified=0",
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--src-prefix=a/",
    "--dst-prefix=b/",
];

function git(args: string[], cwd: string): string {
    return runChecked("git", args, { cwd });
}

function stagedFiles(cwd: string): string[] {
    return git(["diff", "--cached", "--name-only", "-z"], cwd).split("\0").filter((item) => item !== "");
}

function gitUserEmail(cwd: string): string {
    const out = run("git", ["config", "--get", "user.email"], { cwd });
    if (out.code === 0) return out.stdout.trim();
    if (out.code === 1) return "";
    throw new CommandError(`\`git config --get user.email\` exited ${out.code}: ${out.stderr.trim()}`);
}

/** Staged mode: errors for the index of the repository at `cwd`; prints the author email line to stdout. */
export function checkStaged(cwd: string): string[] {
    const errors: string[] = [];
    if (stagedFiles(cwd).length === 0) {
        errors.push("No staged changes. Stage one atomic logical change before committing.");
    }
    const email = gitUserEmail(cwd);
    if (!email) {
        errors.push("git config user.email is empty. Configure an anonymous email.");
    } else if (isAnonymousEmail(email)) {
        console.log(`Author email: ${email} (anonymous)`);
    } else {
        errors.push(
            "git config user.email is not a GitHub/GitLab anonymous address. Use a noreply address " +
                "(<id>+<user>@users.noreply.github.com).",
        );
    }
    errors.push(...scanDiff(git(["diff", "--cached", ...DIFF_FLAGS], cwd)));
    return errors;
}

function resolveCommit(ref: string, cwd: string): string {
    const out = run("git", ["rev-parse", "--verify", "--quiet", "--end-of-options", `${ref}^{commit}`], { cwd });
    const sha = out.stdout.trim();
    if (out.code !== 0 || !/^[0-9a-f]{40}([0-9a-f]{24})?$/.test(sha)) {
        throw new CommandError(
            `\`git rev-parse ${ref}\` found no commit (exit ${out.code}); fetch it first ` +
                "(a full-history checkout, or git fetch of the pull request head)",
        );
    }
    return sha;
}

/** Range mode: errors for each commit in `<base>..<head>`; prints the commit count to stdout. */
export function checkRange(range: string, fork: boolean, cwd: string): string[] {
    const [baseRef, headRef] = parseRange(range);
    const base = resolveCommit(baseRef, cwd);
    const head = resolveCommit(headRef, cwd);
    const commits = git(["rev-list", "--reverse", `${base}..${head}`], cwd).split("\n").filter((l) => l !== "");
    for (const sha of commits) {
        if (!/^[0-9a-f]{40}([0-9a-f]{24})?$/.test(sha)) {
            throw new CommandError(`\`git rev-list ${base}..${head}\` printed \`${sha}\`, not a commit id`);
        }
    }
    console.log(`Checked ${commits.length} commit(s) in ${range}.`);
    if (commits.length === 0) return [`No commits in ${range}; check the base and head.`];

    const errors: string[] = [];
    for (const sha of commits) {
        const short = sha.slice(0, 12);
        const meta = git(["show", "-s", "--format=%ae%x00%B", sha], cwd);
        const nul = meta.indexOf("\0");
        if (nul === -1) throw new CommandError(`\`git show -s ${sha}\` printed no author/message separator`);
        const email = meta.slice(0, nul).trim();
        const message = meta.slice(nul + 1);
        if (!fork && !isAnonymousEmail(email)) {
            errors.push(`${short}: author email is not a GitHub/GitLab noreply address; rewrite the commit's author`);
        }
        for (const line of attributionLines(message)) {
            errors.push(`${short}: agent-attribution trailer \`${line}\`; remove it from the message`);
        }
        const diff = git(["show", "--format=", "--diff-merges=first-parent", ...DIFF_FLAGS, sha], cwd);
        for (const finding of scanDiff(diff)) errors.push(`${short}: ${finding}`);
    }
    return errors;
}

/** Runs the gate in the repository at `cwd` (default: the process's working directory). */
export function main(argv: string[], cwd = "."): number {
    try {
        const args = parseArgs(argv, { string: ["range"], boolean: ["fork", "help"], alias: { h: "help" } });
        if (args.flags.help) {
            console.log(USAGE);
            return 0;
        }
        if (args.positional.length > 0) throw new UsageError(`unexpected argument ${args.positional[0]}`);
        if (args.flags.fork && args.values.range === undefined) throw new UsageError("--fork needs --range");

        const errors = args.values.range === undefined
            ? checkStaged(cwd)
            : checkRange(args.values.range, args.flags.fork, cwd);
        if (errors.length > 0) {
            eprint("\nCommit safety gate failed:");
            for (const error of errors) eprint(`- ${error}`);
            return 1;
        }
        console.log("Commit safety gate passed.");
        return 0;
    } catch (error) {
        if (error instanceof UsageError) {
            eprint(`${USAGE}\n\nerror: ${error.message}`);
            return 2;
        }
        if (error instanceof CommandError) {
            eprint(`error: ${error.message}`);
            return 2;
        }
        throw error;
    }
}

if (import.meta.main) Deno.exit(main(Deno.args));
