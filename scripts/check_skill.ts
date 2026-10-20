#!/usr/bin/env -S deno run --no-prompt --allow-read=.
// Lints one or more skill directories against the Agent Skills spec.
//
// Usage: scripts/check_skill.ts <skill-dir> [<skill-dir>...]   (called by `just check-skill`)
//
// ERROR   = spec hard-rule violation or repo rule; fails the check.
// WARNING = spec recommendation or quality guidance; reported, never fails.
// Findings go to stderr; the closing OK line goes to stdout.
//
// Lengths count Unicode code points and lines split as Python's str.splitlines() does, so the
// limits read the same as the Agent Skills spec states them. Token estimates use a chars/4
// heuristic, reasonable for this repository's English-only skills.
//
// scripts/validate_skills.ts imports this module's checks and runs them over every skill.
//
// Exit codes: 0 = clean or warnings only, 1 = at least one error,
// 2 = bad arguments (e.g. a path that is not a directory).
import { parseArgs } from "./lib/args.ts";
import { eprint, ROOT, UsageError } from "./lib/repo.ts";

export const NAME_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;
export const NAME_MAX = 64;
export const DESC_MAX = 1024;
export const DESC_WARN = 900;
export const COMPAT_MAX = 500;
export const BODY_LINES_WARN = 500;
export const BODY_TOKENS_WARN = 5000;

export const ERROR = "ERROR";
export const WARNING = "WARNING";
export type Severity = typeof ERROR | typeof WARNING;
export type Finding = [Severity, string];

const USAGE = `usage: scripts/check_skill.ts <skill-dir> [<skill-dir>...]

Lint skill directories against the Agent Skills spec and this repository's rules.
ERROR fails the check (exit 1); WARNING is reported and never fails.
Exit 2 on bad arguments.`;

// --- Python string semantics the limits and the frontmatter reader rely on ---

// The characters Python's str.isspace() accepts, which str.strip() removes.
const PY_SPACE = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "g");
const PY_RSTRIP = new RegExp(`[${PY_SPACE}]+$`);

/** Python's str.strip(): removes leading and trailing whitespace. */
export function pyStrip(text: string): string {
    return text.replace(PY_STRIP, "");
}

/** Python's str.rstrip(). */
export function pyRstrip(text: string): string {
    return text.replace(PY_RSTRIP, "");
}

/** Python's str.splitlines(): every line boundary Python knows, no trailing empty line. */
export function splitLines(text: string): string[] {
    if (text === "") return [];
    // deno-lint-ignore no-control-regex -- \v, \f, and \x1c-\x1e are line boundaries in Python
    const lines = text.split(/\r\n|[\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]/);
    if (lines[lines.length - 1] === "") lines.pop();
    return lines;
}

/** Length in Unicode code points, as Python's len(str). */
export function codePoints(text: string): number {
    let count = 0;
    for (const _ of text) count++;
    return count;
}

// --- filesystem helpers shared with validate_skills.ts ---

const UTF8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/**
 * Decodes a file's bytes as UTF-8 the way Python's text mode reads them: a byte-order mark is
 * kept, invalid UTF-8 throws, and every `\r\n` or lone `\r` becomes `\n` (universal newlines),
 * so line-anchored patterns still match in a CRLF file.
 */
export function decodeUtf8(bytes: Uint8Array, path: string): string {
    try {
        return UTF8.decode(bytes).replace(/\r\n?/g, "\n");
    } catch (error) {
        if (!(error instanceof TypeError)) throw error;
        throw new Error(`${relTo(ROOT, path)}: not valid UTF-8 (${error.message}); save the file as UTF-8`);
    }
}

/** Reads a file as UTF-8 text with universal newlines (see decodeUtf8). */
export function readUtf8(path: string): string {
    return decodeUtf8(Deno.readFileSync(path), path);
}

/** Whether a path is a directory, following symlinks (Path.is_dir()). */
export function isDir(path: string): boolean {
    try {
        return Deno.statSync(path).isDirectory;
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
    }
}

/** Whether a path is a regular file, following symlinks (Path.is_file()). */
export function isFile(path: string): boolean {
    try {
        return Deno.statSync(path).isFile;
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
    }
}

/** Whether a path is a symlink, without following it (Path.is_symlink()). */
export function isSymlink(path: string): boolean {
    try {
        return Deno.lstatSync(path).isSymlink;
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
    }
}

/** The entries of a directory as full paths, sorted by name. */
export function listDir(dir: string): string[] {
    return [...Deno.readDirSync(dir)].map((entry) => entry.name).sort().map((name) => `${dir}/${name}`);
}

/** The last component of a path. */
export function baseName(path: string): string {
    return path.replace(/\/+$/, "").split("/").pop() ?? path;
}

/** A path as pathlib prints it: no empty or `.` components, no trailing slash. */
export function normalizePath(path: string): string {
    const parts = path.split("/").filter((part) => part !== "" && part !== ".");
    const joined = parts.join("/");
    if (path.startsWith("/")) return "/" + joined;
    return joined === "" ? "." : joined;
}

/** A path relative to `root` when it lies inside it, else the path as pathlib prints it. */
export function relTo(root: string, path: string): string {
    const normal = normalizePath(path);
    if (normal === root) return ".";
    return normal.startsWith(root + "/") ? normal.slice(root.length + 1) : normal;
}

// --- the checks ---

/**
 * Returns [frontmatter fields, body]. Fields is null when frontmatter is missing or
 * unterminated. Handles single-line values and YAML block scalars (| and >, with optional
 * +/- chomping).
 */
export function parseFrontmatter(text: string): [Record<string, string> | null, string] {
    const lines = splitLines(text);
    if (lines.length === 0 || pyStrip(lines[0]) !== "---") return [null, text];
    const fields: Record<string, string> = {};
    let i = 1;
    while (i < lines.length) {
        const line = lines[i];
        if (pyStrip(line) === "---") return [fields, lines.slice(i + 1).join("\n")];
        const match = /^([A-Za-z][\p{L}\p{N}_-]*):\s*(.*)$/u.exec(line);
        if (match) {
            const key = match[1];
            const value = pyStrip(match[2]);
            if (["|", ">", "|-", ">-", "|+", ">+"].includes(value)) {
                const block: string[] = [];
                i++;
                while (i < lines.length && (/^[ \t]/.test(lines[i]) || pyStrip(lines[i]) === "")) {
                    if (pyStrip(lines[i]) === "---") break;
                    block.push(pyStrip(lines[i]));
                    i++;
                }
                fields[key] = block.filter((part) => part !== "").join(" ");
                continue;
            }
            fields[key] = value.replace(/^["']+|["']+$/g, "");
        }
        i++;
    }
    return [null, text]; // no closing --- found
}

/** Lints one skill directory. Paths in messages are relative to `root` when inside it. */
export function checkSkillDir(skillDir: string, root: string = ROOT): Finding[] {
    const findings: Finding[] = [];
    const rel = relTo(root, skillDir);
    const error = (message: string) => findings.push([ERROR, message]);
    const warn = (message: string) => findings.push([WARNING, message]);

    const skillMd = `${skillDir}/SKILL.md`;
    if (!isFile(skillMd)) {
        error(
            `${rel}: missing SKILL.md. A skill is defined by its SKILL.md; ` +
                `without it nothing can be discovered or loaded. Fix: create ` +
                `${rel}/SKILL.md with \`name\` and \`description\` frontmatter.`,
        );
        return findings;
    }

    const [fields, body] = parseFrontmatter(readUtf8(skillMd));
    if (fields === null) {
        error(
            `${rel}/SKILL.md: frontmatter missing or unterminated. Agents ` +
                `cannot read metadata without a \`---\` ... \`---\` block at the ` +
                `top. Fix: add frontmatter with \`name\` and \`description\`.`,
        );
        return findings;
    }
    const dirName = baseName(Deno.realPathSync(skillDir));

    // --- name (spec hard rules) ---
    const name = fields["name"] ?? "";
    if (!name) {
        error(
            `${rel}/SKILL.md: missing \`name\` in frontmatter. The spec ` +
                `requires it. Fix: add \`name: ${dirName}\`.`,
        );
    } else {
        if (!NAME_RE.test(name)) {
            error(
                `${rel}/SKILL.md: name \`${name}\` is invalid. The spec allows ` +
                    `lowercase a-z, digits, and single hyphens only (no ` +
                    `leading/trailing/consecutive hyphens, no uppercase, no ` +
                    `underscores or spaces). Fix: rename to kebab-case.`,
            );
        }
        const nameLength = codePoints(name);
        if (nameLength > NAME_MAX) {
            error(
                `${rel}/SKILL.md: name is ${nameLength} chars; the spec maximum is ${NAME_MAX}. ` +
                    `Fix: shorten the name.`,
            );
        }
        if (name !== dirName) {
            error(
                `${rel}/SKILL.md: name \`${name}\` does not match the ` +
                    `directory name \`${dirName}\`. The spec requires an exact ` +
                    `match; a mismatch breaks installation and discovery. ` +
                    `Fix: rename one of them.`,
            );
        }
    }

    // --- description (spec hard rules + quality guidance) ---
    const description = fields["description"] ?? "";
    const descLength = codePoints(description);
    if (!description) {
        error(
            `${rel}/SKILL.md: missing or empty \`description\`. It is the ` +
                `only text an agent reads before deciding to activate the ` +
                `skill, so without it the skill never triggers. Fix: add a ` +
                `third-person capability sentence plus 'Use when...' triggers.`,
        );
    } else if (descLength > DESC_MAX) {
        error(
            `${rel}/SKILL.md: description is ${descLength} chars; ` +
                `the spec maximum is ${DESC_MAX}. Fix: trim it below the ` +
                `limit, keeping the trigger phrases.`,
        );
    } else if (descLength > DESC_WARN) {
        warn(
            `${rel}/SKILL.md: description is ${descLength} chars, ` +
                `approaching the ${DESC_MAX}-char spec limit. Consider ` +
                `trimming before it becomes a hard failure.`,
        );
    }

    // --- compatibility (spec hard rule, optional field) ---
    const compatLength = codePoints(fields["compatibility"] ?? "");
    if (compatLength > COMPAT_MAX) {
        error(
            `${rel}/SKILL.md: compatibility is ${compatLength} chars; ` +
                `the spec maximum is ${COMPAT_MAX}. Fix: state only the ` +
                `specific environment requirements, or omit the field.`,
        );
    }

    // --- body size (spec recommendation) ---
    const bodyLines = splitLines(body).length;
    const bodyTokens = Math.floor(codePoints(body) / 4);
    if (bodyLines > BODY_LINES_WARN || bodyTokens > BODY_TOKENS_WARN) {
        warn(
            `${rel}/SKILL.md: body is ${bodyLines} lines / ~${bodyTokens} ` +
                `tokens (chars/4 heuristic); the spec recommends under ` +
                `${BODY_LINES_WARN} lines / ~${BODY_TOKENS_WARN} tokens. Move ` +
                `branch-specific material to references/ with precise load ` +
                `conditions (progressive disclosure).`,
        );
    }

    // --- repo rule: no README.md in a skill root ---
    if (exists(`${skillDir}/README.md`)) {
        error(
            `${rel}: contains README.md. Skill roots must not have a ` +
                `README; the catalog README describes skills. Fix: move the ` +
                `content into SKILL.md or the catalog README and delete it.`,
        );
    }

    // --- empty optional subdirectories (quality guidance) ---
    for (const sub of ["scripts", "references", "assets"]) {
        const subDir = `${skillDir}/${sub}`;
        if (isDir(subDir) && listDir(subDir).length === 0) {
            warn(
                `${rel}/${sub}/: empty directory. It misleads agents into ` +
                    `expecting files that don't exist. Fix: delete it until ` +
                    `there is a file to put in it.`,
            );
        }
    }

    return findings;
}

/** Whether a path exists, following symlinks (Path.exists()). */
function exists(path: string): boolean {
    try {
        Deno.statSync(path);
        return true;
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
    }
}

/**
 * Every skill directory in the repository: public skills under skills/<catalog>/, plus
 * project-only skills in .agents/skills/ (symlinks excluded — their targets are already covered).
 */
export function repoSkillDirs(root: string = ROOT): string[] {
    const dirs: string[] = [];
    const skillsRoot = `${root}/skills`;
    if (isDir(skillsRoot)) {
        for (const catalog of listDir(skillsRoot).filter(isDir)) {
            dirs.push(...listDir(catalog).filter(isDir));
        }
    }
    const agentsRoot = `${root}/.agents/skills`;
    if (isDir(agentsRoot)) {
        dirs.push(...listDir(agentsRoot).filter((p) => isDir(p) && !isSymlink(p)));
    }
    return dirs;
}

/** Prints the findings and the verdict; returns the exit code. */
export function report(findings: Finding[]): number {
    const errors = findings.filter(([severity]) => severity === ERROR);
    const warnings = findings.filter(([severity]) => severity === WARNING);
    for (const [severity, message] of findings) eprint(`${severity}: ${message}\n`);
    if (errors.length > 0) {
        eprint(
            `FAIL: ${errors.length} error(s), ${warnings.length} warning(s). ` +
                `Standards: .agents/knowledge/skill-quality.md. Re-run with ` +
                `\`just check-skill <dir>\` after fixing.`,
        );
        return 1;
    }
    if (warnings.length > 0) console.log(`OK with ${warnings.length} warning(s).`);
    else console.log("OK: all skill checks passed.");
    return 0;
}

export function main(argv: string[]): number {
    let paths: string[];
    try {
        const args = parseArgs(argv, { boolean: ["help"], alias: { h: "help" } });
        if (args.flags["help"]) {
            console.log(USAGE);
            return 0;
        }
        if (args.positional.length === 0) throw new UsageError("pass one or more skill directories.");
        paths = args.positional;
    } catch (error) {
        if (!(error instanceof UsageError)) throw error;
        eprint(USAGE);
        eprint(`ERROR: ${error.message}`);
        return 2;
    }
    try {
        for (const raw of paths) {
            if (!isDir(raw)) {
                eprint(`ERROR: ${raw} is not a directory.`);
                return 2;
            }
        }
        return report(paths.flatMap((raw) => checkSkillDir(raw)));
    } catch (error) {
        if (!(error instanceof Deno.errors.NotCapable)) throw error;
        eprint(
            `ERROR: ${error.message}. The read permission covers the working directory only; ` +
                `run this script from the repository root on skill directories inside it.`,
        );
        return 2;
    }
}

if (import.meta.main) Deno.exit(main(Deno.args));
