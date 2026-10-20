// Shared helpers for the repository's management scripts. This module imports nothing remote, so a
// script that runs beside a write token (the privileged workflows in .agents/knowledge/github-checks.md)
// may import it and still load no code a lockfile would have to pin (.agents/knowledge/management-code.md).

/** The repository root: two levels above this file (scripts/lib/repo.ts). */
export const ROOT = decodeURIComponent(new URL("../../", import.meta.url).pathname).replace(/\/$/, "");

/** The GitHub repository this harness belongs to. */
export const REPO = "ryan-minato/skills";

/** A path relative to the repository root, for messages. */
export function rel(path: string): string {
    return path.startsWith(ROOT + "/") ? path.slice(ROOT.length + 1) : path;
}

/** Joins path segments with "/" without normalizing; the inputs are repository paths. */
export function join(...parts: string[]): string {
    return parts.filter((p) => p !== "").join("/").replace(/\/{2,}/g, "/");
}

/** Reads a UTF-8 file. */
export function readText(path: string): string {
    return Deno.readTextFileSync(path);
}

/** Whether a path exists (following symlinks). */
export function exists(path: string): boolean {
    try {
        Deno.statSync(path);
        return true;
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
    }
}

/** The result of an external command. */
export interface Output {
    code: number;
    stdout: string;
    stderr: string;
}

/** Runs a command and returns its exit status and output; the caller decides what a status means. */
export function run(cmd: string, args: string[], options: { cwd?: string; env?: Record<string, string> } = {}): Output {
    const result = new Deno.Command(cmd, {
        args,
        cwd: options.cwd ?? ROOT,
        env: options.env,
        stdin: "null",
        stdout: "piped",
        stderr: "piped",
    }).outputSync();
    const decoder = new TextDecoder();
    return { code: result.code, stdout: decoder.decode(result.stdout), stderr: decoder.decode(result.stderr) };
}

/** Runs a command with optional stdin and returns its exit status and output. */
export async function runAsync(
    cmd: string,
    args: string[],
    options: { cwd?: string; stdin?: string; env?: Record<string, string> } = {},
): Promise<Output> {
    const child = new Deno.Command(cmd, {
        args,
        cwd: options.cwd ?? ROOT,
        env: options.env,
        stdin: options.stdin === undefined ? "null" : "piped",
        stdout: "piped",
        stderr: "piped",
    }).spawn();
    if (options.stdin !== undefined) {
        const writer = child.stdin.getWriter();
        await writer.write(new TextEncoder().encode(options.stdin));
        await writer.close();
    }
    const result = await child.output();
    const decoder = new TextDecoder();
    return { code: result.code, stdout: decoder.decode(result.stdout), stderr: decoder.decode(result.stderr) };
}

/** Runs a command that must succeed and returns its stdout; a failure names the command and its stderr. */
export function runChecked(cmd: string, args: string[], options: { cwd?: string } = {}): string {
    const out = run(cmd, args, options);
    if (out.code !== 0) {
        throw new CommandError(`\`${[cmd, ...args].join(" ")}\` exited ${out.code}: ${out.stderr.trim()}`);
    }
    return out.stdout;
}

/** A command the script depends on failed; main() reports it and exits 2. */
export class CommandError extends Error {}

/** An argument or input the caller gave is unusable; main() prints it with the usage and exits 2. */
export class UsageError extends Error {}

/** Writes a line to stderr. */
export function eprint(message: string): void {
    console.error(message);
}
