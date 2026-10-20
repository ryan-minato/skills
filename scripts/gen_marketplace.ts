#!/usr/bin/env -S deno run --no-prompt --allow-read=skills,.claude-plugin/marketplace.json --allow-write=.claude-plugin/marketplace.json
// Regenerates each plugin's `skills[]` in .claude-plugin/marketplace.json.
//
// Every catalog plugin (its `name` equals a catalog directory under skills/) gets its `skills` array
// rewritten to the sorted list of that catalog's skill directories (`./skills/<catalog>/<skill>`, each
// holding a SKILL.md). Those exact paths are what the vercel `skills` CLI matches to group the picker by
// catalog, and what Claude Code loads as the plugin's skills.
//
// Human-owned fields (name, source, strict, description), key order, and plugin order are preserved; the
// generator never adds or removes plugin entries. A non-empty catalog with no entry, or an entry whose
// catalog is empty or missing, is reported to stderr as a warning (validate_skills turns those into hard
// errors) and the file is still written.
//
// Usage: ./scripts/gen_marketplace.ts   (from the repository root; rewrites the file in place)
// Called by: `just gen-marketplace`, run by hand after a skill or catalog is added or removed.
// Exit codes: 0 written (warnings included); 2 usage error or an unusable marketplace.json.
import { parseArgs } from "./lib/args.ts";
import { eprint, join, readText, ROOT, UsageError } from "./lib/repo.ts";

export const MARKETPLACE = ".claude-plugin/marketplace.json";

export const USAGE = `usage: gen_marketplace.ts [-h]

Rewrite each catalog plugin's skills[] in ${MARKETPLACE} to the sorted
skill directories of that catalog on disk. Run from the repository root.

options:
  -h, --help  show this help message and exit`;

/** marketplace.json is not the shape this generator rewrites; main() reports it and exits 2. */
export class ManifestError extends Error {}

export interface Plugin {
    name: string;
    skills?: unknown;
    [key: string]: unknown;
}

export interface Manifest {
    plugins: Plugin[];
    [key: string]: unknown;
}

function isDir(path: string): boolean {
    try {
        return Deno.statSync(path).isDirectory;
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
    }
}

function isFile(path: string): boolean {
    try {
        return Deno.statSync(path).isFile;
    } catch (error) {
        if (error instanceof Deno.errors.NotFound) return false;
        throw error;
    }
}

/** Names of the directories (symlinks followed) directly inside `dir`, sorted. */
function subdirs(dir: string): string[] {
    return [...Deno.readDirSync(dir)].map((e) => e.name).filter((name) => isDir(join(dir, name))).sort();
}

/** Sorted `./skills/<catalog>/<skill>` for every skill directory (one holding a SKILL.md) in the catalog. */
export function catalogSkillPaths(skillsDir: string, catalog: string): string[] {
    const catDir = join(skillsDir, catalog);
    if (!isDir(catDir)) return [];
    return subdirs(catDir)
        .filter((name) => isFile(join(catDir, name, "SKILL.md")))
        .map((name) => `./skills/${catalog}/${name}`);
}

/** One warning per catalog that has skills but no plugin entry. */
export function coverageWarnings(skillsDir: string, listed: Set<string>): string[] {
    if (!isDir(skillsDir)) return [];
    return subdirs(skillsDir)
        .filter((cat) => !listed.has(cat) && catalogSkillPaths(skillsDir, cat).length > 0)
        .map((cat) =>
            `catalog \`${cat}\` has skills but no plugin entry; add one with a name and description, then re-run.`
        );
}

/** Parses marketplace.json and checks it has the fields the generator relies on. */
export function parseManifest(text: string): Manifest {
    let data: unknown;
    try {
        data = JSON.parse(text);
    } catch (error) {
        throw new ManifestError(`${MARKETPLACE} is not valid JSON (${(error as Error).message}); fix it by hand.`);
    }
    if (typeof data !== "object" || data === null || Array.isArray(data)) {
        throw new ManifestError(`${MARKETPLACE}: the top level is not an object; fix it by hand.`);
    }
    const plugins = (data as Record<string, unknown>).plugins;
    if (!Array.isArray(plugins)) {
        throw new ManifestError(
            `${MARKETPLACE}: \`plugins\` is ${JSON.stringify(plugins)}, not an array; fix it by hand.`,
        );
    }
    plugins.forEach((entry, i) => {
        if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
            throw new ManifestError(`${MARKETPLACE}: plugins[${i}] is not an object; fix it by hand.`);
        }
        const name = (entry as Record<string, unknown>).name;
        if (typeof name !== "string" || name === "") {
            throw new ManifestError(
                `${MARKETPLACE}: plugins[${i}].name is ${
                    JSON.stringify(name)
                }, not a non-empty string; fix it by hand.`,
            );
        }
    });
    return data as Manifest;
}

/** Rewrites every plugin's `skills` in place and returns the warnings in the order they are printed. */
export function regenerate(manifest: Manifest, skillsDir: string): string[] {
    const warnings: string[] = [];
    const listed = new Set<string>();
    for (const entry of manifest.plugins) {
        listed.add(entry.name);
        const expected = catalogSkillPaths(skillsDir, entry.name);
        if (expected.length === 0) {
            warnings.push(
                `plugin \`${entry.name}\`: catalog skills/${entry.name}/ is empty or missing; ` +
                    "remove the entry or add a skill.",
            );
        }
        entry.skills = expected;
    }
    warnings.push(...coverageWarnings(skillsDir, listed));
    return warnings;
}

/** The file text: two-space-indented JSON with non-ASCII kept and a trailing newline. */
export function render(manifest: Manifest): string {
    return JSON.stringify(manifest, null, 2) + "\n";
}

export function main(argv: string[], root: string = ROOT): number {
    try {
        const args = parseArgs(argv, { boolean: ["help"], alias: { h: "help" } });
        if (args.flags.help) {
            console.log(USAGE);
            return 0;
        }
        if (args.positional.length > 0) throw new UsageError(`unexpected argument ${args.positional[0]}`);
    } catch (error) {
        if (!(error instanceof UsageError)) throw error;
        eprint(USAGE);
        eprint(`gen_marketplace.ts: error: ${error.message}`);
        return 2;
    }

    const path = join(root, MARKETPLACE);
    let manifest: Manifest;
    try {
        manifest = parseManifest(readText(path));
    } catch (error) {
        if (!(error instanceof ManifestError)) throw error;
        eprint(`ERROR: ${error.message}`);
        return 2;
    }
    for (const warning of regenerate(manifest, join(root, "skills"))) eprint(`WARNING: ${warning}`);
    Deno.writeTextFileSync(path, render(manifest));
    console.log(`Wrote ${MARKETPLACE}.`);
    return 0;
}

if (import.meta.main) Deno.exit(main(Deno.args));
