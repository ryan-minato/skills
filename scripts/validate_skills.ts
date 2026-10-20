#!/usr/bin/env -S deno run --no-prompt --allow-read=skills,.agents/skills,.claude-plugin,ARCHITECTURE.md,README.md,README.zh.md
// Validates the repository's skill layout and harness consistency.
//
// Usage: scripts/validate_skills.ts   (no arguments; called by `just validate` and the
// `validate-skills` pre-commit hook, from the repository root)
//
// Per-skill SKILL.md checks (spec frontmatter rules, size limits, README rule) live in
// check_skill.ts; this validator runs them over every skill in the repository and adds the
// repo-level checks:
//
//   1. Self-containment: no markdown links escaping a public skill's directory (targets
//      starting with `../` or `/`).
//   2. No install command (`npx skills add`) in any file of a public skill other than
//      core/ryan-minato-skills-installing, which owns installing; every handoff routes
//      through that skill.
//   3. Every public skill is symlinked into .agents/skills/ and no symlink there dangles.
//   4. marketplace.json publishes one plugin per non-empty catalog, and no legacy
//      .claude-plugin/plugin.json exists.
//   5. Every catalog has README.md, README.zh.md, and CONTEXT.md; the root README.zh.md exists.
//   6. Catalog directories match the catalog list in ARCHITECTURE.md.
//   7. Catalogs that reserve a name prefix only hold skills carrying it.
//   8. Every skill in a disposable catalog opens its description with the shared disposable
//      marker.
//   9. The `## Harness Methodology` sections of core/meta-harness and
//      meta/meta-harness-architecture are identical.
//
// Errors and warnings go to stderr; the OK line goes to stdout. Exit 0 = no errors (warnings
// from check_skill.ts are reported but never fail), 1 = at least one error, 2 = bad arguments.
// Messages explain what failed, why it matters, and how to fix it.
import {
    baseName,
    checkSkillDir,
    decodeUtf8,
    ERROR,
    isDir,
    isFile,
    isSymlink,
    listDir,
    NAME_RE,
    normalizePath,
    parseFrontmatter,
    pyRstrip,
    readUtf8,
    relTo,
    repoSkillDirs,
} from "./check_skill.ts";
import { parseArgs } from "./lib/args.ts";
import { eprint, ROOT, UsageError } from "./lib/repo.ts";

const HARNESS_METHODOLOGY_HEADING = "## Harness Methodology";
// Catalogs whose CONTEXT.md reserves a name prefix for their skills. The prefix is what tells
// an installed skill's reader which catalog's contract governs it, so drift here is not cosmetic.
export const CATALOG_NAME_PREFIXES: Record<string, string> = { meta: "meta-", scaffold: "scaffold-" };
// Every skill in a disposable catalog opens its description with this exact sentence. It is
// the only key the disposal skill matches on (installers rename skills, so names and
// directories are not stable), and it is shared by both catalogs so one disposal skill
// removes them together.
export const DISPOSABLE_MARKER = "Disposable builder skill (delete after the harness is built):";
export const DISPOSABLE_CATALOGS = ["meta", "scaffold"];
// The one public skill allowed to name an install command: it owns installing.
export const INSTALLER_SKILL = "skills/core/ryan-minato-skills-installing";
export const INSTALL_COMMAND = "npx skills add";

const ESCAPING_LINK = /\]\((?:\.\.\/|\/)[^)]*\)/g;

const USAGE = `usage: scripts/validate_skills.ts

Validate the skill layout and harness consistency of the repository this script lives in.
Takes no arguments. Exit 0 = clean (warnings allowed), 1 = problems found, 2 = bad arguments.`;

/** The outcome of a validation run. */
export interface Result {
    errors: string[];
    warnings: string[];
}

/** One validation run over the repository at `root`. */
class Validator {
    readonly errors: string[] = [];
    readonly warnings: string[] = [];
    readonly skillsDir: string;
    readonly agentSkillsDir: string;
    readonly marketplaceManifest: string;
    readonly legacyPluginManifest: string;
    readonly architectureMd: string;
    readonly coreMetaHarness: string;
    readonly metaArchitecture: string;

    constructor(readonly root: string) {
        this.skillsDir = `${root}/skills`;
        this.agentSkillsDir = `${root}/.agents/skills`;
        this.marketplaceManifest = `${root}/.claude-plugin/marketplace.json`;
        this.legacyPluginManifest = `${root}/.claude-plugin/plugin.json`;
        this.architectureMd = `${root}/ARCHITECTURE.md`;
        this.coreMetaHarness = `${this.skillsDir}/core/meta-harness/SKILL.md`;
        this.metaArchitecture = `${this.skillsDir}/meta/meta-harness-architecture/SKILL.md`;
    }

    fail(message: string): void {
        this.errors.push(message);
    }

    rel(path: string): string {
        return relTo(this.root, path);
    }

    /** Runs the per-skill linter over public and project-only skills. */
    checkAllSkills(): void {
        for (const skillDir of repoSkillDirs(this.root)) {
            for (const [severity, message] of checkSkillDir(skillDir, this.root)) {
                if (severity === ERROR) this.fail(message);
                else this.warnings.push(message);
            }
        }
    }

    // The public skills' directories, files, and contents, read once per run: two checks scan them.
    private publicDirs: string[] | undefined;
    private readonly files = new Map<string, string[]>();
    private readonly contents = new Map<string, Uint8Array>();

    publicSkillDirs(): string[] {
        if (this.publicDirs === undefined) {
            this.publicDirs = isDir(this.skillsDir)
                ? listDir(this.skillsDir).filter(isDir).flatMap((catalog) => listDir(catalog).filter(isDir))
                : [];
        }
        return this.publicDirs;
    }

    skillFiles(skillDir: string): string[] {
        if (!this.files.has(skillDir)) this.files.set(skillDir, walkFiles(skillDir));
        return this.files.get(skillDir)!;
    }

    bytes(path: string): Uint8Array {
        if (!this.contents.has(path)) this.contents.set(path, Deno.readFileSync(path));
        return this.contents.get(path)!;
    }

    /** Public skills must not reference anything outside their own root; project-only skills are exempt. */
    checkSelfContainment(): void {
        for (const skillDir of this.publicSkillDirs()) {
            for (const file of this.skillFiles(skillDir).filter((path) => path.endsWith(".md"))) {
                for (const match of decodeUtf8(this.bytes(file), file).matchAll(ESCAPING_LINK)) {
                    this.fail(
                        `${this.rel(file)}: link ` +
                            `${match[0]} escapes the skill directory. ` +
                            `Installed skills lose access to everything outside ` +
                            `their own root, so the link breaks. Fix: copy the ` +
                            `content into the skill, or hand off to the other skill: ` +
                            `load the \`ryan-minato-skills-installing\` skill and install ` +
                            `it as that skill directs, never through an install command ` +
                            `(see .agents/knowledge/skill-quality.md).`,
                    );
                }
            }
        }
    }

    /** No public skill other than the installer names an install command. */
    checkInstallCommands(): void {
        for (const skillDir of this.publicSkillDirs()) {
            if (this.rel(skillDir) === INSTALLER_SKILL) continue;
            for (const file of this.skillFiles(skillDir)) {
                // Every file counts, binary ones included, so bytes that are not UTF-8 decode lossily here.
                const text = new TextDecoder().decode(this.bytes(file));
                text.split("\n").forEach((line, index) => {
                    if (!line.includes(INSTALL_COMMAND)) return;
                    this.fail(
                        `${this.rel(file)}:${index + 1}: contains the install command ` +
                            `\`${INSTALL_COMMAND}\`. A public skill never prints an install ` +
                            `command: the installing skill owns runner choice and install ` +
                            `scope, and every handoff routes through it. Fix: replace the ` +
                            `command with a handoff that loads the ` +
                            `\`ryan-minato-skills-installing\` skill and installs the other ` +
                            `skill as it directs (see .agents/knowledge/skill-quality.md).`,
                    );
                });
            }
        }
    }

    checkCatalogs(): void {
        if (!isDir(this.skillsDir)) return;
        for (const catalog of listDir(this.skillsDir).filter(isDir)) {
            const rel = this.rel(catalog);
            for (const required of ["README.md", "README.zh.md", "CONTEXT.md"]) {
                if (!isFile(`${catalog}/${required}`)) {
                    this.fail(
                        `${rel}: missing ${required}. Every catalog needs ` +
                            `README.md (English), README.zh.md (Chinese ` +
                            `translation), and CONTEXT.md (catalog-specific rules). ` +
                            `Fix: create ${rel}/${required}.`,
                    );
                }
            }
        }
    }

    /** Keeps prefix-reserving catalogs free of skills that omit the prefix. */
    checkCatalogNamePrefixes(): void {
        for (const catalog of Object.keys(CATALOG_NAME_PREFIXES).sort()) {
            const prefix = CATALOG_NAME_PREFIXES[catalog];
            for (const skillDir of catalogSkillSubdirs(`${this.skillsDir}/${catalog}`)) {
                const name = baseName(skillDir);
                if (name.startsWith(prefix)) continue;
                this.fail(
                    `skills/${catalog}/${name}: name does not start ` +
                        `with \`${prefix}\`. The ${catalog} catalog reserves that ` +
                        `prefix so an installed skill announces which catalog's ` +
                        `contract governs it (see ` +
                        `skills/${catalog}/CONTEXT.md). Fix: rename the directory ` +
                        `and its \`name\` field to \`${prefix}...\`, or move the skill ` +
                        `to a catalog that fits it.`,
                );
            }
        }
    }

    /** Every skill in a disposable catalog carries the shared marker. */
    checkDisposableMarkers(): void {
        for (const catalog of DISPOSABLE_CATALOGS) {
            for (const skillDir of catalogSkillSubdirs(`${this.skillsDir}/${catalog}`)) {
                const skillMd = `${skillDir}/SKILL.md`;
                const [fields] = parseFrontmatter(readUtf8(skillMd));
                const description = fields?.["description"] ?? "";
                if (description.startsWith(DISPOSABLE_MARKER)) continue;
                this.fail(
                    `${this.rel(skillMd)}: description does not ` +
                        `open with \`${DISPOSABLE_MARKER}\`. That sentence is how the ` +
                        `disposal skill finds every temporary builder in a target ` +
                        `project; a builder without it survives cleanup and keeps ` +
                        `loading in normal development. Fix: start the description ` +
                        `with the exact marker (see skills/${catalog}/CONTEXT.md).`,
                );
            }
        }
    }

    checkArchitectureMdCatalogList(): void {
        if (!isFile(this.architectureMd)) {
            this.fail(
                "ARCHITECTURE.md: missing. It documents the repo mechanisms " +
                    "and holds the catalog list. Fix: restore it from git history.",
            );
            return;
        }
        const section = markdownSectionBody(readUtf8(this.architectureMd), "## Catalogs");
        if (section === null) {
            this.fail(
                "ARCHITECTURE.md: no `## Catalogs` section found. The validator " +
                    "(and agents) read the catalog list from that section. Fix: add " +
                    "a `## Catalogs` heading listing each catalog as `- `name` — ...`.",
            );
            return;
        }
        const listed = new Set([...section.matchAll(/(?:^|(?<=\n))- `([\p{L}\p{N}_-]+)`/gu)].map((m) => m[1]));
        const actual = new Set(isDir(this.skillsDir) ? listDir(this.skillsDir).filter(isDir).map(baseName) : []);
        for (const missing of [...actual].filter((name) => !listed.has(name)).sort()) {
            this.fail(
                `ARCHITECTURE.md: catalog \`${missing}\` exists in skills/ but is ` +
                    `not listed in the Catalogs section. Agents rely on ` +
                    `ARCHITECTURE.md to know the catalogs. Fix: add \`- ` +
                    `\\\`${missing}\\\`\` with a short description to ARCHITECTURE.md.`,
            );
        }
        for (const stale of [...listed].filter((name) => !actual.has(name)).sort()) {
            this.fail(
                `ARCHITECTURE.md: catalog \`${stale}\` is listed but ` +
                    `skills/${stale}/ does not exist. Stale harness misleads ` +
                    `agents. Fix: remove the entry or create the catalog.`,
            );
        }
    }

    checkSymlinks(): void {
        if (!isDir(this.agentSkillsDir)) {
            this.fail(
                ".agents/skills/: missing. Project agents discover skills " +
                    "there. Fix: recreate it and re-run this validator.",
            );
            return;
        }
        for (const entry of listDir(this.agentSkillsDir)) {
            if (isSymlink(entry) && !isDir(entry)) {
                this.fail(
                    `.agents/skills/${baseName(entry)}: dangling symlink (target ` +
                        `missing). Agents fail to load the skill. Fix: remove the ` +
                        `link or restore the target skill directory.`,
                );
            }
        }
        for (const skillDir of this.publicSkillDirs()) {
            const name = baseName(skillDir);
            const link = `${this.agentSkillsDir}/${name}`;
            const rel = this.rel(skillDir);
            if (!isSymlink(link)) {
                this.fail(
                    `${rel}: no symlink in .agents/skills/. Without it this ` +
                        `repo's agents cannot use the skill. Fix: ` +
                        `ln -s ../../${rel} .agents/skills/${name}`,
                );
            } else if (resolvePath(link) !== resolvePath(skillDir)) {
                this.fail(
                    `.agents/skills/${name}: points to ` +
                        `${resolvePath(link)} instead of ${rel}. Fix: recreate it with ` +
                        `ln -sfn ../../${rel} .agents/skills/${name}`,
                );
            }
        }
    }

    /** Catalogs under skills/ that publish at least one skill. */
    nonemptyCatalogNames(): Set<string> {
        if (!isDir(this.skillsDir)) return new Set();
        return new Set(
            listDir(this.skillsDir).filter((catalog) => catalogSkillSubdirs(catalog).length > 0).map(baseName),
        );
    }

    /**
     * The repo publishes as a pure marketplace: one plugin per non-empty catalog, each entry
     * using a marketplace-root `source` and a `skills` array that enumerates the catalog's
     * skill directories (`./skills/<catalog>/<skill>`). Those exact paths are what the vercel
     * `skills` CLI groups on and what Claude Code loads; the list must match the catalog on
     * disk (kept in sync by `just gen-marketplace`).
     */
    checkMarketplaceManifest(): void {
        if (isFile(this.legacyPluginManifest)) {
            this.fail(
                ".claude-plugin/plugin.json: the repo now publishes as a " +
                    "marketplace (.claude-plugin/marketplace.json), not one plugin. " +
                    "A leftover plugin.json is merged into every marketplace-root " +
                    "plugin and cross-loads all catalogs. Fix: remove it " +
                    "(git rm .claude-plugin/plugin.json).",
            );
        }

        if (!isFile(this.marketplaceManifest)) {
            this.fail(
                ".claude-plugin/marketplace.json: missing. It publishes each " +
                    "catalog as an installable plugin. Fix: create it with `name`, " +
                    "`owner`, and one `plugins` entry per non-empty catalog.",
            );
            return;
        }
        let manifest: unknown;
        try {
            manifest = JSON.parse(readUtf8(this.marketplaceManifest));
        } catch (error) {
            if (!(error instanceof SyntaxError)) throw error;
            this.fail(
                `.claude-plugin/marketplace.json: invalid JSON (${error.message}). ` +
                    `Marketplace adds will fail. Fix: repair the syntax.`,
            );
            return;
        }
        if (!isObject(manifest)) {
            this.fail(
                ".claude-plugin/marketplace.json: the top level must be an object " +
                    "with `name`, `owner`, and `plugins`. Fix: run `just gen-marketplace`.",
            );
            return;
        }

        if (!truthy(manifest["name"])) {
            this.fail(
                ".claude-plugin/marketplace.json: missing top-level `name` (the " +
                    "marketplace id users type as `<plugin>@<name>`). Fix: add a " +
                    "kebab-case `name`.",
            );
        }
        const owner = manifest["owner"];
        if (!isObject(owner) || !truthy(owner["name"])) {
            this.fail(
                ".claude-plugin/marketplace.json: missing `owner.name`. The " +
                    'schema requires an owner. Fix: add `"owner": {"name": "..."}`.',
            );
        }
        const plugins = manifest["plugins"];
        if (!Array.isArray(plugins) || plugins.length === 0) {
            this.fail(
                ".claude-plugin/marketplace.json: `plugins` must be a non-empty " +
                    "list, one entry per non-empty catalog. Fix: add the entries.",
            );
            return;
        }

        const listedCatalogs = new Set<string>();
        for (const entry of plugins) {
            if (!isObject(entry)) {
                this.fail(
                    ".claude-plugin/marketplace.json: every `plugins` entry must " +
                        "be an object with `name`, `source`, and `skills`.",
                );
                continue;
            }
            const rawName = entry["name"] ?? "";
            const name = typeof rawName === "string" ? rawName : JSON.stringify(rawName);
            listedCatalogs.add(name);
            if (!name || typeof rawName !== "string" || !NAME_RE.test(name)) {
                this.fail(
                    `.claude-plugin/marketplace.json: plugin name \`${name}\` is ` +
                        `not kebab-case; claude.ai marketplace sync rejects ` +
                        `non-kebab names. Fix: rename to lowercase-hyphen.`,
                );
            }
            if (entry["source"] !== "./") {
                this.fail(
                    `.claude-plugin/marketplace.json: plugin \`${name}\` source must ` +
                        `be \`"./"\` (the marketplace root); that is what makes the ` +
                        `\`skills\` filter replace the default scan so the plugin loads ` +
                        `only its own catalog. Fix: set \`"source": "./"\`.`,
                );
            }
            const rawSkills = entry["skills"] ?? [];
            const listed = typeof rawSkills === "string" ? [rawSkills] : rawSkills;
            if (!Array.isArray(listed) || !listed.every((path) => typeof path === "string")) {
                this.fail(
                    `.claude-plugin/marketplace.json: plugin \`${name}\` \`skills\` must ` +
                        `be a list of path strings. Fix: run \`just gen-marketplace\`.`,
                );
                continue;
            }
            const skills = listed as string[];
            for (const path of skills) {
                if (!path.startsWith("./")) {
                    this.fail(
                        `.claude-plugin/marketplace.json: plugin \`${name}\` skills ` +
                            `path \`${path}\` must be a relative path starting with ` +
                            `\`./\` (no \`..\` traversal). Fix: run \`just gen-marketplace\`.`,
                    );
                }
            }

            // Each plugin must enumerate exactly its catalog's skill directories, so the vercel
            // picker groups them and Claude Code loads them. `just gen-marketplace` keeps this
            // list in sync.
            const expected = catalogSkillSubdirs(`${this.skillsDir}/${name}`).map(
                (dir) => `./skills/${name}/${baseName(dir)}`,
            );
            if (expected.length === 0) {
                this.fail(
                    `.claude-plugin/marketplace.json: plugin \`${name}\` has no ` +
                        `catalog skills/${name}/ with skills. An empty or missing ` +
                        `catalog must not be published. Fix: remove the entry.`,
                );
            } else if (!sameList([...skills].sort(), [...expected].sort())) {
                const missing = [...new Set(expected)].filter((path) => !skills.includes(path)).sort();
                const stale = [...new Set(skills)].filter((path) => !expected.includes(path)).sort();
                const detail = [
                    missing.length > 0 ? `missing ${pyListRepr(missing)}` : "",
                    stale.length > 0 ? `stale ${pyListRepr(stale)}` : "",
                ].filter((part) => part !== "").join(", ");
                this.fail(
                    `.claude-plugin/marketplace.json: plugin \`${name}\` skills[] is ` +
                        `out of sync with skills/${name}/ (${detail}). Fix: run ` +
                        `\`just gen-marketplace\`.`,
                );
            }
        }

        for (const missing of [...this.nonemptyCatalogNames()].filter((name) => !listedCatalogs.has(name)).sort()) {
            this.fail(
                `.claude-plugin/marketplace.json: catalog \`${missing}\` has skills ` +
                    `but no plugin entry, so it cannot be installed. Fix: add a ` +
                    `\`plugins\` entry with \`"skills": ["./skills/${missing}"]\`.`,
            );
        }
    }

    checkRootFiles(): void {
        if (isFile(`${this.root}/README.md`) && !isFile(`${this.root}/README.zh.md`)) {
            this.fail(
                "README.zh.md: missing. The root README must have a " +
                    "content-identical Chinese translation. Fix: create README.zh.md.",
            );
        }
    }

    /** Keeps the durable core methodology equal to its complete superset. */
    checkMetaHarnessMethodology(): void {
        if (!isFile(this.coreMetaHarness) || !isFile(this.metaArchitecture)) return;
        const core = markdownSection(readUtf8(this.coreMetaHarness), HARNESS_METHODOLOGY_HEADING);
        const architecture = markdownSection(
            readUtf8(this.metaArchitecture),
            HARNESS_METHODOLOGY_HEADING,
        );
        if (core === null) {
            this.fail(
                `${this.rel(this.coreMetaHarness)}: missing ` +
                    `\`${HARNESS_METHODOLOGY_HEADING}\`. The core skill is the durable ` +
                    "methodology subset; restore it from meta-harness-architecture.",
            );
        }
        if (architecture === null) {
            this.fail(
                `${this.rel(this.metaArchitecture)}: missing ` +
                    `\`${HARNESS_METHODOLOGY_HEADING}\`. The complete skill owns the ` +
                    "shared methodology section.",
            );
        }
        if (core !== null && architecture !== null && core !== architecture) {
            this.fail(
                "meta methodology drifted: the `## Harness Methodology` sections " +
                    "in core/meta-harness and meta/meta-harness-architecture must " +
                    "be byte-identical. Update the architecture source first, then " +
                    "copy that section into the core subset.",
            );
        }
    }
}

/** Runs every check over the repository at `root` and returns the findings. */
export function validate(root: string = ROOT): Result {
    const v = new Validator(root);
    v.checkAllSkills();
    v.checkSelfContainment();
    v.checkInstallCommands();
    v.checkCatalogs();
    v.checkArchitectureMdCatalogList();
    v.checkCatalogNamePrefixes();
    v.checkDisposableMarkers();
    v.checkSymlinks();
    v.checkMarketplaceManifest();
    v.checkRootFiles();
    v.checkMetaHarnessMethodology();
    return { errors: v.errors, warnings: v.warnings };
}

/** Skill directories (each holding a SKILL.md) directly under a catalog, sorted. */
export function catalogSkillSubdirs(catalog: string): string[] {
    if (!isDir(catalog)) return [];
    return listDir(catalog).filter((path) => isDir(path) && isFile(`${path}/SKILL.md`));
}

/** Every file under a directory, sorted, entering no symlinked directory. */
export function walkFiles(dir: string): string[] {
    const files: string[] = [];
    const entries = [...Deno.readDirSync(dir)].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    for (const entry of entries) {
        const path = `${dir}/${entry.name}`;
        if (entry.isSymlink) {
            if (isFile(path)) files.push(path);
        } else if (entry.isDirectory) {
            files.push(...walkFiles(path));
        } else if (entry.isFile) {
            files.push(path);
        }
    }
    return files;
}

// Python's `^` and `$` under re.MULTILINE: line starts and ends at "\n" only.
const BOL = "(?:^|(?<=\\n))";
const EOL = "(?=\\n|$)";

function escapeRegExp(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** The text after a level-2 heading line up to the next `## ` line, or null. */
export function markdownSectionBody(text: string, heading: string): string | null {
    const match = new RegExp(`${BOL}${escapeRegExp(heading)}${EOL}([\\s\\S]*?)(?=${BOL}## |$)`).exec(text);
    return match ? match[1] : null;
}

/** One level-2 Markdown section, including its heading, trailing whitespace removed; or null. */
export function markdownSection(text: string, heading: string): string | null {
    const match = new RegExp(`${BOL}${escapeRegExp(heading)}${EOL}[\\s\\S]*?(?=${BOL}## (?!#)|$)`).exec(text);
    return match ? pyRstrip(match[0]) : null;
}

/** A path with every symlink resolved; a dangling link resolves to its target path. */
export function resolvePath(path: string): string {
    try {
        return Deno.realPathSync(path);
    } catch (error) {
        if (!(error instanceof Deno.errors.NotFound) || !isSymlink(path)) throw error;
        const target = Deno.readLinkSync(path);
        const parent = Deno.realPathSync(path.replace(/\/[^/]*$/, "") || "/");
        return normalizePath(target.startsWith("/") ? target : `${parent}/${target}`)
            .split("/")
            .reduce((parts: string[], part) => {
                if (part === "..") parts.pop();
                else parts.push(part);
                return parts;
            }, [])
            .join("/");
    }
}

function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Python truthiness for a JSON value. */
function truthy(value: unknown): boolean {
    if (Array.isArray(value)) return value.length > 0;
    if (isObject(value)) return Object.keys(value).length > 0;
    return Boolean(value);
}

function sameList(a: string[], b: string[]): boolean {
    return a.length === b.length && a.every((item, index) => item === b[index]);
}

/** A list of strings as Python's repr() prints it, so messages read as before the port. */
export function pyListRepr(items: string[]): string {
    const repr = (text: string) => {
        const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
        const body = text.replaceAll("\\", "\\\\").replaceAll("\n", "\\n").replaceAll("\r", "\\r")
            .replaceAll("\t", "\\t");
        return quote + (quote === "'" ? body.replaceAll("'", "\\'") : body) + quote;
    };
    return `[${items.map(repr).join(", ")}]`;
}

/** Prints the result; returns the exit code. */
export function report(result: Result): number {
    for (const warning of result.warnings) eprint(`WARNING: ${warning}\n`);
    if (result.errors.length > 0) {
        eprint(`FAIL: ${result.errors.length} problem(s) found\n`);
        for (const error of result.errors) eprint(`  * ${error}\n`);
        eprint(
            "Source of truth: AGENTS.md, ARCHITECTURE.md, and " +
                ".agents/knowledge/skill-quality.md. " +
                "Re-run with `just validate` after fixing.",
        );
        return 1;
    }
    const suffix = result.warnings.length > 0 ? ` (${result.warnings.length} warning(s))` : "";
    console.log(`OK: skill layout and harness consistency checks passed${suffix}`);
    return 0;
}

export function main(argv: string[], root: string = ROOT): number {
    try {
        const args = parseArgs(argv, { boolean: ["help"], alias: { h: "help" } });
        if (args.flags["help"]) {
            console.log(USAGE);
            return 0;
        }
        if (args.positional.length > 0) throw new UsageError(`unexpected argument ${args.positional[0]}`);
    } catch (error) {
        if (!(error instanceof UsageError)) throw error;
        eprint(USAGE);
        eprint(`ERROR: ${error.message}`);
        return 2;
    }
    try {
        return report(validate(root));
    } catch (error) {
        if (!(error instanceof Deno.errors.NotCapable)) throw error;
        eprint(
            `ERROR: ${error.message}. The read permission is scoped to repository paths ` +
                `relative to the working directory; run this script from the repository root.`,
        );
        return 2;
    }
}

if (import.meta.main) Deno.exit(main(Deno.args));
