// A small command-line parser with no remote import (scripts/lib/repo.ts explains why). It accepts
// `--name value`, `--name=value`, and boolean `--flag`; anything it does not know is a UsageError.
import { UsageError } from "./repo.ts";

export interface ArgSpec {
    /** Options that take a value; the last occurrence wins unless listed in `multiple`. */
    string?: string[];
    /** Options that take no value. */
    boolean?: string[];
    /** String options that may repeat; their values are collected in order. */
    multiple?: string[];
    /** Short aliases, e.g. { h: "help" }. */
    alias?: Record<string, string>;
}

export interface ParsedArgs {
    values: Record<string, string | undefined>;
    lists: Record<string, string[]>;
    flags: Record<string, boolean>;
    positional: string[];
}

export function parseArgs(argv: string[], spec: ArgSpec): ParsedArgs {
    const strings = new Set([...(spec.string ?? []), ...(spec.multiple ?? [])]);
    const booleans = new Set(spec.boolean ?? []);
    const multiple = new Set(spec.multiple ?? []);
    const parsed: ParsedArgs = { values: {}, lists: {}, flags: {}, positional: [] };
    for (const name of spec.multiple ?? []) parsed.lists[name] = [];
    for (const name of booleans) parsed.flags[name] = false;
    let i = 0;
    while (i < argv.length) {
        const arg = argv[i];
        if (arg === "--") {
            parsed.positional.push(...argv.slice(i + 1));
            break;
        }
        let name: string | undefined;
        let inline: string | undefined;
        if (arg.startsWith("--")) {
            const eq = arg.indexOf("=");
            name = eq === -1 ? arg.slice(2) : arg.slice(2, eq);
            inline = eq === -1 ? undefined : arg.slice(eq + 1);
        } else if (/^-[A-Za-z]$/.test(arg)) {
            name = spec.alias?.[arg.slice(1)];
            if (name === undefined) throw new UsageError(`unknown option ${arg}`);
        } else {
            parsed.positional.push(arg);
            i++;
            continue;
        }
        if (booleans.has(name)) {
            if (inline !== undefined) throw new UsageError(`--${name} takes no value`);
            parsed.flags[name] = true;
            i++;
            continue;
        }
        if (!strings.has(name)) throw new UsageError(`unknown option --${name}`);
        let value = inline;
        if (value === undefined) {
            if (i + 1 >= argv.length) throw new UsageError(`--${name} needs a value`);
            value = argv[i + 1];
            i++;
        }
        if (multiple.has(name)) parsed.lists[name].push(value);
        else parsed.values[name] = value;
        i++;
    }
    return parsed;
}
