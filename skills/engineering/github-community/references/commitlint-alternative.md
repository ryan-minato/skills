# commitlint as the validator (explicit request only)

Load condition: the user explicitly asks for commitlint — typically a
Node project that already carries a JS toolchain and wants husky hooks
or an existing shared config.

The default validator for this skill is the committed Python script:
zero dependencies, same behavior on every runner, one CONFIG block.
commitlint trades that for the npm ecosystem — a reasonable trade only
when the project already lives there. It is a third-party npm package:
pin its version and get the user's explicit opt-in.

## CI job (no third-party actions)

Replace the validation step in `.github/workflows/commit-check.yml`
with a pinned `npx` run — still no third-party *action*:

```yaml
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - name: Validate commit messages with commitlint
        run: >
          npx --yes @commitlint/cli@19 @commitlint/config-conventional@19
          --extends @commitlint/config-conventional
          --from "${{ github.event.pull_request.base.sha }}"
          --to "${{ github.event.pull_request.head.sha }}" --verbose
```

## Config file

Commit `commitlint.config.mjs` at the repo root and mirror the
convention doc in it:

```js
export default {
  extends: ["@commitlint/config-conventional"],
  rules: {
    "header-max-length": [2, "always", 72],
    // scope-enum example for a required scope set:
    // "scope-enum": [2, "always", ["api", "cli", "docs"]],
  },
};
```

## Local hook

Husky is the common local runner; it is also an npm dependency the
project must already accept. `npx husky init` then put
`npx --no -- commitlint --edit "$1"` in `.husky/commit-msg`. Without
husky, the Python validator's `--file` mode works as a plain
`.git/hooks/commit-msg` hook with no dependencies at all.
