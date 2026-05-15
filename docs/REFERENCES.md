# REFERENCES.md — External Documentation

## Skills Specification

| Resource | Where | When to Use |
|---|---|---|
| Skills spec (agentskills.io) | `skills-spec` MCP server in devcontainer | Verifying frontmatter rules, naming constraints, body limits |

## Runtimes

| Tool | Reference | Notes |
|---|---|---|
| uv | https://docs.astral.sh/uv/llms.txt | Python package manager; required for inline-metadata scripts |
| uv inline script metadata | https://docs.astral.sh/uv/guides/scripts/index.md | `# /// script` block format |
| Deno | https://docs.deno.com/llms.txt | TypeScript/JavaScript runtime for skill and repo scripts |
| Deno permissions | https://docs.deno.com/runtime/fundamentals/security.md | Required flags for file, net, env access |

## Commit Standards

| Standard | Reference |
|---|---|
| Conventional Commits 1.0.0 | https://raw.githubusercontent.com/conventional-commits/conventionalcommits.org/refs/heads/master/content/v1.0.0/index.md |

## Security Tooling

| Tool | Reference | Notes |
|---|---|---|
| detect-secrets | https://raw.githubusercontent.com/Yelp/detect-secrets/refs/heads/master/README.md | Baseline: `.secrets.baseline` |
| gitleaks | https://raw.githubusercontent.com/gitleaks/gitleaks/refs/heads/master/README.md | Pre-commit hook: `v8.24.2` |
| pre-commit | https://pre-commit.com/ | Config: `.pre-commit-config.yaml` |
