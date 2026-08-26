# Resolve GitLab Facts and Tooling

Read before the session's first GitLab documentation or platform query, when a
recorded URL stops resolving, or when host, version, tier, permissions, or
tooling are uncertain.

## Resolve current documentation

1. Fetch `https://docs.gitlab.com/llms.txt` and search it for the capability,
   not a remembered page path. It is a discovery entrypoint, not a source to
   copy into the target harness.
2. Open the resulting first-party topic page and read its Tier, Offering,
   History, prerequisites, and linked API or YAML reference.
3. For self-managed GitLab, prefer the instance's `/help` documentation and
   query its version. Public docs describe the newest release, not necessarily
   the target.
4. Check live availability and permissions by a read operation before
   promising the feature. Design the Free-tier fallback first.
5. Record only stable entrypoints and topic-search instructions in durable
   guidance. Do not deposit a static URL catalog, copied API inventory, or
   versioned syntax table.

Suggested search terms by branch belong in that branch's reference. Re-run
discovery when implementing or revising the corresponding target artifact.

## Resolve host and project

Parse `git remote get-url origin`. The host follows `https://` or `@`; the
project path is everything after the host with `.git` removed. Preserve nested
groups. The group path is the project path without its last component. If the
user named another target, use it and do not silently substitute the checkout.

## Choose one tool path

Run `uv run scripts/check_tooling.py --hostname HOST`.

1. Prefer authenticated `glab` for broad GitLab operations.
2. Otherwise use a connected GitLab MCP capability whose description exactly
   matches the needed operation. Describe capabilities, never hardcoded MCP
   tool names.
3. Otherwise use `rest_read.py` for the minimal read-only portion when the
   project is public or an appropriate token is already available.
4. For a write without an authenticated path, retain the reviewed draft and
   stop with exact setup requirements. Never collect or echo token values.

Use one path for one operation. Do not mix clients halfway through a write
unless the selected client lacks a documented capability and the switch is
explicitly recorded.

## Permissions and external actions

- Authentication proves identity, not authorization from the user.
- Read project/group membership and relevant permissions before applying
  settings or assigning people.
- Use least-privilege tokens, protected/masked variables, and host-specific
  configuration. Never assume a token intended for GitLab.com targets a
  self-managed host.
- All writes remain behind the exact-payload review and user-approval gates in
  SKILL.md.

Done when: host, full project/group path, version, tier, available capability,
permission level, and the single selected tool path are evidenced.
