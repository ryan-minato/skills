# GitHub Documentation and Tooling Resolution

Read before the session's first GitHub documentation or platform query, when
a recorded procedure stops resolving, or when host, owner type, plan,
permissions, or tooling are uncertain.

## Resolve documentation through the docs APIs, not a topic index

`https://docs.github.com/llms.txt` is a short curated pointer file, not a
searchable topic index; `llms-full.txt` does not exist and there is no
per-page `.md` variant. Resolve every volatile capability like this:

1. Search: `https://docs.github.com/api/search/v1?query=<terms>&language=en&version=<version>&client_name=<any-identifier>`.
   The `client_name` parameter is required for external callers even though
   the docs never say so; omitting it returns an error, not results.
2. Fetch the winning hit as markdown:
   `https://docs.github.com/api/article/body?pathname=<url-path>`. A wrong
   path fails loudly — always search first, never guess paths.
3. Choose `version=` from the capability quadrant: `free-pro-team@latest`
   for github.com, `enterprise-cloud@latest` only after the plan is
   confirmed Enterprise, `enterprise-server@<X.Y>` for GHES (read the
   version from the instance's `/api/v3/meta`).
4. Never take a plan or permission fact from an API-returned article body:
   the rendered "Who can use this feature?" callout is stripped from it.
   Read the rendered page or the GitHub plans documentation instead, and
   confirm gated features by a live read before promising them.
5. Record only stable entrypoints and this resolution procedure in durable
   guidance. One stable entrypoint plus a parameter grammar is a procedure;
   a per-capability URL list, a copied API inventory, or a version-pinned
   syntax table is a catalog — do not deposit catalogs.

## Derive the target before any call

Derive `OWNER/REPO` and the host from `git remote get-url origin`. A GHES
host changes the API root (`https://<host>/api/v3`), the docs `version=`
value, and the available feature set. Never assume github.com.

## One tool path, chosen once

Prefer, in order, and do not mix within one operation:

1. An authenticated `gh` CLI (`gh auth status` exits 0 for the target host).
2. A connected GitHub MCP capability, matched by its described purpose —
   never by tool name, which churns across server versions. Some toolsets
   (discussions, labels, several Actions operations) are off by default in
   the reference MCP server, and `gh` has no `milestone` or `discussion`
   command group — those go through `gh api` on the CLI path.
3. `scripts/rest_read.py` for read-only fallback: public
   target or a token in the environment, minimum requests, never writes.
4. Stop, and tell the user exactly which tooling and authentication the
   blocked step needs. Authentication proves identity, not authorization
   from the user; a `GH_TOKEN` in the environment overrides `gh auth login`
   silently — check both before diagnosing auth failures.

## Probe before asserting

Run `scripts/check_tooling.py` with
`--repo OWNER/REPO` at stage 1. It evidences the capability quadrant,
Actions availability, the allowed-actions policy, and the default
`GITHUB_TOKEN` workflow permissions, and prints a `verify` hint for every
field it cannot read. A 403 or 404 can mean plan, owner type, visibility,
permissions, or genuine absence — it is never, on its own, proof that a
feature does not exist. Treat "Actions assumed available" as false until
the probe or the user says otherwise.

Done when: the host, quadrant, tool path, and every capability the design
will rely on are evidenced or recorded as an explicit unknown with its
verification path.
