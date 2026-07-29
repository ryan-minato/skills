# Search

Loaded when the task searches the instance, a group, or a project. GitLab
has no Discussions forum; instance-, group-, and project-wide search
covers the discovery ground instead. glab has no first-class search
command — use `glab api` (or list filters, which are usually better for
per-project term search).

## Scopes

```bash
glab api "search?scope=SCOPE&search=TEXT"
```

`SCOPE` is one of `projects`, `issues`, `merge_requests`, `milestones`,
`users`, `snippet_titles`. Two more scopes are gated: `commits` and
`blobs` (code) require Elasticsearch/advanced search on the instance —
Premium/Ultimate on gitlab.com; self-managed only when the admin enabled
it. A 400 naming the scope means the instance does not support it.

## Group and project scoping

```bash
glab api "groups/GROUP%2FSUBGROUP/search?scope=issues&search=TEXT"
glab api "projects/:fullpath/search?scope=merge_requests&search=TEXT"
```

URL-encode `/` in group and project paths as `%2F` (inside a checkout,
`:fullpath` does it for you).

## Filter-based alternatives

Per-project term search is usually better served by list filters, which
need no Search API:

```bash
glab issue list -R G/P --search "TEXT" --in title,description
glab mr list -R G/P --search "TEXT"
```

## MCP capabilities

The search capability (18.4) covers instance/group/project search; the
semantic code search capability (18.7, Beta) finds code by meaning rather
than keywords — it needs GitLab Duo with semantic search enabled on the
instance, so treat its absence as a version/tier gate, not an error.

## REST fallback tier

`rest_read.py search --scope issues --query "TEXT" --host HOST
[--group GROUP]` — see [rest-fallback.md](rest-fallback.md). The search
API needs a token on gitlab.com even for public projects.
