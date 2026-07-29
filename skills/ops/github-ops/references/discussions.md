# Discussions

Loaded when the task touches GitHub Discussions. Reads are research and
need no gate. Posting or replying in a discussion is publishing: the
pre-publish gate applies, and gh has no discussion write command — use the
MCP capability if the connected server provides one; otherwise tell the
user writing is not available on the current path.

| Task | MCP capability | gh |
|---|---|---|
| List discussions | list discussions (when ordering, the direction parameter is required too) | GraphQL block below |
| Read one discussion | read a discussion by number | GraphQL block below |
| Read its comments | read a discussion's comments | GraphQL block below |
| List categories | list discussion categories | GraphQL block below |

gh has no first-class `gh discussion` command (the gh team declined to add
one), so every Discussions read on the gh path goes through
`gh api graphql`. Each block below is complete and copy-paste ready:
replace `O`/`R` (and `NUMBER`/`TEXT`) with real values. Pass variables
with `-F` and the query with `-f query='...'`. Keep the field sets as
small as the task allows — every selected field lands in agent context.

## List recent discussions

The 20 most recently updated discussions, newest first.

```bash
gh api graphql -F owner='O' -F name='R' -f query='
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    discussions(first: 20, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        title
        category { name }
        author { login }
        updatedAt
        url
      }
    }
  }
}'
```

## List discussion categories

Category ids and names, needed to interpret or filter listings.

```bash
gh api graphql -F owner='O' -F name='R' -f query='
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    discussionCategories(first: 25) {
      nodes { id name description }
    }
  }
}'
```

## Read one discussion (with comments)

Full body plus the first 30 comments of discussion NUMBER.

```bash
gh api graphql -F owner='O' -F name='R' -F number=NUMBER -f query='
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    discussion(number: $number) {
      title
      body
      author { login }
      category { name }
      url
      comments(first: 30) {
        nodes { author { login } body createdAt }
      }
    }
  }
}'
```

## Paginated listing (more than 100 items)

For `--paginate` to work, the query must declare `$endCursor: String`,
pass it as `after: $endCursor`, and select
`pageInfo { hasNextPage endCursor }`; gh then follows the cursor until
the last page.

```bash
gh api graphql --paginate -F owner='O' -F name='R' -f query='
query($owner: String!, $name: String!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 100, after: $endCursor,
                orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes { number title updatedAt url }
      pageInfo { hasNextPage endCursor }
    }
  }
}'
```

## Search discussions

Full-text search scoped to one repository; TEXT is the search phrase.

```bash
gh api graphql -f query='
query {
  search(query: "repo:O/R TEXT", type: DISCUSSION, first: 20) {
    nodes {
      ... on Discussion { number title url }
    }
  }
}'
```

## REST fallback tier

`rest_read.py discussions` / `discussion --number N` — see
[rest-fallback.md](rest-fallback.md). Tokenless access is best-effort: the
list comes from the public Atom feed (latest ~25, no category filter) and
a single discussion is text extracted from the HTML page; a token in
`GH_TOKEN`/`GITHUB_TOKEN` upgrades both to full-fidelity GraphQL.

## Gotchas

- The Discussions toolset is not in the GitHub MCP server's default set —
  if other GitHub tools exist but no discussion capability does, the
  toolset must be enabled server-side (see
  [tooling-setup.md](tooling-setup.md)).
- Discussion numbers are per-repository and not shared with issue/PR
  numbers: discussion 42 and issue #42 are unrelated items.
