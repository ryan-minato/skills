# git flow

Read when the project already runs git flow, or when the user chooses it after
being told the cost. Do not select it for a new project: the two scenarios it
was designed for — explicitly versioned software and several releases in the
wild — are served more cheaply by the GitLab Flow release-branch variant, and
its own author now directs teams doing continuous delivery to a simpler model.

## The structure

| Branch | Cut from | Merges into | Lifetime |
|---|---|---|---|
| `main` | — | — | permanent; every commit is a release |
| `develop` | `main` | — | permanent; the integration branch |
| `feature/*` | `develop` | `develop` | one change |
| `release/*` | `develop` | `main` **and** `develop` | one release, stabilization only |
| `hotfix/*` | `main` | `main` **and** `develop` | one urgent fix |

Two rules carry the model: nothing merges into `main` except a release or
hotfix branch, and everything that merges into `main` also merges back into
`develop`. Skipping the back-merge is the single most common way the model
breaks, and it breaks silently — the fix is in production and absent from the
next release.

## Operating an inherited git flow

Preserve it. A working git flow is a working model, and migration costs more
than it returns unless something concrete is blocked.

Record these, because they are what the branch names do not say:

- What may still land on a release branch once it is cut, and who decides.
- Who performs the back-merge into `develop`, and when — attached to the
  release, not to someone's memory.
- Whether tags are cut on `main` after the merge or on the release branch
  before it.
- Whether the tooling in use is the `git-flow` command set or plain git with
  these conventions. The two produce different merge shapes and different
  expectations about fast-forwards.

Prefer merge commits over rebase for the release and hotfix merges into `main`.
The model's auditability comes from those merge commits being visible.

## When the cost has come due

These are the signs that the structure is no longer paying for itself. Report
them; migrating is still the user's decision.

- `main` and `develop` are effectively identical because every merge is
  released immediately. The project is doing GitHub Flow with extra steps.
- Release branches live for weeks and accumulate features, which means they are
  a second `develop`.
- Hotfixes are routinely applied to `main` without the back-merge, so `develop`
  ships regressions.
- Only one version is ever supported, which removes the reason the model exists.

## Exit path, when the user asks for one

Migrate on a release boundary, never mid-stabilization: cut and finish the
current release, back-merge it, verify `develop` and `main` agree, then make
`main` the single integration branch and delete `develop`. Move protection
rules and CI triggers to `main` in the same change, or the first push after the
migration lands unprotected.
