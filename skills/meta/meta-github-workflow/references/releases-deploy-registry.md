# Releases, Deployments, and Registries

Read when the project publishes releases or packages, deploys to
environments, uses Pages, or stores container, package, or model
artifacts.

## Releases: labels in, notes out

Generated release notes consume merged **pull-request labels** through
`.github/release.yml`: categories map labels to sections, evaluated in
order, with the `*` catch-all last — an unlabeled PR lands in the
catch-all, and a category naming a nonexistent label is silently empty.
The label taxonomy and the release notes are one system; design and check
them together (the taxonomy check covers this edge).

Release mechanics worth encoding: a release is tag plus notes plus assets
(up to 1000 assets, 2 GiB per file); under **immutable releases** create a
draft first and attach every asset before publishing, because publication
freezes the release; `--generate-notes` works from the compare range;
`--latest` does not automatically track the highest semver; deleting a
release does not delete its tag. Draft releases are collaborator-only —
the one draft surface on GitHub that is actually private.

Versioning: record the version source, tag format, and bump rules in the
committed versioning policy; ship
`scripts/next_version.py` only when the project
chose SemVer, and pair the tag-check workflow with any tag ruleset so
format and protection agree.

## Environments and deployments

Environments gate deployment workflows. Required reviewers and wait timers
are free on public repositories; on a **private** repository they need a
paid plan, so confirm the probed plan before promising them. Where the plan
does not carry them, deploy authority is a ruleset condition or a named
human step — say which.
Prefer OIDC to long-lived cloud secrets; deployment workflows carry
`queue`-shaped concurrency, not cancel-in-progress. Record each
environment, its approvers, and its secrets (names only) in
`platform-settings.md`.

## Pages

Pages is the docs-hosting default when the project wants rendered
documentation: repository-sourced, built by an Actions workflow, no wiki
limitations. Record the build workflow in the job registry like any other.

## Registries

Packages (npm, Maven, NuGet, RubyGems, Gradle) and the container registry
(`ghcr.io`, OCI) hold binary artifacts; public packages are free, private
ones meter against plan quotas. Git LFS carries per-file plan limits.
**Actions artifacts expire — 90 days by default, capped at 90 public /
400 private — and are never release assets, never a registry.** Publishing
workflows hold the narrowest token (`packages: write`) and pin their
publish action by SHA if third-party.

Done when: the release procedure, versioning policy, environment
authority, and registry rules are deposited; `release.yml` categories
reference only existing labels; and nothing durable depends on an Actions
artifact surviving.
