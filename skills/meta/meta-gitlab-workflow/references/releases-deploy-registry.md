# Releases, Deployments, and Registries

Read when the project publishes releases or packages, deploys to environments,
uses review apps, or stores container, package, infrastructure, or model
artifacts.

Resolve the selected `releases`, `changelogs`, `environments`, `deployments`,
`review apps`, `package registry`, `container registry`, and relevant package
format topics through `llms.txt`. Verify the package manager and target
instance rather than copying a generic upload command.

## Release contract

Derive versioning and tag style from manifests, tags, changelog, and history.
Preserve a working scheme. If none exists, recommend SemVer for public
compatibility-bearing software and let the user choose; never apply the bundled
version calculator to CalVer or a custom scheme.

Record the single changelog source, version owner, tag format, release trigger,
artifact set, signing/provenance requirements, milestone relationship, rollback
or yanking procedure, and who approves publication. Fix and verify a manual
release path before automating it.

Rework `assets/versioning-policy.md` when a durable release contract is
selected. Rework `assets/changelog-config.yml` only when the
project chooses GitLab-generated changelog entries; its category keys must
exactly match the approved commit trailers.

## Environments and deployments

Inventory actual targets and classify development, test, staging, production,
and dynamic review environments. Record environment ownership, protected
deployment rules, environment-scoped variables, concurrency/resource locking,
health verification, rollback, cleanup, and retention.

Deploy only immutable identified artifacts that passed the agreed gates. Review
apps need a stop/cleanup path and cost owner. Production remains manually
approved by default even when other agent operations are unattended.

## Registries

Choose registry types from real products: language packages, generic build
artifacts, containers, Terraform modules, or models. For each, record naming,
version immutability, authentication, consumers, promotion, signing/SBOM,
retention, cleanup, and deletion authority. Do not use expiring CI job artifacts
as permanent release assets.

Validate a publish with a disposable prerelease or dry run where the ecosystem
supports it, then install/pull the exact result as a consumer would. All tokens
stay out of commands, logs, committed config, and descriptions.
