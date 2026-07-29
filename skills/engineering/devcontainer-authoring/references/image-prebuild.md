# Image prebuild

Read this in full before prebuilding or publishing a shared dev image.
`devcontainer build` executes a configuration's Dockerfile and Features
**once** and bakes the merged result — including the config itself, as
the `devcontainer.metadata` image label — into a pushable image.
Consumers then reference that image and inherit the environment without
rebuilding it.

## Build and push

```bash
# Verify locally first (loads into the local docker daemon):
devcontainer build --workspace-folder .

# Then publish:
devcontainer build --workspace-folder . \
  --image-name ghcr.io/acme/devimage:1 \
  --image-name ghcr.io/acme/devimage:2026-07-06 \
  --push true
```

- `--image-name` is repeatable — publish one **mutable** tag consumers
  track (`:1` or `:latest`) plus one **immutable** tag (date or commit
  SHA) for pinning and rollback.
- `--config` selects a non-default devcontainer.json;
  `--platform linux/amd64,linux/arm64` builds multi-arch (needs
  BuildKit + QEMU or native runners);
  `--cache-from ghcr.io/acme/devimage:1` reuses published layers.

## What gets baked: the devcontainer.metadata label

The label embeds an array of config fragments — every applied Feature's
contribution plus the build config's properties (remoteUser, env,
mounts, customizations, lifecycle commands...). At create time tools
merge that array with the consumer's devcontainer.json appended last:
booleans OR together, arrays union, per-key/single values take the last
(consumer) value, and lifecycle commands **accumulate**. Two rules of
thumb fall out:

- Consumers can override single-valued settings but can never subtract a
  baked mount/env/capability/lifecycle command — removing baked behavior
  means republishing the image.
- Bake what is expensive and stable (toolchains, Features, system
  packages); leave project- and person-specific setup (dependency
  install, dotfiles) to consumer lifecycle commands.

Read [image-metadata-merge.md](image-metadata-merge.md) when a
consumer config behaves differently than its text suggests, or when
deciding what to bake versus leave to consumers.

## Consumer configuration

Consumers keep a thin config:

```jsonc
{
  "name": "myproject",
  "image": "ghcr.io/acme/devimage:1",
  "postCreateCommand": "uv sync"   // project install stays consumer-side
}
```

Safe to add on top: `customizations`, extra lifecycle commands, mounts,
ports. Adding **features** to the consumer config triggers a local build
again — if a feature is needed by everyone, bake it and republish
instead.

## CI automation

Use the `devcontainers/ci` GitHub Action (`imageName`, `imageTag`,
`push`, `cacheFrom`, `platform`, `configFile`; wrapper over the CLI),
triggered on changes to `.devcontainer/**`, pushing the mutable + SHA
tags with `cacheFrom` pointing at the previous image. Also set
`"build": { "cacheFrom": "<image>" }` in the source devcontainer.json so
local rebuilds reuse published layers. Read
[image-ci.md](image-ci.md) when creating or
modifying the pipeline (full annotated workflow, multi-arch, non-GitHub
CI via the raw CLI, post-publish verification).

## Gotchas

- `devcontainer build` does **not** run lifecycle commands —
  `postCreateCommand` work is not "prebuilt". Anything you want baked
  must happen in the Dockerfile or a Feature; `onCreateCommand` runs at
  the consumer's first create, not at build.
- Deleting a setting from the consumer config does not remove it from
  the image (the metadata fragment still applies) — republish to remove.
- A mutable tag goes stale on dev machines: consumers must `docker pull`
  (or "rebuild without cache") to pick up a republished `:1`; pinning
  the immutable tag avoids the ambiguity.
- Without an arm64 build (or manifest list), Apple Silicon consumers
  silently fall back to emulation or local rebuilds.
- The publishing registry ref is outside the `devcontainer-setup`
  trusted-source defaults consumers may be operating under; teams should
  add their own image registry to their trust policy explicitly.

## Doc references

- Image metadata spec: <https://raw.githubusercontent.com/devcontainers/spec/main/docs/specs/image-metadata.md>
- Merge-logic table: <https://raw.githubusercontent.com/devcontainers/spec/main/docs/specs/devcontainer-reference.md>
- devcontainers/ci action inputs: <https://raw.githubusercontent.com/devcontainers/ci/main/action.yml>
