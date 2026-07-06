# devcontainer.metadata merge behavior

Read this when a consumer's devcontainer.json behaves differently than its
text suggests, or when deciding what to bake into the image versus leave
to consumers. Authoritative table ("Merge Logic"):
<https://raw.githubusercontent.com/devcontainers/spec/main/docs/specs/devcontainer-reference.md>

## The mechanism

`devcontainer build` embeds a `devcontainer.metadata` image label holding
an **array** of config fragments: one per applied Feature plus the
build-time devcontainer.json's relevant properties. At create time, tools
merge that array **with the consumer's devcontainer.json appended last**,
property by property. For "last value wins" properties, that means the
consumer's config wins; for accumulating properties, nothing is ever
removed by the consumer — only added.

## Per-property merge rules

| Property | Merge logic |
|---|---|
| `init`, `privileged` | `true` if **any** fragment says true (OR). |
| `capAdd`, `securityOpt` | Union without duplicates. |
| `entrypoint` | All collected entrypoints run. |
| `mounts` | Collected; on a target conflict, last source wins. |
| `onCreateCommand`, `updateContentCommand`, `postCreateCommand`, `postStartCommand`, `postAttachCommand` | Collected — **every** fragment's command runs, image-baked ones first. |
| `containerEnv`, `remoteEnv` | Per variable, last value wins. |
| `containerUser`, `remoteUser`, `userEnvProbe`, `overrideCommand`, `shutdownAction`, `updateRemoteUserUID`, `waitFor` | Last value wins (consumer config, being last, can override). |
| `forwardPorts` | Union without duplicates. |
| `portsAttributes` | Per **port** (not per attribute), last value wins. |
| `customizations` | Merging is tool-specific (VS Code unions extension lists). |
| `hostRequirements` | Max value wins. |

## Consequences that surprise people

- **You cannot subtract by editing the consumer config.** A mount, env
  var, capability, or lifecycle command baked into the image stays active
  even after deleting it from the consumer's devcontainer.json — the
  image fragment still supplies it. Removing baked behavior requires
  republishing the image.
- **Lifecycle commands accumulate, not replace.** If the image was built
  from a config with a `postCreateCommand` and the consumer sets another
  one, **both** run (image's first). Keep baked lifecycle commands
  idempotent and cheap, or leave lifecycle to consumers entirely.
- **Booleans are sticky.** One Feature setting `privileged: true` at
  build time makes every consumer container privileged; no consumer
  setting can turn it off.
- **`remoteUser` is overridable.** The consumer's value wins because
  their config merges last — useful when the baked default is wrong for
  one team.
- **Feature-contributed metadata rides along.** Everything the baked
  Features declared (env, mounts, capAdd, customizations) applies to
  consumers even though their config never mentions the features — that
  is the point of prebuilding, but it also means image audits must read
  the label, not just the source config. Inspect with:

  ```bash
  docker inspect <image> \
    --format '{{ index .Config.Labels "devcontainer.metadata" }}' | jq .
  ```

## Design guidance

Bake what is expensive and stable (toolchains, features, system
packages); leave to consumers what is project- or person-specific
(project dependency installs, dotfiles, editor extensions beyond a shared
baseline). When both sides must cooperate, prefer baked `onCreateCommand`
(prebuild-friendly) plus consumer `postCreateCommand` (project install).
