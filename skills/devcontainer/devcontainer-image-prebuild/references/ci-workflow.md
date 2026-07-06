# CI automation for prebuilt dev container images

Read this when creating or modifying the pipeline that builds and
publishes the image. Action input reference:
<https://raw.githubusercontent.com/devcontainers/ci/main/action.yml>

## GitHub Actions (devcontainers/ci)

```yaml
name: prebuild-devcontainer

on:
  push:
    branches: [main]
    # Rebuild only when the container definition changes.
    paths: [".devcontainer/**"]
  workflow_dispatch: {}

# Least privilege: reading the repo and pushing the package is all this
# job does.
permissions:
  contents: read
  packages: write

jobs:
  prebuild:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      # Only needed for multi-architecture builds.
      - uses: docker/setup-qemu-action@v3

      - uses: devcontainers/ci@v0.3
        with:
          imageName: ghcr.io/${{ github.repository }}/devcontainer
          # Mutable moving tag + immutable SHA tag (comma-separated).
          imageTag: latest,${{ github.sha }}
          # Reuse layers from the previous published image.
          cacheFrom: ghcr.io/${{ github.repository }}/devcontainer:latest
          platform: linux/amd64,linux/arm64
          push: always
```

Adaptation points:

- `imageName`/registry: any OCI registry works; swap the login step
  accordingly.
- `imageTag` strategy: keep one mutable tag consumers track (`latest` or
  a major like `1`) plus one immutable tag per build (`${{ github.sha }}`
  or a date) for reproducible pinning and rollback.
- `push` (default `filter`): `filter` pushes only when
  `refFilterForPush`/`eventFilterForPush` match (defaults: any ref, event
  `push`) — useful to build-without-push on pull requests and push on
  main. `always`/`never` are explicit overrides.
- `platform`: drop the QEMU step and this input for single-arch. Without
  an arm64 variant (or manifest list), Apple Silicon users fall back to
  local rebuilds or emulation.
- `cacheFrom` also belongs in the source `.devcontainer/devcontainer.json`
  (`"build": { "cacheFrom": "ghcr.io/..." }`) so local rebuilds reuse the
  published layers too.
- Pin the action to a full release tag (e.g. `devcontainers/ci@v0.3.x`,
  see its releases page) when supply-chain policy requires immutable
  action versions; `@v0.3` is the documented moving tag.
- To validate the environment during the same run, add `runCmd` (e.g.
  run the test suite inside the freshly built container) — the action
  starts the container after building when `runCmd` is present.

## Any other CI: the raw CLI

The action is a wrapper over `@devcontainers/cli`; the equivalent
pipeline step is:

```bash
npm install -g @devcontainers/cli@0.87.0   # or npx -y @devcontainers/cli@0.87.0
docker login ghcr.io -u "$CI_USER" -p "$CI_TOKEN"
devcontainer build \
  --workspace-folder . \
  --image-name ghcr.io/acme/devimage:latest \
  --image-name ghcr.io/acme/devimage:"$COMMIT_SHA" \
  --cache-from ghcr.io/acme/devimage:latest \
  --platform linux/amd64,linux/arm64 \
  --push true
```

Verified flags: `--workspace-folder`, `--config` (non-default
devcontainer.json path), `--image-name` (repeatable), `--cache-from`,
`--cache-to`, `--platform`, `--push`, `--no-cache`, `--output`,
`--additional-features`, `--log-level`. Multi-platform pushes require
BuildKit with QEMU (or native runners per architecture).

## Verifying the published image

After the pipeline runs, confirm the artifact actually works before
telling consumers to switch:

```bash
docker pull ghcr.io/acme/devimage:latest
docker inspect ghcr.io/acme/devimage:latest \
  --format '{{ index .Config.Labels "devcontainer.metadata" }}' | jq .
```

Then open a scratch project whose devcontainer.json is just
`{ "image": "ghcr.io/acme/devimage:latest" }` and check the expected
tools, user, and customizations are present.
