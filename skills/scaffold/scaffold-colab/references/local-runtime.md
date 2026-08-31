# Local Colab Runtime

Read when setting up the devcontainer, or when the user wants the Colab web
UI connected to local hardware. The official runtime images track the hosted
Colab environment closely, so a container built from them is a first-pass
validation environment: what runs here will usually run on Colab, though only
a real Colab run confirms it.

## Images

| Runtime | Image |
|---|---|
| CPU | `<region>-docker.pkg.dev/colab-images/public/cpu-runtime` |
| GPU | `<region>-docker.pkg.dev/colab-images/public/runtime` |

`<region>` is `us`, `europe`, or `asia`. The images are byte-identical across
regions; the choice affects only download locality. Suggest the region
nearest the user and let the user decide — never pick silently.

The GPU image corresponds to the hosted T4/L4/A100 runtimes and needs NVIDIA
drivers plus the NVIDIA Container Toolkit on the host (`--gpus=all` at run
time; the GPU devcontainer asset already carries it in `runArgs`). There is
no TPU image — TPU work is validated only on real Colab.

When anything here looks stale — image paths, ports, flags — the first-party
source is https://research.google.com/colaboratory/local-runtimes.html.

## Devcontainer

Copy the skill's `assets/devcontainer-cpu.json` or
`assets/devcontainer-gpu.json` to `.devcontainer/devcontainer.json` and
rework it: resolve `<region>` and
`<project name>`, and drop `google-colab-cli` from `toolsToInstall` when the
user declines the CLI. Keep the python feature's `"version": "none"` — tools
install via pipx without shadowing the image's preinstalled interpreter, and
that preinstalled environment is the point of using this image. Verify the
container builds and `python -c "import sys; print(sys.executable)"` resolves
to the image's Python, not a feature-installed one.

This reference supplies the Colab-specific facts; the generic devcontainer
decisions belong to the `devcontainer-setup` skill when it is installed (the
SKILL.md workflow covers installing or declining it). If the user declines,
this reference alone is the fallback.

## Connecting the Colab web UI to a local runtime

For running the Colab front-end against local hardware (the `runtime` recipe
in the justfile asset):

    docker run -p 127.0.0.1:9000:8080 <image>              # CPU
    docker run --gpus=all -p 127.0.0.1:9000:8080 <image>   # GPU

The container prints a URL containing an auth token. In Colab: Connect →
"Connect to a local runtime", paste the URL. Warn the user before they
connect anything shared: a notebook attached to a local runtime can execute
arbitrary commands on the machine, so only trusted notebooks belong there.
