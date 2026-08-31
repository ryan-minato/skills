# Connecting to Real Colab

Read when registering the tools that reach real Colab sessions: colab-mcp
(required in the notebook shape) and google-colab-cli (asked-about in the
notebook shape, required in the ephemeral-runtime shape).

## colab-mcp (required for notebook projects)

colab-mcp bridges the local agent to a Colab session in the user's browser.
It is the final-validation channel (run the mirror on real Colab) and the
primary mirror-synchronization channel. Register it in the client's
project-scoped MCP configuration; the skill's `assets/mcp-config.json`
carries the entry:

    {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }

For Claude Code that means `.mcp.json` at the repository root; other clients
have their own project-level mechanism — write it there, not into a global
config. Requirements: `uv` on the host (the devcontainer assets install it)
and a client that supports the MCP `notifications/tools/list_changed`
notification. When registration or the handshake fails, resolve current
facts from the first-party https://github.com/googlecolab/colab-mcp.

## google-colab-cli (ask first)

Ask the user whether to install `google-colab-cli` as an auxiliary tool;
never install it unprompted. If declined, remove it from the devcontainer
`toolsToInstall`. The `colab` command provisions and drives Colab VMs from
the terminal:

- Sessions: `colab new` (flags select GPU/TPU/high-memory), `colab status`,
  `colab sessions`, `colab stop`.
- Execution: `colab exec` (stdin, `.py`, or `.ipynb`), `colab run` (temporary
  VM with auto-teardown), `colab repl`, `colab console`.
- Files: `colab upload` / `colab download` — the auxiliary mirror-sync path.
- Extras: `colab auth`, `colab drivemount`, `colab install`.

## Ephemeral-runtime shape

When the project only borrows Colab hardware for scripts, the CLI is the
primary tool and is required. The loop: `colab new` a session sized to the
workload (or `colab run` for one-shot execute-and-teardown), `colab exec`
the local script, `colab download` any artifacts, `colab stop`. colab-mcp
becomes optional — add it only when the user also wants browser-session
access.
