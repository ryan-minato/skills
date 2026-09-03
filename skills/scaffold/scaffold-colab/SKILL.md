---
name: scaffold-colab
description: >-
  Disposable builder skill (delete after the harness is built): scaffolds a
  Colab-centric project — a repository whose deliverable is Google Colab
  notebooks, mirrored locally and validated in an official Colab-runtime
  devcontainer before running on real Colab. Use when creating or hardening a
  repository that produces notebooks people open and run in Colab — tutorials,
  course materials, runnable demos, shareable analyses — or when a project
  uses Colab as its runtime, including driving Colab VMs as disposable
  compute for local scripts. Not for generic Jupyter-notebook work, projects
  where notebooks are incidental exploration beside a library or pipeline, or
  mature migrations; preserves working choices.
license: Apache-2.0
compatibility: Local runtime validation requires Docker; colab-mcp requires uv (uvx).
---

# Colab-Centric Project Scaffold

Build only after inspecting what the notebooks deliver, who reads and runs
them, which runtime they need (CPU, GPU, TPU), and every working tool or
structure already present. Present the intended concrete project shape to the
user before creating anything.

## Choose the project shape

- **Notebook project (default)**: the deliverable is Colab notebooks. The
  repository root holds `.ipynb` files as local mirrors of the remote Colab
  notebooks, and the full workflow below applies.
- **Ephemeral runtime**: Colab is only a disposable compute backend — local
  scripts run on temporary VMs through `google-colab-cli`, and there may be
  no notebook at all. Choose this only when the user's goal is explicitly to
  borrow Colab hardware for scripts. The CLI then becomes required rather
  than asked-about, colab-mcp becomes optional, and the mirror rule, notebook
  doctrine, and cell-syntax references do not apply; steps 2, 6, and 7 below
  reduce to scripts plus the CLI run loop described in
  [references/colab-connection.md](references/colab-connection.md).

Do not blend the shapes. When in doubt, ask the user; default to the notebook
project.

## Workflow

1. Confirm the project shape, then what the notebooks teach or demonstrate,
   the audience, the runtime type (CPU, GPU, TPU), whether cell outputs are
   committed, and the current repository state.
2. Lay out the repository. Notebooks are the product and live in the
   repository root: a single-notebook project keeps one `notebook.ipynb`; a
   multi-notebook project keeps several descriptively named root `.ipynb`
   files — no `notebooks/` subdirectory. Each file is the local mirror of a
   remote Colab notebook and the two must stay identical. Around them goes a
   thin shell: `.editorconfig`, pre-commit config,
   [assets/justfile](assets/justfile), and
   [assets/agents-md.md](assets/agents-md.md) copied to `AGENTS.md`. Rework
   every line of every copied asset; an unresolved placeholder is an unmade
   project decision. Do not create `src/` machinery — reuse and abstraction
   belong in Python libraries, not in a notebook project.
3. Treat the Colab preinstalled environment as the dependency baseline. Read
   [references/environment-debugging.md](references/environment-debugging.md)
   before adding, pinning, or upgrading any package, and whenever local
   behavior differs from real Colab.
4. Set up the devcontainer on the official Colab runtime image: read
   [references/local-runtime.md](references/local-runtime.md) first. Suggest
   a registry region (`us`, `europe`, `asia` serve identical images) but let
   the user decide it. Copy and rework the matching devcontainer asset. Make
   generic devcontainer decisions with the `devcontainer-setup` skill; if it
   is not installed, load the `ryan-minato-skills-installing` skill and
   install `devcontainer-setup` as it directs — never run an install command
   yourself. If the user declines, proceed with the reference alone.
5. Wire the real-Colab connection: read
   [references/colab-connection.md](references/colab-connection.md). Register
   colab-mcp in the client's project-scoped MCP configuration, then ask the
   user whether to also install `google-colab-cli` as an auxiliary tool —
   never install it unprompted.
6. Author the notebooks: read
   [references/notebook-doctrine.md](references/notebook-doctrine.md) before
   writing or reviewing any notebook cell. Read
   [references/colab-forms.md](references/colab-forms.md) when adding
   parameters, cell titles, or hidden-code forms, and
   [references/colab-markdown.md](references/colab-markdown.md) when writing
   markdown cells.
7. Validate and synchronize. Edit the local mirror, execute it headlessly in
   the local runtime container as a first pass, then run it in a real Colab
   session through colab-mcp for final confirmation. After a change on either
   side, synchronize the other immediately — colab-mcp is the primary
   channel, `colab upload` / `colab download` the auxiliary one when the CLI
   is installed. The mirror and the remote notebook must never drift.
8. Work tracking, planning, and agent-autonomy rules are designed with the
   `meta-workflow-design` and `meta-agent-authority` skills, not improvised
   here — do not invent an issue, review, or autonomy flow in the scaffold.
   If they are not installed, load the `ryan-minato-skills-installing` skill
   and install the whole `meta` catalog at project scope as it directs (its
   builders stack and are disposed together); never run an install command
   yourself. If the user declines, leave management design out and record
   the gap in
   the handoff.
9. Deposit the durable rules — mirror consistency, dependency baseline,
   notebook doctrine digest, validation loop — into the target `AGENTS.md`.
   Never copy this skill's disposable marker into any generated file.
10. Run the target repository's checks and inspect the result with the user.
11. Hand the rest of the harness — entrypoint depth, knowledge, project skills,
    synchronization — and the closing of the build to the `meta-harness-
    building` skill; the same `meta` catalog install from step 8 covers it when
    it is not installed. If the user declines that handoff, close here: once
    the deposit is verified and before the work goes to review, ask the user
    whether to delete the disposable builders now — the build request is not
    deletion consent — and on that decision load `meta-disposal`, which lists,
    confirms, and removes them. If they keep the builders, leave them in place
    and out of every commit, and record it in the handoff.

Done when: a fresh checkout opens in the devcontainer, colab-mcp is
registered and reachable, every root notebook matches its remote Colab
counterpart, the first notebook follows the doctrine and runs clean both in
the local container and on real Colab, and the durable rules are recorded in
the target AGENTS.md.

- Disposable builders never enter a commit: before the first commit, add
  every skill directory whose description opens with
  `Disposable builder skill (delete after the harness is built):` to
  `$(git rev-parse --git-path info/exclude)`, stage explicit paths, and read
  `git status` before each commit.

## Gotchas

- The local runtime image approximates Colab; it does not equal it. Drive
  mounting, Google auth flows, and form-control rendering exist only in real
  Colab — a local pass is preliminary, only a real Colab run confirms.
- pip upgrades inside the preinstalled environment cascade through its
  tightly coupled dependency set. Add missing packages; upgrade only when
  unavoidable.
- The regional registries are byte-identical mirrors. The choice is download
  locality, never a technical difference — and it is the user's to make.
- Prefer `# @param` form comments over ipywidgets: they render natively in
  Colab and degrade to harmless comments everywhere else.
