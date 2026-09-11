## Purpose
Governs what an agent that loaded the `scaffold-ml` builder observably does when it initializes a machine-learning project: the single project shape it establishes, the configuration, provenance, tracker, image, test, hook, and research-task rules it deposits so the generic harness builders keep them, and the builders and durable skills it hands off to.

## ADDED Requirements

### Requirement: Trigger: description
The builder description SHALL open with the disposable-builder marker and SHALL cause the builder to load when an empty or early repository that trains or evaluates models needs its structure, environment, commands, checks, provenance, and agent guidance established, and SHALL not cause it to load for a mature project's migration, for an inference-only application, or for a data pipeline that consumes models without training them.

#### Scenario: Empty training repository
- **WHEN** the user says "this repo will fine-tune a small language model — set it up properly: layout, environment, commands, checks, and the agent guidance"
- **THEN** the builder loads

#### Scenario: Early repository with one script
- **WHEN** the user says "we have a train.py and a notebook; turn this into a proper ML project before the team grows"
- **THEN** the builder loads

#### Scenario: Inference service (near-miss)
- **WHEN** the user says "scaffold a FastAPI service that serves our exported model"
- **THEN** the builder does not load

#### Scenario: Data pipeline (near-miss)
- **WHEN** the user says "set up a reproducible data-cleaning pipeline that runs a pretrained classifier over our records"
- **THEN** the builder does not load

### Requirement: Behavior: One project shape, with existing choices preserved
The builder SHALL establish one shape — a package or flat module chosen by the extraction rule, `configs/` with a typed schema and YAML states, a committed dependency lock from the carrier the user chose, a training entry point that is an explicit loop on an acceleration library with seeding, gradient accumulation, checkpoint save and resume, an evaluation cadence, one logging seam and one stage-trace seam, an evaluation entry point bound to the project's recorded benchmark or evaluation-set identity, a launch command for multi-device runs, `data/raw/` as the local cache of immutable inputs whose identities the manifest records and which no transformation writes into, `outputs/<run_id>/` for run artifacts, focused tests, and an agent entrypoint — SHALL default the framework to PyTorch for an otherwise empty project and use JAX only when the user asks or the project already depends on that ecosystem, SHALL not offer a quick-experiment versus maintainable-project choice, and SHALL keep a working configuration framework, settings library, or requirements workflow the repository already uses, adding only the missing provenance, tracker, and marker rules.

#### Scenario: Empty repository
- **WHEN** the repository has no training code and the user asks for the scaffold
- **THEN** the builder creates the single shape without asking which of two modes to use

#### Scenario: Existing Hydra project
- **WHEN** the repository already runs a Hydra `configs/` tree and the user asks the builder to harden it
- **THEN** the builder keeps Hydra, records the two rules (no object instantiation from configuration; a pinned output directory), adds the manifest, tracker, and marker rules, and does not migrate to OmegaConf alone

#### Scenario: Evaluation before the first experiment
- **WHEN** the project has a training goal but no stated way to judge "better"
- **THEN** the builder asks for the benchmark or evaluation set, records its identity and the metric definitions with the evaluation entry point, and does not leave evaluation as a later task

#### Scenario: Transformation targets the raw cache
- **WHEN** a scaffolded workflow would write derived data under `data/raw/`
- **THEN** the deposited rule directs it elsewhere and the builder writes no code that modifies the raw cache

### Requirement: Behavior: The dependency carrier is the user's choice, a uv project by default
The builder SHALL ask the user which dependency carrier the project uses and SHALL recommend a uv project — `pyproject.toml` with a `dev` dependency group and `uv.lock`, accelerator wheels routed through explicit indexes and sources — as the default, SHALL accept the four-file requirements workflow as the alternative — `requirements.in` for the runtime dependencies and `requirements.dev.in` that includes it plus the development tools, each compiled by uv into a fully pinned `requirements.txt` and `requirements.dev.txt` with the torch backend flag, all four committed together, the training machine syncing the runtime file and a development machine the dev file — SHALL treat the committed runtime lock of either carrier (`uv.lock`, or `requirements.txt`) as the environment identity the manifest records (a requirements file with ranges is not a lock), SHALL keep the carrier an existing repository already uses, and SHALL write the environment stage of the container recipe and the setup and lock commands for the chosen carrier only.

#### Scenario: No preference stated
- **WHEN** the user asks for the scaffold and says nothing about dependencies
- **THEN** the builder recommends the uv project and asks the user to confirm or choose the requirements carrier before writing the environment files

#### Scenario: Requirements carrier chosen
- **WHEN** the user chooses the requirements carrier
- **THEN** the builder writes `requirements.in` and `requirements.dev.in`, the compile commands that produce both pinned files and the sync commands that select the runtime or the dev file per machine with the backend flag, the container's environment stage installs from the runtime file, and the deposited guidance says the compiled files are the locks, are committed with their sources, and are never edited by hand

#### Scenario: uv project chosen
- **WHEN** the user confirms the uv project
- **THEN** the builder writes `pyproject.toml` with the runtime dependencies, a `dev` dependency group for the development tools, the accelerator index and source routing, and commits `uv.lock`; the container's environment stage installs with a frozen sync without the dev group

#### Scenario: Existing requirements project
- **WHEN** the repository already runs a compiled requirements workflow
- **THEN** the builder keeps it and does not ask the carrier question

#### Scenario: Development machine differs from the training box
- **WHEN** developers work on a machine without the training accelerator (a macOS laptop) and train on a Linux GPU host
- **THEN** the uv project gates the accelerator routing with environment markers, or the requirements carrier records one backend per machine in the task runner, and the guidance names which machine uses which

### Requirement: Behavior: The configuration surface is typed values with a resolved dump
The builder SHALL default an unsettled project to a typed schema (dataclasses) merged with YAML states and command-line overrides into one resolved configuration that the training entry point saves under the run's output directory before training starts, SHALL expose only values a run may choose (named choices, never import paths, registries, control flow, or deep inheritance), and SHALL name searched values that are not hyperparameters search variables.

#### Scenario: Optimizer selection
- **WHEN** the scaffolded configuration must let a run choose the optimizer
- **THEN** the schema carries a named choice among the optimizers the code supports and construction stays in code

#### Scenario: Resolved dump
- **WHEN** the scaffolded training entry point starts a run
- **THEN** it writes the merged configuration to `outputs/<run_id>/config.resolved.yaml` before the first step and the manifest records its hash

### Requirement: Behavior: Every run writes a manifest and logs to a selected tracker
The scaffolded entry point SHALL write a run manifest at start (status running) and finalize it at the end — run id, executed commit and dirty flag, resolved-configuration hash, image digest or lock digest, host and runtime facts, input identities, seed, parent run — keeping the start-time record, SHALL send the manifest's scalar fields to the tracker as parameters and tags, and the builder SHALL select the tracker by precedence: a working tracker the project already uses, else the hosting platform's experiment tracking when the platform provides it, else Trackio; and SHALL deposit the snapshot-commit and snapshot-retention rules in the agent guidance.

#### Scenario: New project on GitHub
- **WHEN** the repository is hosted on GitHub and uses no tracker
- **THEN** the builder wires Trackio through the loop's logging seam and records the choice and its reason in the agent guidance

#### Scenario: Existing tracker
- **WHEN** the repository already logs to a working tracker
- **THEN** the builder keeps it, adds the manifest fields to it, and does not propose a migration

#### Scenario: Dirty tree at launch
- **WHEN** the deposited guidance is read by an agent about to launch a run with uncommitted changes
- **THEN** it instructs a snapshot commit on the experiment branch first and names the retention rule that keeps cited snapshots reachable

### Requirement: Behavior: The container recipe yields a recorded image identity
When the user opts into containers, the builder SHALL provide a multi-stage recipe — an environment stage installed from the committed lock on a bare CUDA or ROCm base image, a runtime target, and a sealed target that adds the source — SHALL record the pushed image's digest (or the local image id when never pushed) as the run's environment identity by injecting it into the run, SHALL keep the Dockerfile and any tag out of the identity, SHALL still record the host facts a container cannot pin, SHALL mount `data/`, `outputs/`, and the model-hub cache as volumes with the container-path mapping recorded and raise the container's shared memory for data-loader workers, and, when a preinstalled-stack image is chosen instead, SHALL make the image's framework authoritative by removing it from the project's dependencies and recording that rule; containers SHALL stay opt-in.

#### Scenario: Container requested
- **WHEN** the user asks for a training image
- **THEN** the builder writes the three-stage recipe, a `docker-digest` recipe that prints the digest to inject, and the AGENTS.md rule that the digest, not the tag, is recorded

#### Scenario: No container requested
- **WHEN** the user does not ask for a container
- **THEN** the builder scaffolds no Dockerfile, Compose file, or dev container and records the lock digest as the environment identity

#### Scenario: Preinstalled-stack image chosen
- **WHEN** the user chooses a vendor image with the framework preinstalled
- **THEN** the builder removes the framework from the project's dependencies, records that the image's stack is authoritative, and still records the image digest as the identity

### Requirement: Behavior: Tests, hooks, and style follow the experiment standard
The builder SHALL configure pytest with `slow` and `gpu` markers, a default suite that is light and CPU-compatible, a `test-gpu` command whose tests fail when hardware is absent, and heavy validation behind an explicit command; SHALL configure Ruff near its defaults with line length 120, docstring code formatting, and a docstring code line length of 80; SHALL not install a static type checker as a gate over model code and SHALL say where static checking may apply (configuration and control-plane modules); SHALL set docstring conventions that carry shape, dtype, device, and mask semantics; and SHALL configure git hooks that run the formatter and linter only, never tests; each of these SHALL be deposited with its reason so the Python defaults builder that runs later inventories it as a settled choice.

#### Scenario: GPU tests requested to skip
- **WHEN** the user asks that GPU tests skip automatically when no GPU is present
- **THEN** the builder keeps them failing under `test-gpu`, excludes them from the default suite, and explains that a silent skip hides a broken test

#### Scenario: Hook runs tests
- **WHEN** the repository's existing pre-commit hook runs pytest
- **THEN** the builder proposes limiting the hook to the formatter and linter, names snapshot-commit latency as the reason, and follows the user's decision

#### Scenario: Python defaults builder runs afterwards
- **WHEN** the Python defaults builder inventories the project after the scaffold's deposit
- **THEN** it finds the typing, test-suite, and hook decisions recorded with their reasons and treats them as settled rather than proposing its baseline

### Requirement: Behavior: The research-task convention is deposited for the project's spec tooling
The builder SHALL deposit the research-task convention — a research spec with an Objective and an Evaluation (Context, Search Scope, Constraints, Completion Condition, Hypotheses as needed), one research task per pull or merge request carrying its hypotheses and runs, the spec evolving while run history stays immutable, completion on the completion condition with negative results as valid outcomes — into the project's agent guidance, SHALL place the spec inside the project's specification contract when one exists and otherwise under `research/<task>/`, SHALL, when the project's spec tool is OpenSpec, copy the bundled `research-task` schema into the tool's schema directory and record how a change selects it, and SHALL record the research task as the project's unit of research work so the workflow-design and specification-workflow builders that run later keep it.

#### Scenario: OpenSpec project
- **WHEN** the repository runs OpenSpec and the user asks for the scaffold
- **THEN** the builder deposits the `research-task` schema at the tool's schema path, the selection rule in the agent guidance, and the research section with the spec's fields

#### Scenario: No spec tool
- **WHEN** the repository runs no spec tool
- **THEN** the builder deposits the research section pointing at `research/<task>/spec.md` and the skeleton, and installs no tool

#### Scenario: Specification workflow builder runs afterwards
- **WHEN** the specification workflow builder inspects the project after the scaffold's deposit
- **THEN** it finds the research-task schema and convention as a spec tool and layout the project already runs and keeps them

### Requirement: Behavior: The deposited guidance makes the project discoverable
The deposited agent entrypoint SHALL name how to run the standard experiment and its evaluation, where configuration enters and how it is overridden, how tests run and which suites exist, where the research spec of the current task lives (the project's specification contract when one exists, else `research/<task>/`), which tracker holds runs, the provenance rules, the error-handling convention (catch only expected data or environment failures; fail early with context for everything else), a code-style section that states the abstraction rule (share only what must stay consistent; duplicate what may diverge), the preference for mature first-party dependencies over unmaintained research code, the explicit loop, block-level comments for mathematical intent and tensor layout, hot-path performance kept over form, and profiling only on demand, the rule that data, weights, checkpoints, and credentials are never committed, and a when-to-read table; it SHALL keep one authoritative source per fact; and it SHALL not carry the disposable marker.

#### Scenario: Entrypoint read after disposal
- **WHEN** a fresh agent reads the deposited entrypoint with the builder deleted
- **THEN** it can name the train command, the configuration override syntax, the test suites, the research-spec location, the tracker, and the snapshot rule from the entrypoint alone

#### Scenario: Marker leak
- **WHEN** the builder copies an asset into the project
- **THEN** no generated file carries the disposable-builder marker

### Requirement: Behavior: The scaffold is verified before the handoff
Before handing off, the builder SHALL run the deposited commands — environment setup, lint, the default test suite, and a smoke run of the training and evaluation entry points on a tiny input — confirm that every link in the deposited entrypoint resolves and that no generated file carries the disposable marker, and inspect the result with the user; it SHALL report any command that could not run on the current machine (no accelerator, no network) instead of skipping it silently.

#### Scenario: Fresh checkout
- **WHEN** the deposit is complete
- **THEN** the builder runs setup, lint, tests, and the smoke run and shows the user the outcome of each before naming the next builder

#### Scenario: No accelerator on the build machine
- **WHEN** the machine has no accelerator and the GPU suite cannot run
- **THEN** the builder reports the GPU suite as not run here, records the command to run it on the training host, and does not mark the scaffold verified for that suite

### Requirement: Handoff: GPU container environments
When containers are requested, the builder SHALL make the base-image, tag-discovery, and GPU-wiring decisions with the GPU container builder of the harness catalog, installing the whole catalog through the installing skill when it is absent and never running an install command, and SHALL, when the user declines, proceed with its own container reference alone.

#### Scenario: Handoff offered
- **WHEN** the user asks for a training image and the GPU container builder is not installed
- **THEN** the builder routes the whole harness catalog's installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the installation
- **THEN** the builder writes the recipe from its own reference and records that the base image was not verified live

### Requirement: Handoff: work tracking and agent authority
The builder SHALL leave work-tracking, planning, and agent-autonomy rules to the workflow-design and agent-authority builders of the harness catalog, routed through the installing skill, and SHALL, when the user declines, leave management design out and record the gap.

#### Scenario: Handoff offered
- **WHEN** the scaffold reaches the point of recording how work is tracked
- **THEN** the builder names the workflow-design and authority builders and routes their installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines
- **THEN** the builder records in the handoff that management design was not covered and invents no issue or review flow

### Requirement: Handoff: durable machine-learning skills
The builder SHALL name the durable skills for run provenance, the research task, experiment code conventions, training instrumentation, and training diagnosis by role as the project's ongoing guidance, routed through the installing skill, and SHALL, when the user declines, leave the deposited entrypoint rules standing alone with a note that the durable skills were not installed.

#### Scenario: Handoff offered
- **WHEN** the deposit is complete
- **THEN** the builder names the five roles, routes their installation through the installing skill, and records in the entrypoint's when-to-read table where each applies

#### Scenario: User declines
- **WHEN** the user declines
- **THEN** the entrypoint keeps the minimum rules (manifest, snapshot, tracker, markers, hooks) and notes that the durable skills are not installed

### Requirement: Handoff: harness entry and closing
The builder SHALL hand the rest of the harness and the closing of the build to the harness entry builder of the harness catalog, and SHALL, when the user declines, close itself: ask whether to delete the disposable builders and, on that decision, load the disposal builder.

#### Scenario: Handoff offered
- **WHEN** the scaffold's deposit is verified
- **THEN** the builder names the harness entry builder for entrypoint depth, knowledge, project skills, and the closing step

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the builder asks whether to delete the builders now, treats the build request as no consent, and loads the disposal builder only on an explicit decision
