## Purpose
Governs what an agent that loaded the `meta-gpu-container` builder observably does when it establishes GPU container environments for a project: whether a container is warranted, the base image and its live tag, GPU wiring, and the image identity it deposits.

## ADDED Requirements

### Requirement: Trigger: description
The builder description SHALL open with the disposable-builder marker and SHALL cause the builder to load when a project must run GPU workloads in containers, when a CUDA or ROCm base image or its tag must be chosen or refreshed, or when a scaffold builder defers GPU container setup, and SHALL not cause it to load for CPU-only containers, application deployment images, or authoring dev container Features and Templates.

#### Scenario: GPU container requested
- **WHEN** the user says "containerize the training so it runs on the GPU server with the right CUDA"
- **THEN** the builder loads

#### Scenario: CPU-only container (near-miss)
- **WHEN** the user says "put the documentation site build into a container"
- **THEN** the builder does not load

### Requirement: Behavior: The image digest is the environment identity
The builder SHALL structure the recipe in stages — an environment stage installed from the project's committed lock, a runtime target, and a sealed target that adds the source — SHALL deposit the rule that the pushed image's digest (or the local image id when never pushed) is recorded with each run as the environment identity and that the Dockerfile and any tag are not identities, SHALL show how the digest is obtained and injected into the run, and SHALL deposit the rule that host facts a container cannot pin — GPU model and count, driver, kernel, fabric topology — are recorded beside the digest for any performance claim.

#### Scenario: Deposit read after disposal
- **WHEN** a fresh agent reads the deposited container rules
- **THEN** it can name where the digest comes from, how the run receives it, and which host facts accompany it

#### Scenario: Tag offered as identity
- **WHEN** the user proposes recording the image tag with each run
- **THEN** the builder records the digest instead and explains that a tag moves
