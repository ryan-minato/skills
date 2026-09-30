## MODIFIED Requirements

### Requirement: Behavior: The container recipe yields a recorded image identity
When the user opts into containers, the builder SHALL provide a multi-stage recipe — an environment stage installed from the committed lock on a bare CUDA or ROCm base image, a runtime target, and a sealed target that adds the source — SHALL record the pushed image's digest (or the local image id when never pushed) as the run's environment identity by injecting it into the run, SHALL keep the Dockerfile and any tag out of the identity, SHALL have the sealed build run the git reads it depends on — the working-tree status and the commit it stamps — before testing their result, so that a failed read stops the build, SHALL install the environment outside the path the source is mounted on and filter the build context so data, outputs, secrets, caches, and the repository metadata never enter an image, SHALL give the dev container the task runner it invokes and take its GPU flags from the GPU container decision, SHALL still record the host facts a container cannot pin, SHALL mount `data/`, `outputs/`, and the model-hub cache as volumes with the container-path mapping recorded and raise the container's shared memory for data-loader workers, and, when a preinstalled-stack image is chosen instead, SHALL make the image's framework authoritative by removing it from the project's dependencies and recording that rule; containers SHALL stay opt-in.

#### Scenario: Container requested
- **WHEN** the user asks for a training image
- **THEN** the builder writes the three-stage recipe, a `docker-digest` recipe that prints the digest to inject, and the AGENTS.md rule that the digest, not the tag, is recorded

#### Scenario: Source mounted over the image
- **WHEN** the Compose service mounts the working tree over the application directory
- **THEN** the image's environment still resolves because it lives outside that directory, and the run records the injected digest

#### Scenario: Sealed image built
- **WHEN** the sealed target is built from a working tree
- **THEN** the build refuses a dirty tree, the build context excludes data, outputs, secrets, caches, and the repository metadata, and the image carries the commit as a revision label and as the stamped value the manifest reads

#### Scenario: Tree state unreadable
- **WHEN** the sealed target is built where git cannot read the working tree, for example outside a repository
- **THEN** the recipe stops with git's error and builds no image, instead of treating the unread tree as clean

#### Scenario: Run inside a sealed image
- **WHEN** the training entry point runs inside a sealed image with no repository present
- **THEN** the manifest records the stamped commit with an unknown dirty flag and the injected digest, and lists no missing commit under degraded

#### Scenario: Compose run without a digest
- **WHEN** the Compose service starts the training entry point with `IMAGE_DIGEST` left at its empty default
- **THEN** the manifest records no image digest, lists `no_image_digest` under degraded, and the run is cited as degraded

#### Scenario: No container requested
- **WHEN** the user does not ask for a container
- **THEN** the builder scaffolds no Dockerfile, Compose file, or dev container and records the lock digest as the environment identity

#### Scenario: Preinstalled-stack image chosen
- **WHEN** the user chooses a vendor image with the framework preinstalled
- **THEN** the builder removes the framework from the project's dependencies, records that the image's stack is authoritative, and still records the image digest as the identity
