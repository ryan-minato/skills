## MODIFIED Requirements

### Requirement: Behavior: Run identity is composed of four immutable parts plus a run id
When asked what a run must record or whether a run is reproducible, the agent SHALL name the source snapshot (the executed commit), the resolved configuration, the environment identity (a container image digest or the dependency lock digest, with the host facts a performance claim needs), and the identities of every input that affects the result, SHALL mint a run id distinct from the commit so one snapshot may produce many runs, SHALL refuse a branch name, a tag such as `latest`, a Dockerfile, or a working-directory path as an identity, and SHALL, through its manifest module, mark a run whose environment declares an image digest variable but leaves it empty as degraded rather than substituting the lock digest for the image identity.

#### Scenario: Run record contents
- **WHEN** the user asks what their run record should contain
- **THEN** the agent lists the four parts and the run id, states that the run id is not the commit, and names for each part the immutable form it takes

#### Scenario: Mutable references offered as identity
- **WHEN** the user's record names the dataset as `main` and the image as `latest`
- **THEN** the agent resolves each to an immutable identity (a dataset revision or checksum, an image digest), records the resolved value, and explains why the mutable name is not an identity

#### Scenario: Performance claim without host facts
- **WHEN** the user wants to record a run whose result is a throughput improvement and the record carries only the image digest
- **THEN** the agent adds the GPU model, driver, and runtime versions to the record and states that the digest alone does not explain a performance result

#### Scenario: Empty image digest declared
- **WHEN** the manifest module runs with `IMAGE_DIGEST` present in the environment but empty, as a Compose default leaves it
- **THEN** the manifest records no image digest, lists `no_image_digest` under degraded, prints the degraded reasons to standard error, and the record is cited as degraded
