## Purpose
Governs what an agent that loaded the `scaffold-data-science` builder observably does to keep the disposable builders out of every commit of the data-science project it scaffolds.

## ADDED Requirements

### Requirement: Behavior: Disposable builders stay out of every commit
The builder SHALL, once the target is a git repository and before the build's first commit, add every project skill directory whose description opens with `Disposable builder skill (delete after the harness is built):` to the repository's local exclude file at `$(git rev-parse --git-path info/exclude)`, stage explicit paths, and read `git status` before each commit, so that no disposable builder enters a commit, whether the build initializes a new package or hardens an existing project.

#### Scenario: First commit of the build
- **WHEN** the builder hardens an existing project in a git repository whose project skills are `scaffold-data-science` and two `meta` builders, and the build makes its first commit
- **THEN** before that commit the exclude file lists each of the three builder directories and `git status` shows none of them, every file of the commit is staged by explicit path, and no commit of the build contains a path under a builder directory

#### Scenario: Directory not yet a repository
- **WHEN** the target directory is not a git repository and holds no package, and the builder initializes the package with `uv init --package`
- **THEN** the exclude entries are written once the directory is a repository and before the commit carrying `uv.lock`, no exclude command fails for want of a repository, and the commit carrying `uv.lock` contains no path under a builder directory

#### Scenario: Durable skill beside the builders
- **WHEN** the repository's project skills also hold a skill whose description does not open with the disposable-builder marker
- **THEN** the exclude file lists every builder directory and does not list that skill's directory
