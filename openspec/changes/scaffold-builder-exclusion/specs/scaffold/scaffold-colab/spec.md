## Purpose
Governs what an agent that loaded the `scaffold-colab` builder observably does to keep the disposable builders out of every commit of the Colab-centric project it scaffolds.

## ADDED Requirements

### Requirement: Behavior: Disposable builders stay out of every commit
The builder SHALL, once the target is a git repository and before the build's first commit, add every project skill directory whose description opens with `Disposable builder skill (delete after the harness is built):` to the repository's local exclude file at `$(git rev-parse --git-path info/exclude)`, stage explicit paths, and read `git status` before each commit, so that no disposable builder enters a commit, whichever project shape the user chose.

#### Scenario: First commit of a notebook project
- **WHEN** the builder scaffolds a notebook project in a git repository whose project skills are `scaffold-colab` and two `meta` builders, and the build makes its first commit
- **THEN** before that commit the exclude file lists each of the three builder directories and `git status` shows none of them, every file of the commit is staged by explicit path, and no commit of the build contains a path under a builder directory

#### Scenario: Directory not yet a repository
- **WHEN** the builder scaffolds a notebook project in a directory that is not yet a git repository, whose project skills are `scaffold-colab` and two `meta` builders, and the build makes its first commit
- **THEN** the exclude entries are written once the directory is a repository and before the build's first commit, no exclude command fails for want of a repository, and no commit of the build contains a path under a builder directory

#### Scenario: Ephemeral runtime
- **WHEN** the user chooses the ephemeral-runtime shape — local scripts run on Colab VMs and there is no notebook — and the build makes its first commit
- **THEN** the exclude file lists every builder directory before that commit, and no commit of the build contains a path under a builder directory

#### Scenario: Durable skill beside the builders
- **WHEN** the repository's project skills also hold a skill whose description does not open with the disposable-builder marker
- **THEN** the exclude file lists every builder directory and does not list that skill's directory
