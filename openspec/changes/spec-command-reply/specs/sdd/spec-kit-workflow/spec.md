## ADDED Requirements

### Requirement: Behavior: The produced comment-command workflow answers every command it admits
On GitHub, the produced comment-command workflow SHALL admit a collaborator's pull request comment only when the comment begins with `/spec` followed by the end of the comment, a space, a tab, or a line break, and answer every comment it admits, once the reads of its pull request succeed (the snapshot, the head the snapshot records, and the pull request's state), with one reply — the output of `show` or `status` for a known command, read with a carriage return at the end of the command line ignored and spaces and tabs both separating words, and the command list for any other `/spec` line or one it cannot read — whose header echoes the command as it was read (`/spec help` for the command list), never the raw line.

#### Scenario: Kit command in a comment
- **WHEN** a collaborator's comment on a pull request begins with `/speckit.plan`, `/specs`, or `/specification`
- **THEN** the produced workflow's job does not run

#### Scenario: Command line ends in a carriage return
- **WHEN** a collaborator comments `/spec status`, a carriage return and a line feed, then `thanks`, and the management script's `status` exits 0
- **THEN** the reply is the status table under the header `/spec status`, the header holds no carriage return, and the run passes

#### Scenario: Tab after /spec
- **WHEN** a collaborator comments `/spec`, a tab, then `status`
- **THEN** the job runs and the reply is the status table under the header `/spec status`

#### Scenario: Bare /spec followed by more text
- **WHEN** a collaborator comments `/spec`, a carriage return and a line feed, then `thanks`
- **THEN** the job runs, the reply is the command list under the header `/spec help`, and the run passes

#### Scenario: Unknown command
- **WHEN** a collaborator comments `/spec foo`
- **THEN** the reply is the command list under the header `/spec help`, and the run passes

#### Scenario: Command in another case
- **WHEN** a collaborator comments `/SPEC status`, which the job admits because GitHub compares strings without regard to case
- **THEN** the reply is the command list under the header `/spec help`, and the run passes

### Requirement: Behavior: The produced comment-command reply names and links the head its content was read from
On GitHub, the produced comment-command workflow SHALL name in the reply's header, and use as the commit of every link the reply carries, the head commit that the snapshot the reply's content was read from records, never a head read by a separate call, and SHALL fail in the step that reads that head, before any reply, when the snapshot records none.

#### Scenario: Head moves between the two reads
- **WHEN** the head the pull request's state read reports differs from the head the snapshot records, as when a push lands between the two reads, and a collaborator comments `/spec show` on a request whose feature document exceeds the reply's size limit
- **THEN** the reply's header names the snapshot's head, every `[view]` link and the truncation link point at `/blob/<snapshot head>/`, and the other head appears nowhere in the reply

#### Scenario: Status names the snapshot's head
- **WHEN** the two reads disagree as above and a collaborator comments `/spec status`
- **THEN** the reply's header names the snapshot's head

#### Scenario: Closed request
- **WHEN** a collaborator comments `/spec status` on a closed pull request
- **THEN** the reply's header carries the snapshot's head and the note that the pull request is `CLOSED`

#### Scenario: Snapshot without a head
- **WHEN** the snapshot file the workflow wrote carries no head commit, or a null one
- **THEN** the step that reads the head fails with a message naming the snapshot file, and no reply is posted
