# Run <run_id>

- Status: <running | complete | failed> · started <start UTC> · ended <end UTC or ->
- Decision: <promoted | baseline | rejected | superseded by run_id>
- Source: <repository> @ <commit> · dirty: <yes/no> · kept reachable by <tag or ref>
- Configuration: <resolved dump path> · sha256 <hash>
- Environment: image <digest> | lock sha256 <hash> · python <version>
- Host: <GPU model × count> · driver <version> · runtime <versions> · node <name>
- Inputs:
  - <name> (<dataset | model | checkpoint | tokenizer | benchmark>): <identity> · preprocessing <version>
- Randomness: seed <value> · deterministic <flags>
- Parent run: <run_id or none>
- Evaluation: set <identity> · evaluation code <commit>
- Metrics (<definition source>):
  - <name>: <value> (threshold agreed before the run: <value or none>)
- Artifacts:
  - <kind>: <governed location> · <immutable id>
- Tracker run: <url or id>
