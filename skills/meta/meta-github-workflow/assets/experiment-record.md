# Run {{RUN_ID}}

<!-- Committed per-run record: the durable identity of an experiment.
The tracker holds the interactive view; this file is what a future agent
can trust from the checkout alone. Rework all slots; add project fields;
never include secrets, dataset contents, or restricted paths. -->

- Status: {{RUNNING_COMPLETE_FAILED}} · started {{START_UTC}} · ended
  {{END_UTC_OR_DASH}}
- Code: commit {{COMMIT_SHA}} ({{BRANCH_OR_PR}})
- Environment: {{LOCKFILE_OR_IMAGE_DIGEST}}
- Data: {{DATASET_IDS_AND_IMMUTABLE_VERSIONS_OR_CHECKSUMS}} ·
  preprocessing {{PREPROCESSING_VERSION}}
- Config: {{CONFIG_PATH_AND_HASH}} · seeds {{SEEDS}}
- Hardware/runtime: {{HARDWARE_AND_RUNTIME}}
- Metrics ({{EVALUATION_SET_IDENTITY}}): {{METRICS_WITH_DEFINITIONS}}
- Artifacts: {{GOVERNED_LOCATIONS_AND_IMMUTABLE_IDS — never Actions
  artifacts}}
- Tracker run: {{TRACKER_RUN_URL_OR_ID}}
- Promotion: {{NOT_PROMOTED_OR_PR_AND_RELEASE_LINKS — threshold
  {{AGREED_METRIC_THRESHOLD}} agreed before the run}}
