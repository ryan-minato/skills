# When Blocking Checks Exceed the Budget

Read when any blocking check would exceed the runtime budget agreed
with the user.

## Order of attack

1. **Cache dependencies.** Restoring a package cache is usually the
   single largest win; fetch the current `cache:` semantics and their
   runner-side limits from <https://docs.gitlab.com/ci/>.
2. **Cancel superseded runs.** A new push to the same merge request
   should cancel the previous pipeline; the docs name the current
   interruptibility and auto-cancel controls.
3. **Split and parallelize.** Separate independent checks into parallel
   jobs; shard a long test suite with the current parallelization
   keywords; let independent jobs start early via the documented
   dependency controls. Wall-clock time drops; total compute does not —
   say so when the user pays for runner minutes.
4. **Demote, don't stretch.** A check that stays over budget after the
   above stops being an MR gate: move it to a scheduled pipeline, keep
   it visible, and record in AGENTS.md who watches it.

## What never becomes an MR gate

Hardware-bound jobs, full end-to-end suites, and anything needing
secrets a fork MR cannot receive. Gate on the fast proxy; run the full
version on a schedule against the default branch.
