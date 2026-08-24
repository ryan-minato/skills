# When Required Checks Exceed the Budget

Read when any required check would exceed the runtime budget agreed
with the user.

## Order of attack

1. **Cache dependencies.** Restoring a package cache is usually the
   single largest win; fetch the current caching mechanism and its
   limits from <https://docs.github.com/en/actions>.
2. **Cancel superseded runs.** A new push to the same pull request
   should cancel the previous run; the docs name the current concurrency
   controls.
3. **Split and parallelize.** Separate independent checks into parallel
   jobs; shard a long test suite across a matrix. Wall-clock time drops;
   billed minutes do not — say so when the user pays for runners.
4. **Demote, don't stretch.** A check that stays over budget after the
   above stops being a PR gate: move it to a schedule or a manual
   trigger, keep it visible, and record in AGENTS.md who watches it.

## What never becomes a PR gate

Hardware-bound jobs, full end-to-end suites, and anything needing
secrets a fork PR cannot receive. Gate on the fast proxy; run the full
version on a schedule against the default branch.
