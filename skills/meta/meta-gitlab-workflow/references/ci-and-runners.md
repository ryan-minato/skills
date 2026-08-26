# CI Quality Gates and Runners

Read when any eligible runner exists, CI is requested, `.gitlab-ci.yml` exists,
or repository rules may depend on pipeline jobs.

Resolve `runners`, `CI/CD YAML syntax`, `workflow rules`, `merge request
pipelines`, `auto cancel redundant pipelines`, `interruptible`, and `CI lint`
through `llms.txt` and the self-managed instance help. Validate syntax against
the target instance before delivery.

## Runner gate

Inventory GitLab-hosted, instance, group, and project runners; their status,
tags, protection, executors, capacity, network, architecture, accelerator, and
secret exposure. A runner is eligible only when a job's tags and requirements
match and its trust boundary is acceptable.

If no eligible runner exists, recommend **no GitLab CI** and ask the user to
choose between that default and separately provisioning a runner. Do not create
a pipeline that will remain pending or install infrastructure as a side effect.

## Mirror local checks

Start from commands that exist and pass locally. CI mirrors them; it does not
invent a second quality system. Record each CI job and its exact local
reproduction command in the durable harness.

Default trigger contract when CI is approved:

- lint and format verification run for every pushed commit;
- tests run for merge-request pipelines and the default branch, not ordinary
  non-MR branch pipelines;
- use the target version's supported workflow/rules configuration to avoid
  duplicate branch and MR pipelines;
- enable redundant-pipeline cancellation and mark cancellable test jobs
  interruptible so one MR retains tests only for its latest commit;
- keep protected-branch or deployment jobs non-interruptible when cancellation
  would leave unsafe partial state.

Format verification checks the tree and fails on differences; it must not
rewrite a CI checkout and report success without proving the committed format.
Keep fork MR jobs tokenless by default. Secrets and protected runners must not
be exposed to untrusted code.

## Integration and validation

Preserve working job names because merge gates and dashboards may key on them.
Agree before converting classic branch pipelines to `workflow:rules`; a partial
conversion can duplicate or suppress pipelines. For monorepos, use change-aware
rules only after mapping every shared dependency and default-branch behavior.

Validate locally where possible, run the instance CI linter, then exercise an
MR and default-branch pipeline. Enable “pipelines must succeed” or job-dependent
repository rules only after the referenced jobs exist and pass.

Done when: every job has an eligible runner and local equivalent, triggers match
the agreed contract, stale MR tests cancel, secrets remain outside untrusted
jobs, and required merge gates reference real passing jobs.
