# Automated Dependency Updates

Read when the project has dependency manifests the user wants kept
current automatically.

## Procedure

1. Detect the ecosystems from the manifests actually present (package
   manifests, lockfiles, container files, workflow files — the workflow
   files' actions are themselves an updatable ecosystem).
2. Fetch the current Dependabot configuration schema and the list of
   supported ecosystems from
   <https://docs.github.com/en/code-security> — the schema and coverage
   evolve; never write the file from memory. Note the distinction the
   docs draw between scheduled version updates (needs the committed
   config) and security updates (a repository setting).
3. Copy [dependabot-template.yml](assets/dependabot-template.yml) to
   `.github/dependabot.yml` and rework it: one update block per detected
   ecosystem, schedule and grouping chosen with the user against the
   fetched schema.
4. Agree the handling policy before the first PR arrives: who merges
   update PRs, whether minor and patch updates are grouped or
   auto-merged, and what happens when an update breaks CI. An update PR
   nobody triages is noise that trains the team to ignore the bot.
5. Record the policy in the AGENTS.md deposit.
