# Automated Dependency Updates

Read when the project has dependency manifests the user wants kept
current automatically.

Detection and updating are separate decisions on GitLab: the platform
scans, but it ships no first-party update bot.

## Detection — dependency scanning

1. Locate the current dependency-scanning capability through the
   llms.txt index (the application-security area,
   <https://docs.gitlab.com/user/application_security/>, is the topic's
   home as of this writing) and read the tier badge — scanning features
   concentrate in Ultimate.
2. When available, it wires into the pipeline as a documented include or
   component; agree with the user before touching `.gitlab-ci.yml`, and
   record who triages findings.

## Updating — a third-party bot

1. Renovate (<https://docs.renovatebot.com/>) is the established option
   for GitLab; fetch its current GitLab-platform setup from its own docs
   — hosted app versus self-hosted runner differs by instance, and the
   config schema evolves.
2. Copy [renovate-template.json](assets/renovate-template.json) to the config filename the
   fetched docs prescribe, and rework it: one rule set per detected
   ecosystem, cadence and grouping chosen with the user.
3. Agree the handling policy before the first MR arrives: who merges
   update MRs, what is grouped or auto-merged, and what happens when an
   update breaks CI. An update MR nobody triages is noise that trains
   the team to ignore the bot.
4. Record the policy in the AGENTS.md deposit.
