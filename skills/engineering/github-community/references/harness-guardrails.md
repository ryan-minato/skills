# GitHub harness guardrails

Read when authoring local files or a reviewed plan for dependency updates,
review routing, branch/tag rulesets, or GitHub security scanning.

1. Inspect repository visibility, plan, existing CODEOWNERS, dependency files,
   default branch, current rulesets, and available security features.
2. Separate local artifacts from remote settings:
   - Local: CODEOWNERS and Dependabot configuration are authored here.
   - Remote: rulesets and security settings are executed through
     `github-ops` after review.
3. Copy and rework
   [codeowners-template](assets/codeowners-template) only when every pattern
   has a real owner. A placeholder owner is worse than no routing.
4. Read [dependency-automation.md](dependency-automation.md) before adapting
   [dependabot-template.yml](assets/dependabot-template.yml). Group and
   schedule updates according to project maintenance capacity.
5. Read [rulesets.md](rulesets.md) to produce the desired reviewed ruleset
   contract: target branches/tags, bypass actors, required checks, reviews,
   update restrictions, and rollout mode.
6. Record the selected guardrails and human activation steps in the harness,
   adapting [the AGENTS.md section](assets/agents-md-guardrails-section.md).

Done when: local files validate, every owner and required check exists, remote
changes are represented by an exact reviewed plan, and unsupported or paid
features are identified rather than implied.
