# .github/policy.tf — the repository settings this project needs, as stateless
# tf: no backend, one import block per resource, nothing ever deleted.
# Each `# {{…}}` comment marks a stand-in value to replace; delete the comment
# once the line holds the project's real value.
# Every declaration states what it sets and why; review rejects one that does
# not. An attribute left out may be reset to the provider's value by an apply,
# so declare everything the harness depends on. Plan and apply only through
# the task-runner recipes the knowledge file names; the plan guard runs there.
# This file holds enumerable repository attributes and only the rules the
# organization does not set uniformly; organization-owned rules (the security
# baseline, team workflow) belong to its configuration repository.

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 6.13"
    }
  }
}

provider "github" {
  owner = "example-org" # {{OWNER}}
}

import {
  to = github_repository.this
  id = "example-repo" # {{REPO}}
}

resource "github_repository" "this" {
  name = "example-repo" # {{REPO}}

  # {{WHY — e.g. "The README and package metadata point here; keep them in step."}}
  description = "Example description." # {{DESCRIPTION}}

  # {{WHY — e.g. "Public: the library is open source and takes outside pull requests."}}
  visibility = "private" # {{VISIBILITY — public, private, or internal}}

  # Intake forms and the triage workflow run on issues.
  has_issues = true

  # {{WHY — the merge method the branch contract settled.}} Declare all three;
  # set the methods the contract excludes to false, so an apply cannot reset
  # them to the provider's default of true.
  allow_squash_merge     = true
  allow_merge_commit     = false
  allow_rebase_merge     = false
  delete_branch_on_merge = true

  # Owned by the organization's configuration repository: secret scanning and
  # push protection can only be set on the repository resource here.
  # {{IGNORE_CHANGES — keep only for an attribute another owner holds or whose
  # reset is irreversible, each with its reason; delete this block otherwise.}}
  lifecycle {
    ignore_changes = [security_and_analysis]
  }
}

# {{WHY — e.g. "One approval on the default branch; the organization ruleset sets none."}}
# Only for a rule the organization does not set uniformly. First creation: a
# ruleset has no ID before it exists, and with no state the next run would
# create it again, so the guard blocks the create unless the plan and apply
# pass --allow-create. After that apply, read its ID with
# `gh api repos/{{OWNER}}/{{REPO}}/rulesets` and add, before anyone plans again:
#   import {
#     to = github_repository_ruleset.default_branch
#     id = "{{REPO}}:<ruleset id>"
#   }
resource "github_repository_ruleset" "default_branch" {
  name        = "default-branch" # {{RULESET_NAME}}
  repository  = github_repository.this.name
  target      = "branch"
  enforcement = "active"

  conditions {
    ref_name {
      include = ["~DEFAULT_BRANCH"]
      exclude = []
    }
  }

  rules {
    pull_request {
      required_approving_review_count = 1
    }
  }
}
