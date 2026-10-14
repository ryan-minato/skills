# .gitlab/policy.tf — the project settings this project needs, as stateless
# tf: no backend (GitLab-managed state included), one import block per
# resource, nothing ever deleted. Each `# {{…}}` comment marks a stand-in value
# to replace; delete the comment once the line holds the project's real value.
# Every declaration states what it sets and why; review rejects one that does
# not. An attribute left out may be reset to the provider's value by an apply,
# so declare everything the harness depends on. Plan and apply only through
# the task-runner recipes the knowledge file names; the plan guard runs there.
# Group-owned rules (the security baseline, team workflow) belong to the
# group's configuration repository, never to this file.

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    gitlab = {
      source  = "gitlabhq/gitlab"
      version = "~> 19.3"
    }
  }
}

# GITLAB_TOKEN (and GITLAB_BASE_URL on a self-managed instance) come from the
# local recipes; never write a token here.
provider "gitlab" {}

import {
  to = gitlab_project.this
  id = "example-group/example-project" # {{PROJECT_FULL_PATH}}
}

resource "gitlab_project" "this" {
  name = "example-project" # {{PROJECT_NAME}}

  # {{WHY — e.g. "The README and package metadata point here; keep them in step."}}
  description = "Example description." # {{DESCRIPTION}}

  # {{WHY — e.g. "Private: the code is internal to the company."}}
  visibility_level = "private" # {{VISIBILITY — public, internal, or private}}

  # Work items and the triage lifecycle run on issues; changes land through merge requests.
  issues_access_level         = "enabled"
  merge_requests_access_level = "enabled"

  # {{WHY — the merge method the branch contract settled.}}
  merge_method                                     = "ff" # {{MERGE_METHOD — merge, rebase_merge, or ff}}
  only_allow_merge_if_pipeline_succeeds            = true
  only_allow_merge_if_all_discussions_are_resolved = true
  remove_source_branch_after_merge                 = true
}

import {
  to = gitlab_branch_protection.default
  id = "12345:main" # {{PROJECT_ID}}:{{DEFAULT_BRANCH}}
}

# {{WHY — e.g. "Only maintainers merge to the default branch; nobody pushes or force-pushes to it."}}
resource "gitlab_branch_protection" "default" {
  project            = gitlab_project.this.id
  branch             = "main" # {{DEFAULT_BRANCH}}
  push_access_level  = "no one"
  merge_access_level = "maintainer"
  allow_force_push   = false
}

# {{WHY — e.g. "One approval from the backend team; the group sets no approval rule."}}
# First creation: an approval rule has no ID before it exists, so it starts
# without an import block and the plan guard lists the create. After the apply,
# read its ID with `glab api projects/<id>/approval_rules` and add:
#   import {
#     to = gitlab_project_approval_rule.extra
#     id = "<project id>:<rule id>"
#   }
resource "gitlab_project_approval_rule" "extra" {
  project            = gitlab_project.this.id
  name               = "example-rule" # {{RULE_NAME}}
  approvals_required = 1
}
