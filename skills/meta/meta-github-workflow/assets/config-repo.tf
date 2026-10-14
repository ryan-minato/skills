# Organization configuration repository — the organization-level settings
# every repository relies on, as stateless tf: no backend, one import block per
# resource, nothing ever deleted. Each `# {{…}}` comment marks a stand-in
# value to replace; delete the comment once the line holds the real value. Every declaration states what it sets and
# why; review rejects one that does not. An attribute left out may be reset to
# the provider's value by an apply. Repository-level settings stay in each
# repository's .github/policy.tf; nothing here declares a single repository.
# Labels stay in each repository's label file and are never declared here.

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 6.13"
    }
  }
}

# Credentials come from the environment only, as GITHUB_TOKEN: in CI the
# workflow mints a GitHub App installation token per run; the local recipes
# pass `gh auth token`. Never write them here.
provider "github" {
  owner = "example-org" # {{ORG}}
}

# {{WHY — e.g. "Every default branch takes changes only through a pull request; the security baseline."}}
# First creation: an organization ruleset has no ID before it exists, so it
# starts without an import block and the plan guard lists the create. After the
# apply, read its ID with `gh api orgs/{{ORG}}/rulesets` and add:
#   import {
#     to = github_organization_ruleset.default_branches
#     id = "<ruleset id>"
#   }
resource "github_organization_ruleset" "default_branches" {
  name        = "default-branches" # {{RULESET_NAME}}
  target      = "branch"
  enforcement = "active"

  conditions {
    ref_name {
      include = ["~DEFAULT_BRANCH"]
      exclude = []
    }
    repository_name {
      include = ["~ALL"]
      exclude = []
    }
  }

  rules {
    deletion         = true
    non_fast_forward = true
    pull_request {
      required_approving_review_count = 0
    }
  }
}
