# Organization configuration repository — the organization-level settings
# every repository relies on. It keeps state in the backend the user chose:
# with state, removing a declaration deletes the object through an approved
# plan, and a `removed` block with `destroy = false` stops managing it
# instead. For stateless tf, delete the `cloud` block; then nothing is ever
# deleted and every resource needs an import block. Each `# {{…}}` comment marks a stand-in
# value to replace; delete the comment once the line holds the real value. Every declaration states what it sets and
# why; review rejects one that does not. An attribute left out may be reset to
# the provider's value by an apply. Repository-level settings stay in each
# repository's .github/policy.tf; nothing here declares a single repository's
# own attributes, and one organization-level rule replaces per-repository ones.
# Labels stay in each repository's label file and are never declared here.

terraform {
  required_version = ">= 1.6.0"

  # Default with Terraform: HCP Terraform, the workspace set to Local
  # execution so it stores and locks state only. With OpenTofu, replace this
  # block with the backend the user named. CI reads the API token from
  # TF_TOKEN_app_terraform_io; locally, `terraform login`.
  cloud {
    organization = "example-org" # {{HCP_TERRAFORM_ORGANIZATION}}
    workspaces {
      name = "github-org-settings" # {{HCP_TERRAFORM_WORKSPACE}}
    }
  }

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
# With state, the first apply creates it and the state remembers it. Without
# state, plan the first creation with --allow-create, then read its ID with
# `gh api orgs/{{ORG}}/rulesets` and add, before anyone plans again:
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
    # One organization-level rule for every repository that needs it; list
    # names here instead of declaring a ruleset per repository.
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
