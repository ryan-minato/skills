# Group configuration repository — the group-level settings every project
# relies on. It keeps state in this project's GitLab-managed state by
# default: with state, removing a declaration deletes the object through an
# approved plan, and a `removed` block with `destroy = false` stops managing
# it instead. For stateless tf, delete the `backend` block; then nothing is
# ever deleted and every resource needs an import block. Each `# {{…}}` comment marks
# a stand-in value to replace; delete the comment once the line holds the real
# value. Every declaration states what it sets and why; review rejects one
# that does not. An attribute left out may be reset to the provider's value by
# an apply. Project-level settings stay in each project's .gitlab/policy.tf;
# nothing here declares a single project's own attributes, and one
# group-level setting replaces per-project ones. Group labels belong here; project
# labels stay in each project's label file.

terraform {
  required_version = ">= 1.6.0"

  # This project's GitLab-managed state. The address, lock, and credentials
  # come from TF_HTTP_* variables: the pipeline sets them with the job token,
  # and a maintainer's local recipe with their own glab login.
  backend "http" {}

  required_providers {
    gitlab = {
      source  = "gitlabhq/gitlab"
      version = "~> 19.3"
    }
  }
}

# GITLAB_TOKEN (a group access token in CI, the maintainer's glab login in the
# local recipes) and GITLAB_BASE_URL on a self-managed instance come from the
# environment; never write a token here.
provider "gitlab" {}

import {
  to = gitlab_group.this
  id = "example-group" # {{GROUP_FULL_PATH}}
}

resource "gitlab_group" "this" {
  name = "Example group" # {{GROUP_NAME}}
  path = "example-group" # {{GROUP_PATH}}

  # {{WHY — e.g. "Every member signs in with two factors; part of the security baseline."}}
  require_two_factor_authentication = true
}

import {
  to = gitlab_group_label.needs_triage
  id = "12345:needs-triage" # {{GROUP_ID}}:{{LABEL — the ID format in the resource's import documentation}}
}

# {{WHY — e.g. "Shared by every project's triage lifecycle."}}
resource "gitlab_group_label" "needs_triage" {
  group       = gitlab_group.this.id
  name        = "status::needs-triage" # {{LABEL_NAME}}
  color       = "#d73a4a"
  description = "Not yet triaged." # {{DESCRIPTION}}
}
