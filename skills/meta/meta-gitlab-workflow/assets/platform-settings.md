# GitLab Platform Settings

Last verified: <timestamp and evidence>

| Concern | Intended setting | Enforcement | Owner | Verify | Update trigger | Described by |
|---|---|---|---|---|---|---|
| Default branch and merge method | <value> | <platform/convention> | <owner> | <readback> | <trigger> | <.gitlab/policy.tf, the group's configuration repository, or none> |
| Protected branches and tags | <value> | <platform/convention> | <owner> | <readback> | <trigger> | <.gitlab/policy.tf, the group's configuration repository, or none> |
| Approvals and CODEOWNERS | <value> | <platform/convention> | <owner> | <readback> | <trigger> | <.gitlab/policy.tf, the group's configuration repository, or none> |
| Required pipeline jobs | <job names or none> | <platform/convention> | <owner> | <readback> | <trigger> | <.gitlab/policy.tf, the group's configuration repository, or none> |
| Runners | <eligible runners or none> | <platform> | <owner> | <readback> | <trigger> | <.gitlab/policy.tf, the group's configuration repository, or none> |
| Environments and deployment approvals | <value or none> | <platform/convention> | <owner> | <readback> | <trigger> | <.gitlab/policy.tf, the group's configuration repository, or none> |
| Scanners and security policies | <value or none> | <platform/convention> | <owner> | <readback> | <trigger> | <.gitlab/policy.tf, the group's configuration repository, or none> |

"Described by" names the file that declares the row. A row a tf file
describes is verified by a plan through the local recipes; a row no
provider covers keeps its readback as the only check.

## Harness options

| Option | Choice | Default | Recorded on |
|---|---|---|---|
| Platform settings as tf | <yes or no> | yes when `tofu` or `terraform` is installed | <date> |
| Label sync | <CI or manual> | CI with a project access token; manual on GitLab.com Free | <date> |
| Label pruning | <off or on> | off — deleting a label strips it from every work item and merge request | <date> |
| Configuration repository sync (configuration repository only) | <CI plans and applies, CI plans and a maintainer applies, or local> | CI plans and applies; local on GitLab.com Free | <date> |

Never record token or variable values. Record names, scopes, roles, expiry, owners, rotation,
and verification only.
