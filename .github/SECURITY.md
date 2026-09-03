# Security Policy

## Reporting a vulnerability

Use GitHub private vulnerability reporting:
<https://github.com/ryan-minato/skills/security/advisories/new>. Never
report a vulnerability through a public issue, discussion, or pull request.

Include the affected skill or script, the unsafe behavior an agent or tool
exhibits, a minimal reproduction, the expected safe behavior, and any known
downstream impact. Do not include real credentials or personal data.

- Acknowledgement: within seven days. Triage owner: the maintainer
  (`ryan-minato`).
- Supported versions: the current `main` branch. Skills are installed by
  copying from it and carry no version; reports should name the commit or
  install date.
- Disclosure: coordinated with the reporter once a fix is on `main`.

## What counts

Skills are instructions that agents execute. A skill or bundled script that
leads an agent to exfiltrate data, run destructive commands without the
approval its text promises, or print secrets is a security defect in this
repository, whichever project it was installed into.
