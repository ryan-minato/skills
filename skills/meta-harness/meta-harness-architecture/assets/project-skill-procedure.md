---
name: <verb-object, e.g. release-app>
description: >-
  Runs this project's <procedure> end to end: <the two or three steps
  that make it non-obvious>. Use when the user asks to <the requests
  that should trigger this>. Not for <the neighboring task this must
  ignore>.
---

# <Procedure>

<One sentence: what this achieves and the state it expects to start
from.>

## Before Starting

- <precondition to verify, and the command that verifies it>

## Steps

1. <step with the real command>
   - Done when: <observable result>
   - If it fails: <the known failure and its fix or rollback>
2. <next step>

Done when: <the end-to-end observable completion criterion>.

## Gotchas

- <the mistake that has actually happened here>

Update this skill in the same change that alters <the commands, paths, or
sequence above>.
