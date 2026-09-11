# Vendoring Research Code

Read when deciding whether to depend on or copy code from a paper's or
another project's repository, and when copying it.

## When to vendor rather than depend

Depend on a library when an organization maintains it, it has a test
suite, and the project needs it as a whole. Vendor — copy the needed part
into the project and maintain it locally — when the source is an
individual's or a lab's repository, is unmaintained, pins an environment
the project cannot share, or is needed only for one component. A vendored
component is the project's responsibility from that moment.

## Two modes

| Mode | Goal | Rule |
|---|---|---|
| Replication | reproduce a baseline faithfully | minimize semantic difference from the original: copy the component, adapt only the interface, change nothing that affects the numbers; keep a characterization test against the original outputs |
| Innovation | build on the idea | keep only the research semantics; the original's engineering structure is not inherited, and the implementation may be better than the source |

State the mode in the vendored module's header and in the project's
conventions; a replication that quietly improved the implementation is
not a replication.

## Procedure

1. Record the upstream: URL, exact commit, license, the files taken, and
   the date. Keep the license text with the code and respect its
   attribution terms.
2. Copy the smallest surface the project needs; delete training loops,
   experiment tracking, dataset preparation, and task code the project
   does not use.
3. Adapt the interface to the project's conventions (configuration
   surface, logging seam, tensor conventions) without touching the
   semantics in replication mode.
4. Protect the contract with a behavior test: shapes, a representative
   output against recorded reference values, and the tolerance chosen
   for device and precision differences. Never widen a tolerance to make
   a test pass; explain any accepted difference.
5. Record the boundary in the project's architecture notes: what is
   vendored, from where, in which mode, and what test guards it.

## Gotchas

- A branch name or "master" is not an upstream identity; record the
  commit.
- Copying the upstream's configuration system along with the model
  imports its abstractions; take the mechanism, leave the machinery.
- Upstream numerical quirks (a nonstandard initialization, an epsilon)
  are part of the semantics in replication mode; document them instead of
  fixing them.
