# Experimental Model Reimplementation

Use this branch only when an experimental external repository is too unstable
to remain a runtime dependency and the analysis still needs its inference
behavior.

1. Record the upstream URL, exact commit, license, model weights, reference
   environment, inputs, outputs, and numeric tolerances before editing.
2. Write a failing characterization test against the upstream implementation:
   weight loading, required tensor shapes, one or more representative outputs,
   and explicit absolute or relative tolerance.
3. Reimplement the smallest inference surface inside `src/`. Remove training,
   optimizer, dataset preparation, experiment tracking, and unused task code.
4. Make the test pass with the original weights. Explain any accepted numeric
   difference by device, precision, nondeterminism, or a documented correction;
   never silently widen tolerance.
5. Keep small deterministic tests in the fast suite. Mark tests that require
   large weights, a GPU, network access, or long execution as `slow` and run
   them manually.
6. Record the reimplementation boundary and upstream revision in
   `ARCHITECTURE.md` and `.agents/knowledge/DATA.md`; do not create a separate
   documentation-URL index.

Done when: the project loads the intended weights without importing the
external repository, representative outputs match within the recorded
tolerance, and no training-only surface remains.
