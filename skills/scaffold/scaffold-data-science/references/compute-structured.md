# Structured Data Compute

Use this branch when the project processes tables, records, or dataframe-like
data.

## Selection

1. Prefer Polars when the workload fits one machine, including workloads its
   lazy and streaming execution can handle.
2. Prefer Dask only when the workload genuinely needs distributed memory or
   distributed execution.
3. Record the scale evidence and chosen engine in `ARCHITECTURE.md` or
   `.agents/knowledge/PROJECT.md`.
4. Install only the chosen engine. Do not add Polars, Dask, pandas, and Spark
   as speculative alternatives.

Keep transformations as functions in `src/<package>/processing/`. Test
project-owned joins, filters, aggregations, schema invariants, boundary
conditions, and error paths where a plausible mistake would corrupt a
product. Do not repeat the engine's own tests.

Verify current lazy, streaming, and distributed execution behavior in the
selected engine's official first-party documentation.
