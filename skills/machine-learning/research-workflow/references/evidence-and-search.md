# Evidence and Automatic Search

Read when deciding whether results support a claim, when comparing
against a baseline, or when automatic search is on the table.

## What a comparison needs

- **The same evaluation**: the same evaluation set identity, the same
  evaluation code (by commit), the same metric definitions, and the same
  preprocessing. A comparison across two evaluation versions is not a
  comparison.
- **A recorded baseline run**: the run id the candidate is compared with,
  reproduced under the current evaluation when the stored number predates
  it.
- **Variance**: repeated seeds, or a confidence bound from the evaluation
  set's size, before a margin is called a difference. A single-seed win by
  a margin inside the seed-to-seed spread is inconclusive.
- **The stated setup**: for a performance claim, the hardware, runtime,
  batch shape, and precision; a speed-up measured on other hardware is a
  different claim.

## Downgrading a claim

When the evidence falls short, the verdict says what was shown, not what
was hoped: "improves the metric by X on set S with one seed; variance not
measured" is a result. Record what would make the claim conclusive and
whether it is worth the runs.

## Negative and inconclusive results

A hypothesis that failed its evaluation, or could not be separated from
the baseline, is recorded with the same care as a win: the runs, the
evidence, the reason it is judged negative, and what it rules out for
later tasks. This is what prevents the next person from repeating it.

## Automatic search

Recommend it when three things hold: the objective is evaluated
automatically, the search space is explicit, and one run is cheap enough
that a systematic search costs less than the human time it replaces.

- **Vocabulary**: hyperparameter optimization when the space contains
  hyperparameters only (learning rate, weight decay, dropout, betas,
  warm-up). When it also varies the dataset, the algorithm, the
  architecture, or the training strategy, the searched values are search
  variables, and the search is an automatic experiment.
- **Division of labor**: humans and the agent define or narrow the space,
  the objective, and the evaluation, and re-read the research question
  from what the search found; the search itself is not the research.
- **Economics**: at large-model scale — days per run, many accelerators —
  a sweep is rarely affordable. Plan a handful of hypotheses chosen by
  reasoning, state that a search was considered and why it was not run,
  and record it in the spec.
- **Record**: every search run is a run like any other, with its own
  identity, and the search's objective and space are part of the spec.
