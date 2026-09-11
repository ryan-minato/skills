## Purpose
Governs what an agent that loaded the `training-instrumentation` skill observably does when it decides what a training run emits, at which frequency and severity, how the signals correlate across layers, and how alerts and dashboards are designed.

## ADDED Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request concerns what a training loop or job should log or measure, how often, how to correlate metrics, traces, and profiles, whether a training metric's value is normal, or designing alerts, severities, or dashboards for training, and SHALL not cause it to load for diagnosing a run that has already failed or slowed, or for logging in a service that trains nothing.

#### Scenario: What to log
- **WHEN** the user says "we only print loss every 100 steps — what should a serious training loop actually record?"
- **THEN** the skill loads

#### Scenario: Alerts for training jobs
- **WHEN** the user says "set up alerting for our multi-node training jobs so we get paged before a run wastes a night"
- **THEN** the skill loads

#### Scenario: Incident already happening (near-miss)
- **WHEN** the user says "loss went NaN at step 48120 last night — what happened?"
- **THEN** the skill does not load

#### Scenario: Service logging (near-miss)
- **WHEN** the user says "add structured request logging to the FastAPI inference service"
- **THEN** the skill does not load

### Requirement: Behavior: Three layers with shared correlation keys and separated instruments
The agent SHALL design observation across an outcome layer, an optimization layer (training dynamics), and a systems layer, SHALL give every emitted record the keys that let signals join — run id, global step, rank and world size, node or pod, device, phase, and a trace context where traces exist — SHALL keep metrics (continuous, cheap), stage-level traces (data, forward, loss, backward, optimizer), and profiling (scheduled windows or incident-triggered) as separate instruments, and SHALL refuse an always-on full profiler or trace of every operator.

#### Scenario: Loop without correlation keys
- **WHEN** the user's loop logs loss and step time with no run id or rank
- **THEN** the agent adds the correlation keys to every record and states which join each key enables

#### Scenario: Full profiler always on
- **WHEN** the user wants the operator profiler enabled for the whole run
- **THEN** the agent declines, configures a scheduled window with skip, wait, warm-up, and active steps, and names an incident trigger for deeper capture

### Requirement: Behavior: A minimal signal set by priority with bounded labels and no raw data
The agent SHALL propose the signal set in priority order — failures explainable first, regressions detectable second, regressions locatable third, automated correlation last — with a sampling frequency per signal proportional to its cost, SHALL keep unbounded values (step, sample id, file path, tensor name, stack trace) out of metric labels, and SHALL keep raw samples, prompts, credentials, presigned URLs, and tensor contents out of logs, directing them to a governed store with identifiers in the logs.

#### Scenario: Small single-GPU project
- **WHEN** the user runs a single-GPU experiment and asks for the smallest useful set
- **THEN** the agent proposes run identity, loss and learning rate, step time and throughput, gradient norm, non-finite checks, device memory, and data-wait fraction with their frequencies, and defers cluster telemetry

#### Scenario: Step as a metric label
- **WHEN** the user's metric exporter labels every series with the global step and the sample id
- **THEN** the agent moves step to the metric's value axis or a field, drops the sample id, and explains the series-cardinality cost

#### Scenario: Prompts in logs
- **WHEN** the user wants each batch's prompts printed to the training log for debugging
- **THEN** the agent logs sample identifiers or hashes and routes any raw sample capture to an access-controlled artifact store

### Requirement: Behavior: Alerts are symptom-first, baseline-relative, persistent, and graded
The agent SHALL alert on training symptoms — no progress, numerical failure, unrecoverable memory exhaustion, collective timeout, missing checkpoint, unexpected exit, throughput regression, data stall, straggler — rather than on every underlying cause, SHALL express thresholds relative to a healthy baseline of the same run or model (a ratio to a baseline percentile, or a robust z-score against a rolling median and median absolute deviation) with a persistence rule before paging, SHALL combine signals for instability (a loss spike together with a gradient, update, or non-finite anomaly), SHALL grade with five severities where the top level means the run is failed, about to fail, or no longer trustworthy, and SHALL label any absolute starting threshold as an initial value to recalibrate.

#### Scenario: Absolute gradient-norm alert requested
- **WHEN** the user asks to page when the gradient norm exceeds 1.0
- **THEN** the agent proposes a rolling-median-relative rule with a persistence count, explains that the normal range differs by model and optimizer, and keeps any absolute number as an initial value only

#### Scenario: Utilization alert requested
- **WHEN** the user asks to page whenever GPU utilization drops below 50%
- **THEN** the agent replaces it with throughput and step-latency regression alerts relative to the baseline and keeps utilization as a dashboard signal for diagnosis

### Requirement: Behavior: Model-health metrics extend loss and are matched to the model family
Beyond loss and the global gradient norm, the agent SHALL propose the update-to-weight ratio, a per-layer gradient view, optimizer second-moment health, per-example loss quantiles, activation statistics, and precision-scaler health with a sampling frequency for each, SHALL add model-family signals — residual and attention-logit statistics for transformers, policy divergence and entropy for reinforcement learning, generator-discriminator balance for adversarial training, per-timestep loss for diffusion, expert load and all-to-all volume for mixture-of-experts, per-stage bubbles for pipeline parallelism — and SHALL state that no single metric is a universal health score.

#### Scenario: Language-model pretraining
- **WHEN** the user instruments a transformer pretraining run
- **THEN** the agent adds residual-stream and attention-logit statistics, precision-scaler skips, and the update-to-weight ratio with their frequencies, and names the second-moment underestimation that precedes spikes as a reason for the optimizer metric

#### Scenario: Policy optimization
- **WHEN** the user instruments a proximal policy optimization run
- **THEN** the agent adds policy divergence from the previous policy, entropy, the clipped fraction, and value loss, and does not propose a supervised validation loss as the health signal

### Requirement: Handoff: training diagnosis
When the request turns from designing signals to explaining a run that is already misbehaving, the agent SHALL offer the diagnosis role through the installing skill without printing an install command, and SHALL, when the user declines, name the signals that would have separated the causes and stop at that.

#### Scenario: Handoff offered
- **WHEN** the user, while designing alerts, asks why last night's run slowed down
- **THEN** the agent names the diagnosis role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent lists the signals that would separate the candidate causes and does not attempt the diagnosis itself
