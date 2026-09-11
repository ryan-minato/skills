# machine-learning/training-diagnosis Specification

## Purpose
Governs what an agent that loaded the `training-diagnosis` skill observably does when a training run misbehaves — locating the first anomaly, correlating signals top-down into a cause, confirming by replay, and reporting an evidence chain — and the contract of its bundled `scripts/scan_series.py`.

## Requirements

### Requirement: Trigger: description
The skill description SHALL cause the skill to load when the request describes a training run that failed, diverged, slowed, stalled, ran out of memory, or behaves unevenly across devices and asks why or how to fix it, and SHALL not cause it to load for deciding what to log or alert on before an incident, or for an environment or installation problem unrelated to a running training job.

#### Scenario: Utilization drop
- **WHEN** the user says "GPU utilization fell from 90% to 40% two hours into the run and tokens per second halved — nothing in the code changed"
- **THEN** the skill loads

#### Scenario: Numerical failure
- **WHEN** the user says "loss became NaN at step 48120; the gradient norm looked fine ten steps earlier"
- **THEN** the skill loads

#### Scenario: Designing signals (near-miss)
- **WHEN** the user says "what should we log so that next time a slow rank is caught early?"
- **THEN** the skill does not load

#### Scenario: Import failure (near-miss)
- **WHEN** the user says "importing torch crashes my Jupyter kernel on the laptop"
- **THEN** the skill does not load

### Requirement: Behavior: The first anomaly is located before causes are argued
The agent SHALL first establish the earliest time or step at which any signal left its baseline, aligning signals by step and wall clock and checking what changed at that point, SHALL then work top-down — correctness damaged? compare with the healthy baseline; device timeline gaps? split the wait into data, host, transfer, compilation, and synchronization; otherwise kernels and memory; then communication, storage, and hardware — SHALL name the causes ruled out at each rung, and SHALL not accept the last error message as the cause.

#### Scenario: Collective timeout reported
- **WHEN** the user reports that the run died with a collective-communication timeout on every rank
- **THEN** the agent looks for an earlier anomaly on a single rank (a crash, a data hang, a hardware event) before attributing the failure to the network, and reports the first anomalous rank and time

#### Scenario: Learning-rate change proposed as the fix
- **WHEN** the user asks to lower the learning rate and restart after a loss spike
- **THEN** the agent proposes locating the first anomalous signal and replaying the offending step first, and explains what a blind restart would hide

### Requirement: Behavior: Symptoms map to ranked causes through playbooks
For a low-utilization or slow-step symptom the agent SHALL check data waiting, host work, host-to-device transfer, compilation, and synchronization before kernel efficiency; for memory growth it SHALL separate a leak (active memory rising), fragmentation (reserved far above active), and non-framework allocation (device total far above what the framework reports); for uneven ranks it SHALL compute the slowest-to-median ratio and examine the slow rank's node, device, and link before the fabric; for a slow checkpoint or read with an idle network link it SHALL examine the storage control plane (request queues, concurrency limits, metadata) rather than bandwidth; and for each it SHALL name the signal that decides between the candidates.

#### Scenario: GPU idle with rising data wait
- **WHEN** step time rose while data-wait fraction rose and communication time stayed flat
- **THEN** the agent follows the data path to storage latency and does not propose kernel optimization

#### Scenario: Memory grows every epoch
- **WHEN** peak allocated memory grows each epoch while reserved memory tracks it
- **THEN** the agent diagnoses a retained-reference leak, asks for an allocation snapshot at two points, and does not blame fragmentation

#### Scenario: Checkpoint slow, link idle
- **WHEN** checkpoint writes take many times the baseline while the network interface reports low throughput
- **THEN** the agent examines the storage service's request concurrency and queueing and states that an idle link does not clear the storage path

### Requirement: Behavior: Numerical instability is confirmed by replay before it is fixed
For a spike, divergence, or non-finite value the agent SHALL reconstruct the timeline across loss, learning rate, precision scale, gradient norm and clipping, update magnitude, optimizer second moment, and activations, SHALL replay the offending step with the same batch and seed, in full precision, and on another device, splitting the batch into micro-batches and logging per layer, SHALL separate data, hyperparameter, optimizer-state, precision, and hardware causes by which replays reproduce the anomaly, and SHALL treat a hardware cause as plausible only when the anomaly follows the device.

#### Scenario: Reproduces everywhere
- **WHEN** the replay reproduces the spike in full precision on a second device with the same batch
- **THEN** the agent attributes it to the data or the model and optimizer state, isolates the responsible micro-batch or layer, and rules out hardware

#### Scenario: Reproduces on one device only
- **WHEN** the replay reproduces the anomaly only on the original device
- **THEN** the agent recommends quarantining that device for health testing and recomputing the affected steps elsewhere, and does not change the learning rate

### Requirement: Behavior: The result is an evidence chain the reader can verify
The agent SHALL report the diagnosis as an evidence chain — first anomaly, symptom, the signals consulted with their values, the causes ruled out and why, the root cause, the fix, the verification that confirmed it, and the signal that was missing — SHALL keep the chain to what the evidence supports, and SHALL run a sensitivity check on any log excerpt before it leaves the machine.

#### Scenario: Diagnosis delivered
- **WHEN** the agent completes a diagnosis
- **THEN** its report carries every element of the chain in order, with the missing signal named even when the cause was found

#### Scenario: Evidence insufficient
- **WHEN** the available signals cannot separate two remaining causes
- **THEN** the agent reports both with the deciding signal to capture next, and does not pick one

### Requirement: Handoff: training instrumentation
When a diagnosis ends with a signal that was missing, the agent SHALL offer the instrumentation role through the installing skill without printing an install command, and SHALL, when the user declines, name the missing signal and its frequency in the evidence chain and stop.

#### Scenario: Handoff offered
- **WHEN** the diagnosis found that no per-rank step time was recorded
- **THEN** the agent names the instrumentation role and routes its installation through the installing skill

#### Scenario: User declines
- **WHEN** the user declines the handoff
- **THEN** the agent records the missing signal and its frequency in the evidence chain and adds nothing else

### Requirement: Script: scan_series.py
The bundled script SHALL read a CSV file of a metric series, compute for one named column a robust z-score against a trailing window's median and median absolute deviation, report the first step at which the score stays beyond a threshold for a required count within a window (the first-anomaly candidate) and, when a grouping column is given, the ratio of the slowest to the median group at each step, SHALL print a JSON object to standard output, SHALL depend on the standard library only, and SHALL exit 0 on success, 1 when the input has no usable rows or the named column is absent, and 2 on bad arguments.

#### Scenario: Help
- **WHEN** the script runs with `--help`
- **THEN** it prints usage naming the input, column, window, threshold, and persistence options and the exit codes, and exits 0

#### Scenario: Representative run
- **WHEN** the script runs with `--input` naming a CSV whose column has a sustained spike after a stable prefix and `--column` naming that column
- **THEN** it prints JSON naming the first-anomaly step, the score at that step, and the window parameters, and exits 0

#### Scenario: Repeated run
- **WHEN** the identical command runs a second time
- **THEN** the output is identical and nothing on disk changes

#### Scenario: Bad arguments
- **WHEN** the script is invoked with an unknown option
- **THEN** it exits 2 and prints a diagnostic naming the option
