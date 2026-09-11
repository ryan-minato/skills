# machine-learning

[中文](README.zh.md)

Durable, per-project skills for the **daily work of a project that trains
or evaluates models**: recording what a run actually used, organizing a
series of hypotheses into one research task, shaping experiment code and
its configuration surface, deciding what a training run emits and when to
alert, and diagnosing a run that misbehaves. Install the ones the project
needs; each works alone and hands off to its siblings by role.

Initializing such a project — layout, environment, commands, checks, agent
guidance — belongs to the disposable `scaffold` catalog (`scaffold-ml`);
GPU container environments to the disposable `meta` catalog.

```bash
npx skills add ryan-minato/skills --skill <skill-name>
```

## Skills

| Skill | Description |
|---|---|
| [experiment-provenance](experiment-provenance/) | Record and judge a run's identity — the executed source snapshot, the resolved configuration, the environment identity (image or lock digest plus host facts), the input identities, and a run id distinct from the commit — keep run history immutable, and wire a tracker (existing → platform → Trackio) that holds the manifest without secrets. |
| [research-workflow](research-workflow/) | Run a research task end to end: a research spec with an objective and an evaluation, one task per pull or merge request carrying many hypotheses, a hypothesis loop with snapshot commits on an isolated branch, evidence that matches the claim, automatic search when the space and the compute allow it, and a closing verdict — negative results included. |
| [experiment-code-conventions](experiment-code-conventions/) | Shape experiment code: abstract semantic coupling and tolerate accidental similarity, prefer mature first-party dependencies and vendor unstable research code with its origin, keep the training loop explicit, keep the configuration surface to values a run may choose, test behavior contracts with a light CPU default suite and GPU-only tests that fail without hardware, keep hooks free of tests, lint near defaults with no global type gate over tensor code, and keep hot-path performance while recovering understandability elsewhere. |
