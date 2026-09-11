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
