# Model Inference

Use this branch when an analysis workflow loads a model.

Prefer a mature maintained library and a versioned Hugging Face Hub model.
Record library version, model repository, immutable revision, preprocessing,
generation or inference parameters, device and precision in configuration
and provenance.

Choose Transformers for ordinary supported model loading and inference.
Choose vLLM when its serving or high-throughput inference model matches the
actual workload. Do not install both without two named consumers.

When the provider supplies only local weights, create `model/`, merge
[local-model-gitignore](assets/model/local-model-gitignore) into
`.gitignore`, and keep one-file weights directly under `model/` or a
multi-file model under `model/<model-name>/`. Never commit weights.

Analysis code must not include training, finetuning, optimizers, training
loaders, or checkpoint management. Produce data artifacts for a separate
training project instead.

Verify current model-loading, revision, device, and precision behavior in the
selected library's official first-party documentation.
