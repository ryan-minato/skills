# Multimedia Compute

Use this branch when the project processes images, audio, video, documents,
or other multimedia collections.

Prefer Ray Data when parallel decoding, transformation, or batch inference
needs to scale. For a small project that remains entirely in Hugging Face
Hub, `datasets` is the lighter alternative. Choose one against the real
volume and operations; do not install both as backups.

Record:

- observation grain and media identity;
- decoding and sampling rules;
- concurrency and resource assumptions;
- corrupt-item policy and counts;
- deterministic seeds where sampling occurs.

Keep decoding and transformation logic in `src/`, not only in notebooks.
Log counts and failures without logging raw media or sensitive metadata.

Verify current decoding, concurrency, and resource behavior in the selected
engine's official first-party documentation.
