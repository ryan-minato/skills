# Continual Pre-training Recipes

How to inject new knowledge (e.g., a new language or domain) into an existing LLM without breaking it.

## 1. The Data Mixture
The biggest risk is **Catastrophic Forgetting**.
* **Target Data**: 90% - 95% (e.g., Technical Manuals, Legal Texts).
* **Replay Data**: 5% - 10% (General English/Code, e.g., FineWeb-Edu, Cosmopedia).
    * *Note*: Without replay data, the model's reasoning and basic grammar capabilities will degrade rapidly.

## 2. Tokenizer Expansion
If your new domain uses specialized vocabulary (e.g., Medical, Korean):
1.  Train a new tokenizer on the target corpus.
2.  Merge it with the base model tokenizer.
3.  Resize model embeddings: `model.resize_token_embeddings(len(new_tokenizer))`.
4.  **Important**: Initialize new tokens with the average embedding of existing tokens, not random noise, to speed up convergence.

## 3. Learning Rate
* Use a **lower LR** than the original pre-training max LR.
* Typically 5% to 10% of the original max LR.
* Use a cosine schedule with warmup.
