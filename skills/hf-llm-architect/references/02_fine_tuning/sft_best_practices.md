# Supervised Fine-Tuning (SFT) Checklist

## 1. Sequence Packing (`packing=True`)
* **Concept**: Concatenates samples to fill the `max_seq_length`.
* **Impact**: 2x-3x training speedup.
* **Code**:
    ```python
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        packing=True,
        max_seq_length=4096,
        dataset_text_field="text" # Dataset must be pre-formatted strings
    )
    ```

## 2. Chat Templates
Never manually concatenate strings. Use the tokenizer's template.
```python
def apply_chat_template(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    }
```

* **Verification**: Run `scripts/stream_data_preview.py` to ensure the template renders special tokens (e.g., `<|im_start|>`) correctly.

## 3. Loss Masking

Ensure you are using `DataCollatorForCompletionOnlyLM`. The model should **not** learn to predict the user's instructions, only the assistant's responses.

```python
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)
```
