# PEFT & LoRA Configuration

## Recommended Settings for 7B-14B Models
Based on community benchmarks and SmolLM findings.

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,                  # Rank: 16-64 is usually sufficient
    lora_alpha=32,         # Alpha: Usually 2 * rank
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear", # Crucial: target all linear layers, not just q_proj, v_proj
    use_dora=True          # DoRA usually outperforms standard LoRA slightly
)
```

## QLoRA (Quantized LoRA)

* Load base model in 4-bit (`load_in_4bit=True` in `BitsAndBytesConfig`).
* Train the adapter in BF16/FP16.
* **VRAM Impact**: Allows fine-tuning a 70B model on two 24GB GPUs (e.g., 3090/4090).
