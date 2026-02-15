---
name: hf-llm-architect
description: An expert LLM development guide for the Hugging Face ecosystem. Features PEP 723 compliant tools for environment checking, data inspection, and VRAM estimation, alongside in-depth guides on Scaling Laws (Step/Farseer), GRPO/DPO, and distributed training.
compatibility: Requires python3, uv (recommended for script execution)
---

# Hugging Face LLM Architect

This skill provides a production-ready workflow for training and deploying LLMs using Hugging Face libraries (`transformers`, `trl`, `peft`, `accelerate`). It incorporates the latest research on scaling laws and reasoning alignment.

## 🛠️ Operational Tools (PEP 723 Compliant)

These scripts utilize `uv` for dependency management. You can run them directly if `uv` is installed.

### 1. Environment & Hardware Check
Diagnose your GPU setup, CUDA versions, and interconnects (P2P/NVLink).
```bash
# Run directly (dependencies auto-installed)
./scripts/env_health_check.py
```

### 2. Dataset Stream Inspection

Preview massive datasets from the Hub without downloading them. Checks column structures for Chat Templates.

```bash
# Example: Inspect the UltraChat dataset
./scripts/stream_data_preview.py HuggingFaceH4/ultrachat_200k --split train_sft
```

### 3. VRAM Requirement Calculator

Estimate memory usage for Training (Full/LoRA) and Inference based on model size and context length.

```bash
# Example: 8B model, 8192 context, fp16 precision
./scripts/calc_vram_requirements.py --size 8 --context 8192 --precision fp16 --mode training
```

---

## 📚 Technical Knowledge Base

### Phase 1: Pre-training & Scaling

* **Theory**: Understand **Step Law** (Hyperparameters) and **Farseer** (Loss Prediction) to optimize compute.
* **Practice**: Data mixing recipes for continual pre-training.
* 🔗 [Scaling Laws & Recipes](https://www.google.com/search?q=references/01_pretraining/scaling_laws_step_farseer.md)
* 🔗 [Continual Pre-training Guide](https://www.google.com/search?q=references/01_pretraining/continual_pretraining_recipes.md)

### Phase 2: Supervised Fine-Tuning (SFT)

* **Core**: Chat Templates, Sequence Packing (`packing=True`), and Loss Masking.
* **Efficiency**: LoRA/QLoRA configuration (`target_modules="all-linear"`).
* 🔗 [SFT Best Practices](https://www.google.com/search?q=references/02_fine_tuning/sft_best_practices.md)
* 🔗 [PEFT & LoRA Config](https://www.google.com/search?q=references/02_fine_tuning/peft_lora_config.md)

### Phase 3: Alignment & Reasoning

* **General**: DPO vs ORPO for chat preferences.
* **Reasoning**: **GRPO** (DeepSeek-R1 style) for math/code tasks without a value model.
* 🔗 [Preference Optimization (DPO/ORPO)](https://www.google.com/search?q=references/03_alignment/dpo_orpo_preference.md)
* 🔗 [Reasoning with GRPO](https://www.google.com/search?q=references/03_alignment/grpo_reasoning.md)

### Phase 4: Infrastructure

* **Distributed**: Choosing between DDP, DeepSpeed ZeRO-1/2/3, and FSDP.
* **Inference**: Flash Attention 2, KV Cache Quantization, and vLLM serving.
* 🔗 [Distributed Strategy Matrix](https://www.google.com/search?q=references/04_distributed/strategy_matrix.md)
* 🔗 [DeepSpeed & FSDP Config](https://www.google.com/search?q=references/04_distributed/deepspeed_fsdp_config.md)
* 🔗 [Inference Optimization](https://www.google.com/search?q=references/05_inference/production_optimization.md)
