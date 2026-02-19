# Advanced Model Tuning & Alignment Strategies

This document serves as a technical reference for the Agent Skills lifecycle, covering the spectrum from Pre-training to specialized Reasoning Reinforcement.

## 1. Foundation: Pre-Training & Continued Pre-Training (CPT)

Before alignment, models must build a compressed representation of world knowledge. SFT is inefficient for fact injection; use CPT or RAG instead.

### Pre-Training (PT)

* **Goal**: General linguistic competence and world knowledge.
* **Data Scale**: Trillions of tokens (e.g., Llama 3 > 15T).
* **Chinchilla Law**: Optimal compute allocation suggests $\approx 20$ tokens per model parameter. Modern "over-training" (e.g., 8B model on 15T tokens) is preferred for superior inference-time capability.

### Continued Pre-Training (CPT) / Domain Adaptation

* **Goal**: Injecting specialized domain knowledge (Medical, Legal, Code) into base models.
* **Challenge**: Catastrophic forgetting of general reasoning.
* **Strategy**: Mix 10-20% general pre-training data with domain corpora.
* **Technique (ADEPT)**: Selective layer expansion and decoupled tuning to balance knowledge retention.

---

## 2. Supervised Fine-Tuning (SFT)

SFT maps pre-trained knowledge to specific instruction-following formats. **Quality outweighs quantity.**

* **Surface Alignment (LIMA Hypothesis)**: General chat capabilities can be "unlocked" with as few as **1,000 highly-curated samples**.
* **Deep Reasoning / Distillation**: Teaching "System 2" behaviors (CoT) requires massive scale. **DeepSeek-R1** utilized **~800k samples** (600k reasoning + 200k general).
* **Cold Start (for RL)**: Establishes the initial reasoning format (e.g., `<thinking>` tags) to stabilize RL exploration. Typically requires **2k–5k high-quality CoT pairs**.

---

## 3. Preference Optimization (Alignment)

Aligns model output distributions with human values (Helpful, Honest, Harmless).

### Offline Methods (Stable & Resource Efficient)

* **DPO (Direct Preference Optimization)**: Uses the reference model as an implicit reward model. Standard for most alignment tasks.
* **ORPO / SimPO**: Reference-free. Integrates alignment into SFT or uses margin-based objectives. Ideal for low-VRAM constraints.
* **KTO**: Uses binary feedback (Thumbs Up/Down). Best for utilizing production logs where paired data is scarce.

### Online Methods (High Performance)

* **PPO (Proximal Policy Optimization)**: The benchmark for exploration. Requires 4 models (Policy, Ref, Reward, Critic) in memory.
* **Iterative DPO**: Periodically updates the preference dataset with on-policy generations labeled by an external RM.

---

## 4. Reasoning & Verifiable RL (The "System 2" Shift)

Focused on verifiable tasks (Math, Logic, Coding) where rewards are rule-based (e.g., compilers).

### Critic-Free Algorithms Comparison (PPO vs. GRPO vs. RLOO)

To save VRAM (~50% reduction) and handle sparse rewards, models use group-based advantages instead of a Critic/Value network.

| Feature | **PPO** | **GRPO** | **RLOO** |
| :--- | :--- | :--- | :--- |
| **Type** | Online (Actor-Critic) | Online (Policy Gradient) | Online (Policy Gradient) |
| **VRAM Usage** | **Very High** (4 Models) | **Low** (2 Models: Policy+Ref) | **Low** (2 Models: Policy+Ref) |
| **Critic Model** | **Required** | Not Required | Not Required |
| **Baseline** | Learned Value Network | Group Mean (Biased) | Leave-One-Out Mean (Unbiased) |
| **Gradient Est.** | Biased | Biased (variance-reduced) | **Unbiased** |
| **Best For** | General Chat, Dense RM | **Reasoning (Math/Code)** | **Reasoning**, Low-VRAM |

---

## 5. Reference: Typical Dataset Scales

Recommended volumes for **7B–70B** parameter models.

### 5.1 Continued Pre-Training (Knowledge Injection)

*Data Format: Raw Text (Unsupervised)*

| Use Case | Typical Volume (Tokens) | Notes |
| :--- | :--- | :--- |
| **Vertical Domain** | **5B - 20B** | Medical/Legal adaptation. Mix in general data. |
| **New Language** | **10B - 50B+** | Requires Tokenizer updates + structural mastery. |
| **Internal Knowledge** | **1B - 5B** | High-density private data (Codebases/Manuals). |

### 5.2 Supervised Fine-Tuning (Capability Activation)

*Data Format: Instruction-Response Pairs*

| Use Case | Typical Volume (Samples) | Notes |
| :--- | :--- | :--- |
| **General Chat** | **1k - 10k** | High human curation; focus on diversity. |
| **Task Specific** | **500 - 2k** | Extraction/Classification. Avoid over-fitting. |
| **Deep Reasoning** | **100k - 1M** | Reasoning Distillation (e.g., DeepSeek-R1). |
| **RL Cold Start** | **2k - 5k** | Stabilize CoT format before RL. |

### 5.3 Value Alignment (Preference Optimization)

*Data Format: Preference Pairs or Prompts*

| Stage/Method | Typical Volume | Notes |
| :--- | :--- | :--- |
| **Reward Modeling** | **100k - 1M (Pairs)** | High diversity required to prevent reward hacking. |
| **DPO / ORPO** | **10k - 60k (Pairs)** | Quality of preference gap is more critical than scale. |
| **Online RL** | **10k - 50k (Prompts)** | RL requires **prompts**; responses are self-generated. |

---

## 6. Decision Logic: Strategy Selection

1. **Need Facts?** -> **CPT** (Tokens) or **RAG**. SFT is not for facts.
2. **Need Reasoning?**
    * *Step 1*: **Cold Start SFT** (2k+ CoT samples).
    * *Step 2*: **GRPO** (Efficient) or **RLOO** (Unbiased). Use **Verifiable Rewards**.
3. **Need Style/Safety?** -> **DPO** (Standard) or **ORPO** (Resource-save).
4. **Limited Data?** -> **SFT** (1k gold samples) or **KTO** (Binary logs).
