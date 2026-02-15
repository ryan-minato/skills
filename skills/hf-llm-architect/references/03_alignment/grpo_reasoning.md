# GRPO: Reasoning Alignment (DeepSeek-R1 Style)

**Group Relative Policy Optimization (GRPO)** is designed for Reinforcement Learning (RL) on tasks with verifiable outcomes (Math, Code) without the massive overhead of a Critic model.

## Core Mechanics
1.  **Group Sampling**: For one prompt, sample $G$ outputs (e.g., 8 or 16).
2.  **Relative Reward**: Score all outputs. Calculate advantage based on the group mean/std.
    * $Adv_i = \frac{Score_i - Mean(GroupScore)}{Std(GroupScore)}$
3.  **Policy Update**: Maximize likelihood of high-advantage outputs.

## Configuration in TRL
```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    num_generations=8,           # The 'Group' size
    max_completion_length=1024,
    use_vllm=True,               # Critical for speed!
    beta=0.04                    # KL penalty
)

# Reward Function: Strict Format + Correctness
def reward_correctness(prompts, completions, answer, **kwargs):
    # logic to check if answer is in completion
    return [1.0 if a in c else 0.0 for a, c in zip(answer, completions)]
```

## Why use vLLM?

GRPO generates data *online* during training. Generating 8 sequences for batch size 4 = 32 sequences per step. Without vLLM optimization (paged attention), the generation step will be 10x slower than the training step.
