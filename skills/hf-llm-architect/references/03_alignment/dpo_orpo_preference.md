# Preference Optimization: DPO vs ORPO

Moving beyond SFT to align with human preferences.

## 1. DPO (Direct Preference Optimization)
* **Status**: Industry standard. Stable and memory efficient.
* **Requires**: SFT Model + Reference Model (usually the same SFT model frozen).
* **Data**: Triplets `(prompt, chosen, rejected)`.
* **Key Param**: `beta` (0.1 is standard). Higher beta = more conservative (closer to ref model).

## 2. ORPO (Odds Ratio Preference Optimization)
* **Status**: Newer method. Merges SFT and Alignment into one stage.
* **Pros**: No Reference Model needed (saved VRAM).
* **Cons**: Slightly more sensitive to hyperparameters.
* **Use Case**: When you want to train from a Base model directly to Aligned model using preference data, skipping a separate SFT stage.

## 3. SimPO (Simple Preference Optimization)
* **Idea**: Like DPO but uses the average log probability of the sequence as the reward, normalized by length.
* **Pros**: Helps prevent the model from gaming the reward by just generating longer responses.
