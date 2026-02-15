# Scaling Laws: Step Law & Farseer

Optimizing compute and hyperparameters before training.

## 1. Step Law (Hyperparameter Optimization)
Step Law focuses on finding the optimal Learning Rate ($lr$) and Batch Size ($B$) for a given model size ($N$) and data size ($D$).

### Core Formulas
* **Learning Rate**: $lr_{opt} \propto N^{-\alpha} \cdot D^{-\beta}$
    * *Insight*: Larger models generally require **lower** learning rates.
* **Batch Size**: $B_{opt} \propto D^{\gamma}$
    * *Insight*: Batch size should grow as you train on more tokens.

### Practical Application
Do not guess the LR. Train a small proxy model (e.g., 100M parameters) on a subset of your data to find its optimal LR. Then, use Step Law coefficients to extrapolate the LR for your target 7B/70B model.

## 2. Farseer (Performance Prediction)
Farseer improves upon Chinchilla by modeling the Loss Surface $L(N, D)$.

* **Variable Compute**: Instead of training one model size to convergence, train multiple small model sizes for varying numbers of steps.
* **Surface Fitting**: Fit the Farseer equation to these points to predict the loss of a much larger model.
* **Efficiency**: Reduces the extrapolation error by ~4x compared to Chinchilla, potentially saving millions in wasted compute on models that won't converge as expected.
