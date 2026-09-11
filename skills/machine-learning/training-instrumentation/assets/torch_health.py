"""Training-health measurements for a PyTorch loop, behind one function.

Drop this module into the project and call it after the optimizer step:

    from torch_health import TrainingHealth

    health = TrainingHealth(model, max_norm=max_norm, per_layer_every=50,
                            histogram_every=1000, histogram_names=("embed", "attn"))
    ...
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
    metrics = health.after_step(model, optimizer, loss_vec, grad_norm, scaler)
    log_metrics(step, metrics)        # the loop's single logging seam

`metrics` is a flat dict of floats, plus a `histograms` entry at the
histogram cadence holding bounded bin counts and edges (never tensors).

Every step: loss mean and quantiles; gradient norm and the clipping
indicator (pass the loop's `max_norm`, or it stays 0); `numerics/*`
finiteness of the loss, the gradient norm, and every parameter after the
step; the scaler's scale, whether this step was skipped (the scale fell),
and the cumulative skip count.

Every `per_layer_every` steps (default 50): per-layer gradient norms, the
global and per-layer update-to-weight ratios (parameters are snapshotted
one step earlier and the copy is freed after the measurement, so no
model-sized copy lives on the hot path), and optimizer second-moment
health: the mean of `v` and the ratio of the current squared gradient to
it (well above 1 means `v` underestimates the gradient). Set
`per_layer_every=0` to disable the sampled measurements.
"""

from __future__ import annotations

import math
from typing import Any

import torch


class TrainingHealth:
    def __init__(
        self,
        model: torch.nn.Module,
        max_norm: float | None = None,
        per_layer_every: int = 50,
        histogram_every: int = 1000,
        histogram_names: tuple[str, ...] = (),
        histogram_bins: int = 64,
    ) -> None:
        self.max_norm = max_norm
        self.per_layer_every = per_layer_every
        self.histogram_every = histogram_every
        self.histogram_names = histogram_names
        self.histogram_bins = histogram_bins
        self.step = 0
        self.skipped_steps = 0
        self._prev_scale: float | None = None
        self._prev: dict[str, torch.Tensor] | None = None
        self._model = model

    @torch.no_grad()
    def after_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_vec: torch.Tensor,
        grad_norm: torch.Tensor | float,
        scaler: Any | None = None,
    ) -> dict[str, Any]:
        self.step += 1
        m: dict[str, Any] = {}
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        lv = loss_vec.detach().float().flatten()
        m["loss/mean"] = lv.mean().item()
        if lv.numel() > 1:
            q = torch.quantile(lv, torch.tensor([0.5, 0.9, 0.99], device=lv.device))
            m["loss/p50"], m["loss/p90"], m["loss/p99"] = (x.item() for x in q)
            m["loss/max"] = lv.max().item()

        gn = float(grad_norm)
        m["grad/norm"] = gn
        m["grad/clipped"] = float(self.max_norm is not None and gn > self.max_norm)

        # Finiteness: loss, gradient norm, and every parameter after the step
        # (one device sync for all parameters, not one per parameter).
        params_finite = torch.ones((), dtype=torch.bool, device=lv.device)
        for _, p in params:
            params_finite &= torch.isfinite(p).all()
        m["numerics/loss_finite"] = float(math.isfinite(m["loss/mean"]))
        m["numerics/grad_finite"] = float(math.isfinite(gn))
        m["numerics/params_finite"] = float(params_finite.item())
        m["numerics/all_finite"] = float(
            m["numerics/loss_finite"] and m["numerics/grad_finite"] and m["numerics/params_finite"]
        )

        if scaler is not None and hasattr(scaler, "get_scale"):
            scale = float(scaler.get_scale())
            # GradScaler lowers the scale exactly when it found non-finite
            # gradients and skipped optimizer.step(); a fall is a skipped step.
            skipped = self._prev_scale is not None and scale < self._prev_scale
            self.skipped_steps += int(skipped)
            m["amp/scale"] = scale
            m["amp/skipped_step"] = float(skipped)
            m["amp/skipped_steps_total"] = float(self.skipped_steps)
            self._prev_scale = scale

        sampled = self.per_layer_every > 0 and self.step % self.per_layer_every == 0
        if sampled:
            update_sq = weight_sq = 0.0
            for name, p in params:
                w = p.detach().float().pow(2).sum().item()
                weight_sq += w
                if p.grad is not None:
                    m[f"layer/{name}/grad_norm"] = p.grad.detach().float().norm().item()
                if self._prev is not None and name in self._prev:
                    u = (p.detach() - self._prev[name]).float().pow(2).sum().item()
                    update_sq += u
                    m[f"layer/{name}/update_weight_ratio"] = math.sqrt(u) / (math.sqrt(w) + 1e-12)
            if self._prev is not None:
                m["update/weight_ratio"] = math.sqrt(update_sq) / (math.sqrt(weight_sq) + 1e-12)
            self._prev = None  # free the model-sized copy
            m.update(self._adam_state(params, optimizer))
        if self.per_layer_every > 0 and (self.step + 1) % self.per_layer_every == 0:
            # Snapshot one step before the next measurement so the ratio is a one-step update.
            self._prev = {n: p.detach().clone() for n, p in params}

        if self.histogram_names and self.histogram_every > 0 and self.step % self.histogram_every == 0:
            m["histograms"] = {
                f"grad/{n}": self._histogram(p.grad)
                for n, p in params
                if p.grad is not None and any(tag in n for tag in self.histogram_names)
            }
        return m

    def _histogram(self, tensor: torch.Tensor) -> dict[str, list[float]]:
        # Bounded bin counts and edges only — never the tensor itself.
        g = tensor.detach().float().flatten()
        lo, hi = g.min().item(), g.max().item()
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return {"counts": [], "edges": [lo, hi]}
        if lo == hi:
            hi = lo + 1e-12
        counts = torch.histc(g, bins=self.histogram_bins, min=lo, max=hi)
        return {"counts": counts.cpu().tolist(), "edges": [lo, hi]}

    @staticmethod
    def _adam_state(params: list[tuple[str, torch.Tensor]], optimizer: torch.optim.Optimizer) -> dict[str, float]:
        # Second-moment health: v that lags a growing g² lets the preconditioned
        # update blow up; the ratio g²/v rising well above 1 is the early sign.
        v_sum, ratio_sum, n = 0.0, 0.0, 0
        for _, p in params:
            v = optimizer.state.get(p, {}).get("exp_avg_sq")
            if v is None or p.grad is None:
                continue
            vf = v.float()
            g2 = p.grad.detach().float().pow(2)
            v_sum += vf.mean().item()
            ratio_sum += (g2.mean() / (vf.mean() + 1e-30)).item()
            n += 1
        if n == 0:
            return {}
        return {"adam/mean_v": v_sum / n, "adam/g2_over_v": ratio_sum / n}
