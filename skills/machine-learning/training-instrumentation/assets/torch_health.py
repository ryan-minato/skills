"""Training-health measurements for a PyTorch loop, behind one function.

Drop this module into the project and call it after the optimizer step:

    from torch_health import TrainingHealth

    health = TrainingHealth(model, per_layer_every=50, histogram_every=1000,
                            histogram_names=("embed", "attn"))
    ...
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
    metrics = health.after_step(model, optimizer, loss_vec, grad_norm, scaler)
    log_metrics(step, metrics)        # the loop's single logging seam

`metrics` is a flat dict of floats (plus a `histograms` entry at the
histogram frequency, holding CPU tensors the seam may forward or drop).
Every-step keys: loss mean and quantiles, gradient norm, clipping
indicator, all-finite flag, scaler scale, global update-to-weight ratio.
Per-layer keys every `per_layer_every` steps: gradient norm and
update-to-weight ratio per parameter group name. Optimizer keys at the
same cadence: mean second moment and the ratio of the current squared
gradient to it (values well above 1 mean v underestimates the gradient).
"""

from __future__ import annotations

import math
from typing import Any

import torch


class TrainingHealth:
    def __init__(
        self,
        model: torch.nn.Module,
        per_layer_every: int = 50,
        histogram_every: int = 1000,
        histogram_names: tuple[str, ...] = (),
        max_norm: float | None = None,
    ) -> None:
        self.per_layer_every = per_layer_every
        self.histogram_every = histogram_every
        self.histogram_names = histogram_names
        self.max_norm = max_norm
        self.step = 0
        self._prev = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

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
        lv = loss_vec.detach().float().flatten()
        m["loss/mean"] = lv.mean().item()
        if lv.numel() > 1:
            q = torch.quantile(lv, torch.tensor([0.5, 0.9, 0.99], device=lv.device))
            m["loss/p50"], m["loss/p90"], m["loss/p99"] = (x.item() for x in q)
            m["loss/max"] = lv.max().item()
        gn = float(grad_norm)
        m["grad/norm"] = gn
        m["grad/clipped"] = float(self.max_norm is not None and gn > self.max_norm)
        finite = math.isfinite(m["loss/mean"]) and math.isfinite(gn)
        m["numerics/all_finite"] = float(finite)
        if scaler is not None and hasattr(scaler, "get_scale"):
            m["amp/scale"] = float(scaler.get_scale())

        per_layer = self.step % self.per_layer_every == 0
        update_sq = weight_sq = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            delta = p.detach() - self._prev[name]
            u = delta.float().pow(2).sum().item()
            w = p.detach().float().pow(2).sum().item()
            update_sq += u
            weight_sq += w
            if per_layer:
                m[f"layer/{name}/update_weight_ratio"] = math.sqrt(u) / (math.sqrt(w) + 1e-12)
                if p.grad is not None:
                    m[f"layer/{name}/grad_norm"] = p.grad.detach().float().norm().item()
            self._prev[name].copy_(p.detach())
        m["update/weight_ratio"] = math.sqrt(update_sq) / (math.sqrt(weight_sq) + 1e-12)

        if per_layer:
            m.update(self._adam_state(model, optimizer))

        if self.histogram_names and self.step % self.histogram_every == 0:
            m["histograms"] = {
                f"grad/{n}": p.grad.detach().float().cpu()
                for n, p in model.named_parameters()
                if p.grad is not None and any(tag in n for tag in self.histogram_names)
            }
        return m

    @staticmethod
    def _adam_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        # Second-moment health: v that lags a growing g² lets the preconditioned
        # update blow up; the ratio g²/v rising well above 1 is the early sign.
        v_sum, ratio_sum, n = 0.0, 0.0, 0
        for group in optimizer.param_groups:
            for p in group["params"]:
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
