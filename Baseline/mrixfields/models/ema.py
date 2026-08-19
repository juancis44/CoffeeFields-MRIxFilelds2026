"""Exponential Moving Average of model weights.

EMA weights are almost always sharper and more stable at eval time for
restoration/GAN models — cheap and consistently helps every metric. Use the
shadow (EMA) weights for validation, checkpointing and inference.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(self._unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = self._unwrap(model).state_dict()
        for k, v in self.ema.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(mv.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(mv)         # buffers like num_batches_tracked

    def state_dict(self):
        return self.ema.state_dict()
