"""Hybrid restoration loss aligned to the Task 2 ranking metrics.

The ranking sums per-metric ranks over {nRMSE, SSIM, LPIPS, Dice, Volume}, so a
single objective must serve pixel accuracy AND perceptual quality at once:

    L1 / Charbonnier   -> nRMSE  (pixel fidelity, anchors structure -> helps Dice)
    1 - SSIM           -> SSIM
    edge (Sobel L1)    -> sharp boundaries (SSIM + perceptual)
    LPIPS              -> LPIPS  (perceptual)

Start L1/SSIM-heavy (regression anchor) and add a small LPIPS weight so outputs
don't blur. A pure-L1 model wins nRMSE/SSIM but loses LPIPS; this balances both.

Convention: `pred` and `target` come in the model/data range [-1, 1]. L1/SSIM/
edge are computed on [0, 1] (metric-aligned); LPIPS expects [-1, 1].
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structure import SSIMLoss
from .perceptual import PerceptualLoss


def charbonnier(pred, target, eps: float = 1e-3):
    """Smooth-L1-like loss; more robust than L1 near zero, sharper than L2."""
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


class HybridRestorationLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 1.0,
        w_ssim: float = 1.0,
        w_edge: float = 0.1,
        w_lpips: float = 0.1,
        use_charbonnier: bool = True,
        lpips_net: str = "alex",
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.w_lpips = w_lpips
        self.use_charbonnier = use_charbonnier

        self.ssim = SSIMLoss() if w_ssim > 0 else None
        self.lpips = PerceptualLoss(net=lpips_net, weight=1.0) if w_lpips > 0 else None
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
        )

    def _edge(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (total_loss, components_dict). pred/target in [-1, 1]."""
        p01 = (pred + 1.0) * 0.5
        t01 = (target + 1.0) * 0.5

        logs: Dict[str, float] = {}
        total = pred.new_zeros(())

        if self.w_l1 > 0:
            l1 = charbonnier(p01, t01) if self.use_charbonnier else F.l1_loss(p01, t01)
            total = total + self.w_l1 * l1
            logs["l1"] = float(l1.detach())

        if self.ssim is not None:
            s = self.ssim(p01, t01)
            total = total + self.w_ssim * s
            logs["ssim"] = float(s.detach())

        if self.w_edge > 0:
            e = F.l1_loss(self._edge(p01), self._edge(t01))
            total = total + self.w_edge * e
            logs["edge"] = float(e.detach())

        if self.lpips is not None:
            lp = self.lpips(pred, target)          # expects [-1, 1]
            total = total + self.w_lpips * lp
            logs["lpips"] = float(lp.detach())

        logs["total"] = float(total.detach())
        return total, logs
