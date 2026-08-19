"""MIND structure-consistency loss (Fase 3).

MIND = Modality-Independent Neighbourhood Descriptor (Heinrich et al., MedIA
2012). For each voxel it encodes the *local self-similarity* pattern, which is
invariant to absolute intensity / contrast. Matching MIND(pred) to MIND(target)
therefore enforces **anatomical structure** without demanding identical
contrast — exactly what we want to protect the SynthSeg Dice / Volume metrics
while a weak GAN sharpens the image.

Why not SynthSeg in the loop? SynthSeg is a heavy 3D TensorFlow model and is not
differentiable here; it stays as the offline *evaluation* segmenter. MIND is a
cheap, differentiable, contrast-robust proxy that correlates with structural
overlap.

Inputs are expected in [-1, 1] (model range) and converted to [0, 1] internally.
Works on (B, 1, H, W).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MINDLoss(nn.Module):
    def __init__(self, radius: int = 2, patch: int = 3, eps: float = 1e-5):
        super().__init__()
        self.patch = patch
        self.eps = eps
        # 4-neighbourhood at the given radius (2D analogue of MIND-SSC)
        self.offsets: List[Tuple[int, int]] = [
            (radius, 0), (-radius, 0), (0, radius), (0, -radius)
        ]

    def _mind(self, img: torch.Tensor) -> torch.Tensor:
        pad = self.patch // 2
        descs = []
        for dy, dx in self.offsets:
            shifted = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
            diff = (img - shifted) ** 2
            Dp = F.avg_pool2d(diff, self.patch, stride=1, padding=pad)  # patch SSD
            descs.append(Dp)
        D = torch.cat(descs, dim=1)                                     # (B, n, H, W)
        V = D.mean(dim=1, keepdim=True).clamp_min(self.eps)            # local variance
        mind = torch.exp(-D / V)
        mind = mind / mind.max(dim=1, keepdim=True)[0].clamp_min(self.eps)
        return mind

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p01 = (pred + 1.0) * 0.5
        t01 = (target + 1.0) * 0.5
        return F.l1_loss(self._mind(p01), self._mind(t01))
