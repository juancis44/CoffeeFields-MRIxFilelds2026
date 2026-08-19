"""NAFNet backbone for 2.5D supervised MRI restoration (Task 2, Fase 2).

NAFNet (Chen et al., ECCV 2022, "Simple Baselines for Image Restoration") is a
strong, efficient restoration net: no attention softmax, no GELU — just a
"SimpleGate" (elementwise product of two halves) and a simplified channel
attention. It fits comfortably in 24 GB at 2.5D slice resolution.

Adaptations for this challenge:
    * input_nc  = 2K+1 (the 2.5D source stack); output_nc = 1.
    * global residual over the CENTER source channel: pred = center + net(x).
      For 0.1T->high-field the contrast gap is large, so the residual just
      gives the optimizer a sensible starting point (identity-ish) — toggle
      with `global_residual`.
    * internal pad/unpad to a multiple of 2**depth, so any crop size works
      regardless of the bbox snap.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and multiply: the core NAFNet nonlinearity."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop: float = 0.0):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)   # depthwise
        self.sg = SimpleGate()                                     # dw -> dw//2
        # simplified channel attention on the gated (dw//2) features
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv1(self.norm1(inp))
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + self.drop1(x) * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + self.drop2(x) * self.gamma


class NAFNet(nn.Module):
    """NAFNet for 2.5D restoration.

    Args:
        input_nc:  number of input channels (2K+1 for a 2.5D stack).
        output_nc: number of output channels (1).
        width:     base channel width.
        enc_blk_nums / dec_blk_nums: blocks per encoder/decoder stage.
                   len(enc) == len(dec) == number of down/up samples (= depth).
        middle_blk_num: blocks at the bottleneck.
        global_residual: add the center input channel to the output.
        drop: dropout in NAFBlocks.
    """

    def __init__(
        self,
        input_nc: int = 5,
        output_nc: int = 1,
        width: int = 32,
        enc_blk_nums=(2, 2, 4),
        middle_blk_num: int = 6,
        dec_blk_nums=(2, 2, 2),
        global_residual: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        assert len(enc_blk_nums) == len(dec_blk_nums), "enc/dec depth must match"
        self.depth = len(enc_blk_nums)
        self.global_residual = global_residual
        self.center = input_nc // 2   # index of the center slice channel

        self.intro = nn.Conv2d(input_nc, width, 3, padding=1)
        self.ending = nn.Conv2d(width, output_nc, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        c = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(c, drop=drop) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2

        self.middle = nn.Sequential(*[NAFBlock(c, drop=drop) for _ in range(middle_blk_num)])

        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c, drop=drop) for _ in range(n)]))

    def _pad(self, x):
        _, _, h, w = x.shape
        m = 2 ** self.depth
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return x, (h, w)

    def forward(self, x):
        center = x[:, self.center:self.center + 1] if self.global_residual else None
        x, (H, W) = self._pad(x)

        feat = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = dec(feat)

        out = self.ending(feat)
        out = out[:, :, :H, :W]                      # unpad
        if self.global_residual:
            out = out + center
        return out


def build_nafnet(cfg: dict) -> NAFNet:
    """Build a NAFNet from a model-config dict."""
    return NAFNet(
        input_nc=cfg.get("input_nc", 5),
        output_nc=cfg.get("output_nc", 1),
        width=cfg.get("width", 32),
        enc_blk_nums=tuple(cfg.get("enc_blk_nums", [2, 2, 4])),
        middle_blk_num=cfg.get("middle_blk_num", 6),
        dec_blk_nums=tuple(cfg.get("dec_blk_nums", [2, 2, 2])),
        global_residual=cfg.get("global_residual", True),
        drop=cfg.get("drop", 0.0),
    )
