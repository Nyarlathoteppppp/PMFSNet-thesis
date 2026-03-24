# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.modules.ConvBlock import ConvBlock


class WaveletSkipRefine(nn.Module):
    def __init__(self, channels, dim="2d"):
        super().__init__()
        if dim != "2d":
            raise RuntimeError("WaveletSkipRefine currently supports only 2d features")

        self.low_head = ConvBlock(
            in_channel=channels,
            out_channel=channels,
            kernel_size=1,
            stride=1,
            batch_norm=True,
            preactivation=True,
            dim=dim,
        )
        self.high_head = nn.Sequential(
            ConvBlock(
                in_channel=channels * 3,
                out_channel=channels * 3,
                kernel_size=1,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim,
            ),
            ConvBlock(
                in_channel=channels * 3,
                out_channel=channels * 3,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim,
            ),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def haar_dwt(x):
        original_size = x.shape[-2:]
        pad_h = original_size[0] % 2
        pad_w = original_size[1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        ll = (a + b + c + d) * 0.25
        lh = (-a - b + c + d) * 0.25
        hl = (-a + b - c + d) * 0.25
        hh = (a - b - c + d) * 0.25
        return ll, lh, hl, hh, original_size

    @staticmethod
    def haar_idwt(ll, lh, hl, hh, output_size):
        batch_size, channels, height, width = ll.shape
        out = ll.new_empty((batch_size, channels, height * 2, width * 2))

        out[..., 0::2, 0::2] = ll - lh - hl + hh
        out[..., 0::2, 1::2] = ll - lh + hl - hh
        out[..., 1::2, 0::2] = ll + lh - hl - hh
        out[..., 1::2, 1::2] = ll + lh + hl + hh

        return out[..., :output_size[0], :output_size[1]]

    def forward(self, x):
        ll, lh, hl, hh, original_size = self.haar_dwt(x)
        delta_ll = self.low_head(ll)
        delta_high = self.high_head(torch.cat([lh, hl, hh], dim=1))
        delta_lh, delta_hl, delta_hh = torch.chunk(delta_high, 3, dim=1)
        delta = self.haar_idwt(delta_ll, delta_lh, delta_hl, delta_hh, original_size)
        return x + self.alpha * delta
