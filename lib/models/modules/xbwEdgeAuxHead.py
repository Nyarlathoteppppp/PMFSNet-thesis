# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

from lib.models.modules.ConvBlock import ConvBlock


class xbwEdgeAuxHead(nn.Module):
    def __init__(self, in_channels, dim="2d"):
        super().__init__()
        if dim == "2d":
            conv = nn.Conv2d
            upsample_mode = "bilinear"
        elif dim == "3d":
            conv = nn.Conv3d
            upsample_mode = "trilinear"
        else:
            raise RuntimeError(f"{dim} dimension is error")

        hidden_channels = in_channels
        self.upsample_mode = upsample_mode
        self.block = ConvBlock(
            in_channel=in_channels,
            out_channel=hidden_channels,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True,
            dim=dim,
        )
        self.out_conv = conv(hidden_channels, 1, kernel_size=1, stride=1)

    def forward(self, x, output_size):
        x = self.block(x)
        x = self.out_conv(x)
        return F.interpolate(x, size=output_size, mode=self.upsample_mode, align_corners=False)
