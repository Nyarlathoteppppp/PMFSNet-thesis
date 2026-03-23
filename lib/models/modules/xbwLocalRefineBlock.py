# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn

from lib.models.modules.ConvBlock import ConvBlock
from lib.models.modules.LocalPMFSBlock import DenseFeatureStackWithLocalPMFSBlock


class xbwECABlock(nn.Module):
    def __init__(self, k_size=3, dim="3d"):
        super().__init__()
        if dim == "3d":
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        elif dim == "2d":
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise RuntimeError(f"{dim} dimension is error")
        self.dim = dim
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels = x.shape[:2]
        y = self.avg_pool(x).reshape(batch_size, channels, 1)
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2))
        if self.dim == "3d":
            y = y.unsqueeze(-1).unsqueeze(-1)
        else:
            y = y.unsqueeze(-1)
        return x * y.expand_as(x)


class xbwMultiScaleDepthwise(nn.Module):
    def __init__(self, channels, dilations=(1, 2), dim="3d"):
        super().__init__()
        if dim == "3d":
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dim == "2d":
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        self.branches = nn.ModuleList([
            nn.Sequential(
                conv(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                ),
                bn(channels),
                nn.ReLU(inplace=True),
            )
            for dilation in dilations
        ])
        self.pointwise = nn.Sequential(
            conv(channels * len(dilations), channels, kernel_size=1, stride=1, bias=False),
            bn(channels),
            nn.ReLU(inplace=True),
        )
        self.eca = xbwECABlock(dim=dim)

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        x = self.pointwise(x)
        return self.eca(x)


class xbwDownSampleWithLocalRefineBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        base_channel,
        kernel_size,
        unit,
        growth_rate,
        skip_channel=None,
        downsample=True,
        skip=True,
        dim="3d",
    ):
        super().__init__()
        self.skip = skip
        self.downsample = ConvBlock(
            in_channel=in_channel,
            out_channel=base_channel,
            kernel_size=kernel_size,
            stride=(2 if downsample else 1),
            batch_norm=True,
            preactivation=True,
            dim=dim,
        )
        self.dfs_with_pmfs = DenseFeatureStackWithLocalPMFSBlock(
            in_channel=base_channel,
            kernel_size=3,
            unit=unit,
            growth_rate=growth_rate,
            dim=dim,
        )
        refine_channels = base_channel + unit * growth_rate
        self.local_refine = xbwMultiScaleDepthwise(refine_channels, dim=dim)

        if skip:
            self.skip_conv = ConvBlock(
                in_channel=refine_channels,
                out_channel=skip_channel,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim,
            )

    def forward(self, x):
        x = self.downsample(x)
        x_local = self.dfs_with_pmfs(x)
        x_refined = self.local_refine(x_local)
        x_out = x_local + x_refined

        if self.skip:
            x_skip = self.skip_conv(x_out)
            return x_out, x_skip
        return x_out
