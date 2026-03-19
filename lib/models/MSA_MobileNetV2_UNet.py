import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.mobilenetv2 import InvertedResidual, mobilenet_v2
from lib.models.modules.ConvBlock import ConvBlock


class ECABlock(nn.Module):
    """Lightweight channel attention used after multi-scale fusion."""

    def __init__(self, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1)  # [B, C, 1]
        y = self.conv(y.transpose(-1, -2))  # [B, 1, C]
        y = self.sigmoid(y.transpose(-1, -2)).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


class MultiScaleDepthwise(nn.Module):
    """Parallel depthwise convs with different dilations, then fuse + channel attention."""

    def __init__(self, channels: int, stride: int, dilations=(1, 2), use_eca: bool = True):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=d, dilation=d,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True)
            )
            for d in dilations
        ])
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels * len(dilations), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True)
        )
        self.eca = ECABlock() if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.pointwise(x)
        return self.eca(x)


class MSAInvertedResidual(nn.Module):
    """
    Inverted residual with multi-scale depthwise conv + ECA.
    This keeps the lightweight MobileNetV2 structure while enhancing receptive fields.
    """

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, dilations=(1, 2), use_eca: bool = True):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (self.stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        layers.append(MultiScaleDepthwise(hidden_dim, stride, dilations=dilations, use_eca=use_eca))

        layers.extend([
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample + concat skip + conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(
            in_channel=in_ch + skip_ch,
            out_channel=out_ch,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True,
            dim="2d"
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MSA_MobileNetV2_UNet(nn.Module):
    """
    Multi-Scale Attention MobileNetV2 U-Net.
    - Encoder: MobileNetV2 backbone; stage 3 & 4 use multi-scale depthwise conv with dilations (1, 2) + ECA.
    - Decoder: UNet-style upsampling with skip connections fused by 1x1/3x3 conv.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 2, dilations=(1, 2), use_eca: bool = True):
        super().__init__()
        backbone = mobilenet_v2(weights="IMAGENET1K_V1")
        # Rebuild encoder so we can swap blocks cleanly.
        settings = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],  # stage 3 -> replace
            [6, 96, 3, 1],  # stage 4 -> replace
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        msa_stages = {3, 4}  # zero-based stage id (after the initial conv)

        features = []
        collect_indices = []

        # First conv
        features.append(backbone.features[0])
        collect_indices.append(len(features) - 1)  # /2

        input_channel = 32
        stage_id = 0
        for t, c, n, s in settings:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                if stage_id in msa_stages:
                    block = MSAInvertedResidual(input_channel, output_channel, stride, t,
                                                dilations=dilations, use_eca=use_eca)
                else:
                    block = InvertedResidual(input_channel, output_channel, stride, t)
                features.append(block)
                input_channel = output_channel

            collect_indices.append(len(features) - 1)
            stage_id += 1

        # Last 1x1 conv
        last_channel = 1280
        features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))
        collect_indices.append(len(features) - 1)

        self.encoder = nn.Sequential(*features)
        self.collect_indices = collect_indices
        # pick 5 scales: conv0(/2), stage1(/4), stage2(/8), stage4(/16), bottleneck(/32)
        # use final 1x1 conv output (1280 ch) as deepest feature
        self.selected_positions = [0, 2, 3, 5, 8]

        # Channel specs for decoder
        skip_channels = [32, 24, 32, 96, last_channel]

        self.up1 = UpBlock(skip_channels[4], skip_channels[3], 256)  # /32 -> /16
        self.up2 = UpBlock(256, skip_channels[2], 160)               # /16 -> /8
        self.up3 = UpBlock(160, skip_channels[1], 96)               # /8 -> /4
        self.up4 = UpBlock(96, skip_channels[0], 64)                # /4 -> /2

        self.head = nn.Sequential(
            ConvBlock(64, 64, kernel_size=3, stride=1, batch_norm=True, preactivation=True, dim="2d"),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]
        feats = []
        out = x
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            if idx in self.collect_indices:
                feats.append(out)

        # Select scales in shallow -> deep order
        feats = [feats[i] for i in self.selected_positions]

        x = feats[-1]
        x = self.up1(x, feats[-2])
        x = self.up2(x, feats[-3])
        x = self.up3(x, feats[-4])
        x = self.up4(x, feats[-5])

        x = self.head(x)
        x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        return x
