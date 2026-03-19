
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate
from lib.models.modules.ConvBlock import ConvBlock

class PMFSNet_MobileNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, dim="2d", **kwargs):
        super(PMFSNet_MobileNet, self).__init__()
        self.dim = dim

        # 加载 MobileNetV2 的特征部分作为编码器
        mobilenet = mobilenet_v2(weights="IMAGENET1K_V1")
        self.encoder = mobilenet.features

        # 用于跳跃连接的层索引和通道数（固定结构）
        self.skip_indices = [2, 4, 7, 14, 18]
        self.skip_channels = [16, 24, 32, 96, 1280]

        # PMFS 多尺度聚合
        self.global_pmfs = GlobalPMFSBlock_AP_Separate(
            skip_channels=self.skip_channels,
            scale_factors=[16, 8, 4, 2, 1],
            in_channels=64,
            inter_channels=64,
            out_channels=64,
            num_levels=5
        )

        # 最后几层卷积生成 mask
        self.decoder = nn.Sequential(
            ConvBlock(sum(self.skip_channels), 64, kernel_size=3, stride=1, batch_norm=True, preactivation=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 提取不同阶段的特征用于跳跃连接
        features = []
        out = x
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            if idx in self.skip_indices:
                features.append(out)

        # 若最后一层不是 skip 则补上
        if len(features) < len(self.skip_channels):
            features.append(out)

        pmfs_out = self.global_pmfs(features)

        # 上采样为原图大小
        out = self.decoder(pmfs_out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
