# -*- encoding: utf-8 -*-
import torch

from lib.models.PMFSNet import PMFSNet
from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate
from lib.models.modules.LocalPMFSBlock import DownSampleWithLocalPMFSBlock
from lib.models.modules.xbwWaveletSkipRefine import WaveletSkipRefine


class xbwPMFSNetV2(PMFSNet):
    def __init__(self, in_channels=1, out_channels=35, dim="3d", scaling_version="TINY"):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            scaling_version=scaling_version,
            basic_module=DownSampleWithLocalPMFSBlock,
            global_module=GlobalPMFSBlock_AP_Separate,
        )
        if dim != "2d":
            raise RuntimeError("xbwPMFSNetV2 currently supports only 2d datasets")

        if scaling_version == "BASIC":
            base_channels = [24, 48, 64]
            units = [5, 10, 10]
        elif scaling_version == "SMALL":
            base_channels = [24, 24, 24]
            units = [5, 10, 10]
        elif scaling_version == "TINY":
            base_channels = [24, 24, 24]
            units = [3, 5, 5]
        else:
            raise RuntimeError(f"{scaling_version} scaling version is not available")

        growth_rates = [4, 8, 16]
        stage1_channels = base_channels[0] + units[0] * growth_rates[0]
        self.skip1_refine = WaveletSkipRefine(stage1_channels, dim=dim)

    def forward(self, x):
        stage1 = self.down_convs[0]
        x1 = stage1.downsample(x)
        x1 = stage1.dfs_with_pmfs(x1)

        if self.scaling_version == "BASIC":
            x1_skip = stage1.skip_conv(self.skip1_refine(x1))
            x2, x2_skip = self.down_convs[1](x1)
            x3 = self.down_convs[2](x2)

            d3 = self.Global([x1, x2, x3])

            d2 = self.up2(d3)
            d2 = torch.cat((x2_skip, d2), dim=1)
            d2 = self.up_conv2(d2)
            d1 = self.up1(d2)
            d1 = torch.cat((x1_skip, d1), dim=1)
            d1 = self.up_conv1(d1)

            out = self.out_conv(d1)
            out = self.upsample_out(out)
        else:
            skip1 = stage1.skip_conv(self.skip1_refine(x1))
            x2, skip2 = self.down_convs[1](x1)
            x3, skip3 = self.down_convs[2](x2)

            x3 = self.Global([x1, x2, x3])
            skip3 = self.bottle_conv(torch.cat([x3, skip3], dim=1))

            skip2 = self.upsample_1(skip2)
            skip3 = self.upsample_2(skip3)

            out = self.out_conv(torch.cat([skip1, skip2, skip3], dim=1))
            out = self.upsample_out(out)

        return out
