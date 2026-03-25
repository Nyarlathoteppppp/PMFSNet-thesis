# -*- encoding: utf-8 -*-
import torch

from lib.models.PMFSNet import PMFSNet
from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate
from lib.models.modules.LocalPMFSBlock import DownSampleWithLocalPMFSBlock
from lib.models.modules.xbwEdgeAuxHead import xbwEdgeAuxHead


class xbwPMFSNetEdgeAux(PMFSNet):
    def __init__(self, in_channels=1, out_channels=35, dim="3d", scaling_version="TINY"):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            scaling_version=scaling_version,
            basic_module=DownSampleWithLocalPMFSBlock,
            global_module=GlobalPMFSBlock_AP_Separate,
        )

        if scaling_version == "BASIC":
            skip_channels = [24, 48, 64]
        elif scaling_version == "SMALL":
            skip_channels = [12, 24, 24]
        elif scaling_version == "TINY":
            skip_channels = [12, 24, 24]
        else:
            raise RuntimeError(f"{scaling_version} scaling version is not available")

        self.edge_head = xbwEdgeAuxHead(skip_channels[0], dim=dim)

    def forward(self, x):
        output_size = x.shape[2:]

        if self.scaling_version == "BASIC":
            x1, x1_skip = self.down_convs[0](x)
            edge_logits = self.edge_head(x1_skip, output_size)
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
            x1, skip1 = self.down_convs[0](x)
            edge_logits = self.edge_head(skip1, output_size)
            x2, skip2 = self.down_convs[1](x1)
            x3, skip3 = self.down_convs[2](x2)

            x3 = self.Global([x1, x2, x3])
            skip3 = self.bottle_conv(torch.cat([x3, skip3], dim=1))

            skip2 = self.upsample_1(skip2)
            skip3 = self.upsample_2(skip3)

            out = self.out_conv(torch.cat([skip1, skip2, skip3], dim=1))
            out = self.upsample_out(out)

        if self.training:
            return out, edge_logits
        return out
