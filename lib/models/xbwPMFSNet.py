# -*- encoding: utf-8 -*-
from lib.models.PMFSNet import PMFSNet
from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate
from lib.models.modules.xbwLocalRefineBlock import xbwDownSampleWithLocalRefineBlock


class xbwPMFSNet(PMFSNet):
    def __init__(self, in_channels=1, out_channels=35, dim="3d", scaling_version="TINY"):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            scaling_version=scaling_version,
            basic_module=xbwDownSampleWithLocalRefineBlock,
            global_module=GlobalPMFSBlock_AP_Separate,
        )
