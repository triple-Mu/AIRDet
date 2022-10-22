# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from ..core.base_ops import BaseConv, CSPLayer, DWConv
from ..core.neck_ops import CSPStage


class GiraffeNeckV2(nn.Module):

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=[2, 3, 4],
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        spp=True,
        reparam_mode=True,
        block_name='BasicBlock',
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        reparam_mode = reparam_mode

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # node x3: input x0, x1
        self.bu_conv13 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        if reparam_mode:
            self.merge_3 = CSPStage(
                block_name,
                int((in_channels[1] + in_channels[2]) * width),
                int(in_channels[2] * width),
                round(3 * depth),
                act = act,
                spp = spp)
        else:
            self.merge_3 = CSPLayer(
                int((in_channels[1] + in_channels[2]) * width),
                int(in_channels[2] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act = act)

        # node x4: input x1, x2, x3
        self.bu_conv24 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        if reparam_mode:
            self.merge_4 = CSPStage(
                block_name,
                int((in_channels[0] + in_channels[1] + in_channels[2]) * width),
                int(in_channels[1] * width),
                round(3 * depth),
                act = act,
                spp = spp)
        else:
            self.merge_4 = CSPLayer(
                int((in_channels[0] + in_channels[1] + in_channels[2]) * width),
                int(in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act = act)


        # node x5: input x2, x4
        if reparam_mode:
            self.merge_5 = CSPStage(
                block_name,
                int((in_channels[1] + in_channels[0]) * width),
                int(out_channels[0] * width),
                round(3 * depth),
                act = act,
                spp = spp)
        else:
            self.merge_5 = CSPLayer(
                int((in_channels[1] + in_channels[0]) * width),
                int(out_channels[0] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act = act)


        # node x8: input x4, x5
        # self.merge_8 = CSPStage(
        #    'BasicBlock',
        #    int((in_channels[0] + in_channels[1]) * width),
        #    int(in_channels[0] * width),
        #    round(3 * depth),
        #    act = act,
        #    spp = spp)

        # node x7: input x4, x5
        self.bu_conv57 = Conv(
            int(out_channels[0] * width), int(out_channels[0] * width), 3, 2, act=act)
        if reparam_mode:
            self.merge_7 = CSPStage(
               block_name,
               int((out_channels[0] + in_channels[1]) * width),
               int(out_channels[1] * width),
               round(3 * depth),
               act = act,
               spp = spp)
        else:
            self.merge_7 = CSPLayer(
               int((out_channels[0] + in_channels[1]) * width),
               int(out_channels[1] * width),
               round(3 * depth),
               False,
               depthwise=depthwise,
               act = act)


        # node x6: input x3, x4, x7
        self.bu_conv46 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.bu_conv76 = Conv(
            int(out_channels[1] * width), int(out_channels[1] * width), 3, 2, act=act)
        if reparam_mode:
            self.merge_6 = CSPStage(
               block_name,
               int((in_channels[1] + out_channels[1] + in_channels[2]) * width),
               int(out_channels[2] * width),
               round(3 * depth),
               act = act,
               spp = spp)
        else:
            self.merge_6 = CSPLayer(
               int((in_channels[1] + out_channels[1] + in_channels[2]) * width),
               int(out_channels[2] * width),
               round(3 * depth),
               False,
               depthwise=depthwise,
               act = act)


    def init_weights(self):
        pass

    def forward(self, out_features):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        #features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = out_features

        # node x3
        x13 = self.bu_conv13(x1)
        x3 = torch.cat([x0, x13], 1)
        x3 = self.merge_3(x3)

        # node x4
        x34 = self.upsample(x3)
        x24 = self.bu_conv24(x2)
        x4 = torch.cat([x1, x24, x34], 1)
        x4 = self.merge_4(x4)

        # node x5
        x45 = self.upsample(x4)
        x5 = torch.cat([x2, x45], 1)
        x5 = self.merge_5(x5)

        # node x8
        # x8 = x5

        # node x7
        x57 = self.bu_conv57(x5)
        x7 = torch.cat([x4, x57], 1)
        x7 = self.merge_7(x7)

        # node x6
        x46 = self.bu_conv46(x4)
        x76 = self.bu_conv76(x7)
        x6 = torch.cat([x3, x46, x76], 1)
        x6 = self.merge_6(x6)

        outputs = (x5, x7, x6)
        return outputs
