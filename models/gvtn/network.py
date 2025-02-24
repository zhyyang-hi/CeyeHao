#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   network.py
@Desc    :   Adapted from https://github.com/zhengyang-wang/GVTNets
'''

import torch
import torch.nn as nn
from .basic_ops import *
from .attention_layer import *
from .attention_module import *
from .resnet_module import *


class GVTN(nn.Module):
    """Global Voxel Transformer Network (GVTN)"""

    def __init__(
        self,
        conf_gvtn,
        conf_basic_ops,
        conf_attn_same,
        conf_attn_up,
        conf_attn_down,
    ) -> None:
        super(GVTN, self).__init__()
        self.depth = conf_gvtn["depth"]
        self.first_output_filters = conf_gvtn["first_output_filters"]
        self.encoding_block_sizes = conf_gvtn["encoding_block_sizes"]
        self.downsampling = conf_gvtn["downsampling"]
        self.bottom_block = conf_gvtn["bottom_block"]
        self.decoding_block_sizes = conf_gvtn["decoding_block_sizes"]
        self.upsampling = conf_gvtn["upsampling"]
        self.skip_method = conf_gvtn["skip_method"]
        self.out_kernel_size = conf_gvtn["out_kernel_size"]
        self.out_kernel_bias = conf_gvtn["out_kernel_bias"]
        self.skip_method = conf_gvtn["skip_method"]
        self.conf_gvtn = conf_gvtn
        self.conf_basic_ops = conf_basic_ops
        self.conf_attn_same = conf_attn_same
        self.conf_attn_up = conf_attn_up
        self.conf_attn_down = conf_attn_down

        self.in_conv = Conv2d_gvtn(
            1, self.first_output_filters, 3, 1, False
        )  # in channels = 1 as input is grayscale image

        # encoding block 1
        self.enc_1 = nn.ModuleList(
            [
                ResBlock(self.first_output_filters, self.first_output_filters)
                for _ in range(0, self.encoding_block_sizes[0])
            ]
        )
        # endocing block 2 to n = downsampling + zero or more res_block, i = 2, 3, ..., depth
        self.enc_n = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        self._get_downsampling_func(
                            self.downsampling[i - 1],
                            self.first_output_filters * (2 ** (i - 1)),
                            self.first_output_filters * (2**i),
                            self.conf_attn_down,
                        ),
                    ]
                    + [
                        ResBlock(
                            self.first_output_filters * (2**i),
                            self.first_output_filters * (2**i),
                        )
                        for _ in range(self.encoding_block_sizes[i])
                    ]
                )
                for i in range(1, self.depth)
            ]
        )

        # bottleneck
        self.bottleneck = nn.ModuleList(
            [
                self._get_bottom_func(
                    self.bottom_block[block_index],
                    self.first_output_filters * (2 ** (self.depth - 1)),
                    self.first_output_filters * (2 ** (self.depth - 1)),
                    self.conf_attn_same,
                )
                for block_index in range(len(self.bottom_block))
            ]
        )

        # decoder
        self.dec_n = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        self._get_upsampling_func(
                            self.upsampling[self.depth - 2 - i],
                            self.first_output_filters * (2 ** (i + 1)),
                            self.first_output_filters * (2**i),
                            self.conf_attn_up,
                        )
                    ]
                    + [
                        (
                            ResBlock(
                                self.first_output_filters * (2**i),
                                self.first_output_filters * (2**i),
                            )
                            if self.skip_method == "add"
                            else ResBlock(
                                self.first_output_filters * (2 ** (i + 1)),
                                self.first_output_filters * (2**i),
                            )
                        )
                        for _ in range(self.decoding_block_sizes[-i - 2 + self.depth])
                    ]
                )
                for i in range(self.depth - 2, -1, -1)
            ]
        )

        self.skip_connection = self._get_skip_func(self.skip_method)

        self.bn = BatchNorm_gvtn(self.first_output_filters, self.conf_basic_ops)
        self.relu = ReLU_gvtn(self.conf_basic_ops)
        self.out_tf = Conv2d_gvtn(
            self.first_output_filters, 2, self.out_kernel_size, 1, self.out_kernel_bias
        )

    def forward(self, inputs):

        skip_inputs = []
        inputs = self.in_conv(inputs)
        for blocks in self.enc_1:
            inputs = blocks(inputs)

        for module_i, modules in enumerate(self.enc_n):  # module_i/j for debug
            skip_inputs.append(inputs)
            for block_j, blocks in enumerate(modules):
                inputs = blocks(inputs)

        for blocks in self.bottleneck:
            inputs = blocks(inputs)

        for i, modules in enumerate(self.dec_n):
            inputs = modules[0](inputs)
            inputs = self.skip_connection(inputs, skip_inputs[-i - 1])
            for blocks in modules[1:]:
                inputs = blocks(inputs)

        pred = self.bn(inputs)
        pred = self.relu(pred)
        pred = self.out_tf(pred)

        return inputs, pred  # inputs is the penult

    def _get_downsampling_func(self, name, in_channels, out_channels, conf):
        if name == "down_gto_v1":
            return DownGTOv1(in_channels, out_channels, conf)
        elif name == "down_gto_v2":
            return DownGTOv2(in_channels, out_channels, conf)
        else:
            raise ValueError(f"Unknown downsampling method {name}")

    def _get_bottom_func(self, name, in_channels, out_channels, conf):
        if name == "same_gto":
            return SameGTO(in_channels, out_channels, conf)
        else:
            raise ValueError(f"Unknown bottom block method {name}")

    def _get_upsampling_func(self, name, in_channels, out_channels, conf):
        if name == "up_gto_v1":
            return UpGTOv1(in_channels, out_channels, conf)
        elif name == "up_gto_v2":
            return UpGTOv2(in_channels, out_channels, conf)
        else:
            raise ValueError(f"Unknown upsampling method {name}")

    def _get_skip_func(self, name):
        if name == "add":
            return lambda x, y: torch.add(x, y)
        elif name == "concat":
            return self.concat_skip
        else:
            raise ValueError(f"Unknown skip method {name}")

    def concat_skip(self, x1, x2):
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        return torch.cat([x1, x2], 1)
