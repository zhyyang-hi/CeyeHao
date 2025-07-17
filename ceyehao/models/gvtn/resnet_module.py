#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   resnet_module.py
@Desc    :   Adapted from https://github.com/zhengyang-wang/GVTNets
'''

import torch
import torch.nn as nn
from .basic_ops import *


class ResBlock(nn.Module):
    def __init__(self, in_channels, output_filters):
        super().__init__()
        self.output_filters = output_filters
        self.bn1 = BatchNorm_gvtn(in_channels)
        self.relu = ReLU_gvtn()
        self.projection_shortcut = Conv2d_gvtn(in_channels, output_filters, 3, 1, False)
        self.conv1 = Conv2d_gvtn(in_channels, output_filters, 3, 1, False)
        self.bn2 = BatchNorm_gvtn(output_filters)
        self.conv2 = Conv2d_gvtn(output_filters, output_filters, 3, 1, False)

    def forward(self, inputs):
        if inputs.shape[1] == self.output_filters:
            shortcut = inputs
            inputs = self.bn1(inputs)
            inputs = self.relu(inputs)
        else:
            inputs = self.bn1(inputs)
            inputs = self.relu(inputs)
            shortcut = self.projection_shortcut(inputs)
        inputs = self.conv1(inputs)
        inputs = self.bn2(inputs)
        inputs = self.relu(inputs)
        inputs = self.conv2(inputs)

        return torch.add(shortcut, inputs)


class DownResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        output_filters,
    ):
        super().__init__()
        self.bn1 = BatchNorm_gvtn(in_channels)
        self.relu = ReLU_gvtn()
        self.projection_shortcut = Conv2d_gvtn(in_channels, output_filters, 1, 2, False)
        self.conv1 = Conv2d_gvtn(in_channels, output_filters, 3, 2, False)
        self.bn2 = BatchNorm_gvtn(output_filters)
        self.relu2 = ReLU_gvtn()
        self.conv2 = Conv2d_gvtn(output_filters, output_filters, 3, 1, False)

    def forward(self, inputs):
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        shortcut = self.projection_shortcut(inputs)
        inputs = self.conv1(inputs)
        inputs = self.bn2(inputs)
        inputs = self.relu2(inputs)
        inputs = self.conv2(inputs)
        return torch.add(shortcut, inputs)
