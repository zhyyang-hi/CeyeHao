#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   basic_ops.py
@Desc    :   Adapted from https://github.com/zhengyang-wang/GVTNets
'''

import torch.nn as nn
from config.gvtn_cfg import conf_basic_ops


class Conv2d_gvtn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bias=True):
        super(Conv2d_gvtn, self).__init__()
        # compute same padding
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            bias=use_bias,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv2dTrans_gvtn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bias=True):
        super(Conv2dTrans_gvtn, self).__init__()
        # compute same padding
        padding = (kernel_size - 1) // 2
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            output_padding=1,
            bias=use_bias,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class BatchNorm_gvtn(nn.Module):
    def __init__(self, num_features, conf_basic_ops=conf_basic_ops):
        super(BatchNorm_gvtn, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features,
            momentum=conf_basic_ops["momentum"],
            eps=conf_basic_ops["epsilon"],
        )

    def forward(self, x):
        x = self.bn(x)
        return x


class ReLU_gvtn(nn.Module):
    def __init__(self, conf_basic_ops=conf_basic_ops):
        super(ReLU_gvtn, self).__init__()
        self.relu = nn.ReLU() if conf_basic_ops["relu_type"] == "relu" else nn.ReLU6()

    def forward(self, x):
        x = self.relu(x)
        return x
