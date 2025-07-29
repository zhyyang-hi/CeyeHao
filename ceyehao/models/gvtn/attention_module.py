#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   attention_module.py
@Desc    :   Adapted from https://github.com/zhengyang-wang/GVTNets
'''

import torch
import torch.nn as nn

from .basic_ops import *
from .attention_layer import Attention_gvtn
from config.gvtn_cfg import conf_attn_same, conf_attn_up, conf_attn_down


class SameGTO(nn.Module):
    def __init__(self, in_channels, output_filters, conf_attn_same=conf_attn_same):
        """Same GTO block.

        Args:
        inputs: a Tensor with shape [batch, (d,) h, w, channels]
        output_filters: an integer
        training: a boolean for batch normalization and dropout
        dimension: a string, dimension of inputs/outputs -- 2D, 3D
        name: a string
        Returns:
        A Tensor of shape [batch, (_d,) _h, _w, output_filters]
        """

        super(SameGTO, self).__init__()
        self.bn1 = BatchNorm_gvtn(in_channels)
        self.relu1 = ReLU_gvtn()
        self.attn = Attention_gvtn(
            in_channels,
            output_filters // conf_attn_same["key_ratio"],
            output_filters // conf_attn_same["value_ratio"],
            output_filters,
            conf_attn_same["num_heads"],
            "SAME",
            conf_attn_same["dropout_rate"],
            conf_attn_same["use_softmax"],
            conf_attn_same["use_bias"],
        )

    def forward(self, inputs):
        shortcut = inputs
        inputs = self.bn1(inputs)
        inputs = self.relu1(inputs)
        inputs, _ = self.attn(inputs)
        return torch.add(inputs, shortcut)


class UpGTOv1(nn.Module):
    def __init__(self, in_channels, output_filters, conf_attn_up=conf_attn_up) -> None:
        super().__init__()
        self.bn1 = BatchNorm_gvtn(in_channels)
        self.relu = ReLU_gvtn()
        self.projection_shortcut = Conv2dTrans_gvtn(
            in_channels, output_filters, 3, 2, False
        )
        self.attn = Attention_gvtn(
            in_channels,
            output_filters // conf_attn_up["key_ratio"],
            output_filters // conf_attn_up["value_ratio"],
            output_filters,
            conf_attn_up["num_heads"],
            "UP",
            conf_attn_up["dropout_rate"],
            conf_attn_up["use_softmax"],
            conf_attn_up["use_bias"],
        )

    def forward(self, inputs):
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        shortcut = self.projection_shortcut(inputs)
        inputs, _ = self.attn(inputs)

        return torch.add(shortcut, inputs)


class DownGTOv1(nn.Module):
    def __init__(
        self, in_channels, output_filters, conf_attn_down=conf_attn_down
    ) -> None:
        super().__init__()
        self.bn1 = BatchNorm_gvtn(in_channels)
        self.relu = ReLU_gvtn()
        self.projection_shortcut = Conv2d_gvtn(in_channels, output_filters, 3, 2, False)
        self.attn = Attention_gvtn(
            in_channels,
            output_filters // conf_attn_down["key_ratio"],
            output_filters // conf_attn_down["value_ratio"],
            output_filters,
            conf_attn_down["num_heads"],
            "DOWN",
            conf_attn_down["dropout_rate"],
            conf_attn_down["use_softmax"],
            conf_attn_down["use_bias"],
        )

    def forward(self, inputs):
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        shortcut = self.projection_shortcut(inputs)
        inputs, _ = self.attn(inputs)

        return torch.add(shortcut, inputs)


class UpGTOv2(nn.Module):
    def __init__(self, in_channels, output_filters, conf_attn_up=conf_attn_up) -> None:
        super().__init__()
        if conf_attn_up["key_ratio"] != 1:
            raise ValueError("Must set key_ratio == 1!")
        self.bn1 = BatchNorm_gvtn(in_channels)
        self.relu = ReLU_gvtn()
        self.attn = Attention_gvtn(
            in_channels,
            output_filters // conf_attn_up["key_ratio"],
            output_filters // conf_attn_up["value_ratio"],
            output_filters,
            conf_attn_up["num_heads"],
            "UP",
            conf_attn_up["dropout_rate"],
            conf_attn_up["use_softmax"],
            conf_attn_up["use_bias"],
        )

    def forward(self, inputs):
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        inputs, query = self.attn(inputs)

        return torch.add(query, inputs)


class DownGTOv2(nn.Module):
    def __init__(
        self, in_channels, output_filters, conf_attn_down=conf_attn_down
    ) -> None:
        super().__init__()
        if conf_attn_down["key_ratio"] != 1:
            raise ValueError("Must set key_ratio == 1!")
        self.bn1 = BatchNorm_gvtn(in_channels)
        self.relu = ReLU_gvtn()
        self.attn = Attention_gvtn(
            in_channels,
            output_filters // conf_attn_down["key_ratio"],
            output_filters // conf_attn_down["value_ratio"],
            output_filters,
            conf_attn_down["num_heads"],
            "DOWN",
            conf_attn_down["dropout_rate"],
            conf_attn_down["use_softmax"],
            conf_attn_down["use_bias"],
        )

    def forward(self, inputs):
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        inputs, query = self.attn(inputs)

        return torch.add(query, inputs)
