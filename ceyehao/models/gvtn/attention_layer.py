#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   attention_layer.py
@Desc    :   Adapted from https://github.com/zhengyang-wang/GVTNets
'''

import torch
import torch.nn as nn

from .basic_ops import Conv2d_gvtn, Conv2dTrans_gvtn, BatchNorm_gvtn, ReLU_gvtn


class Attention_gvtn(nn.Module):
    def __init__(
        self,
        in_channels,
        total_key_filters,
        total_value_filters,
        output_filters,
        num_heads,
        layer_type,
        dropout_rate=0.0,
        use_softmax=True,
        use_bias=True,
    ) -> None:
        """Multihead scaled-dot-product attention with input/output transformations.

        Args:
                inputs: a Tensor with shape [batch, channels, h, w]
                total_key_filters: an integer. Note that queries have the same number
                        of channels as keys.
                total_value_filters: an integer
                output_filters: an integer
                num_heads: an integer dividing total_key_filters and total_value_filters
                training: a boolean for dropout
                dimension: a string, dimension of inputs/outputs -- 2D, 3D
                layer_type: a string, type of this layer -- SAME, DOWN, UP, UP4
                name: a string
                dropout_rate: a float between 0.0 and 1.0. No dropout if dropout_rate = 0.0
                use_softmax: a boolean deciding whether to use softmax. Note that use_softmax = False
                        will automatically set dropout_rate = 0.0
                use_bias: a boolean deciding whether to use the bias term in input/output transformations

        Returns:
                A Tensor of shape [batch, (_d,) _h, _w, output_filters]

        Raises:
                ValueError: if the total_key_filters or total_value_filters are not divisible
                        by the number of attention heads.
                ValueError: if dimension is not one of ['2D', '3D'].
                ValueError: if layer_type is not one of ['SAME', 'DOWN', 'UP'].
        """
        super().__init__()
        if total_key_filters % num_heads != 0:
            raise ValueError(
                "Key depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_key_filters, num_heads)
            )
        if total_value_filters % num_heads != 0:
            raise ValueError(
                "Value depth (%d) must be divisible by the number of "
                "attention heads (%d)." % (total_value_filters, num_heads)
            )
        if layer_type not in ["SAME", "DOWN", "UP"]:
            raise ValueError(
                "Layer type (%s) must be one of SAME, DOWN, UP, UP4." % (layer_type)
            )
        self.num_heads = num_heads
        self.total_key_filters = total_key_filters

        self.compute_qkv = ComputeQKV(
            layer_type, in_channels, total_key_filters, total_value_filters, use_bias
        )

        self.dot_product_attention = DotProductAttention(dropout_rate, use_softmax)
        self.output_transfrom = Conv2d_gvtn(
            output_filters, output_filters, 1, 1, use_bias
        )

    def forward(self, x):  # x: [batch, channels, h, w]
        q, k, v = self.compute_qkv(
            x
        )  # q: [batch, total_key_filters, _h, _w], k: [batch, total_key_filters, h, w], v: [batch, total_value_filters, h, w]
        q_split = split_heads(
            q, self.num_heads
        )  # q_split: [batch, num_heads, channels_qs, _h, _w]
        k_split = split_heads(
            k, self.num_heads
        )  # k_split: [batch, num_heads, channels_ks, h, w]
        v_split = split_heads(
            v, self.num_heads
        )  # v_split: [batch, num_heads, channels_vs, h, w]

        # Scale query to prevent the dot product between q and k from growing too large.
        depth = self.total_key_filters // self.num_heads  # channels_ks
        q_split *= depth**-0.5

        output_shape = v_split.shape[:-2] + q_split.shape[-2:]

        # Flatten extra dimensions
        q_new = unfold(q_split)  # q_new: [batch, num_heads, channels_ks, _h * _w]
        k_new = unfold(k_split)  # k_new: [batch, num_heads, channels_ks, h * w]
        v_new = unfold(v_split)  # v_new: [batch, num_heads, channels_vs, h * w]

        o = self.dot_product_attention(
            q_new, k_new, v_new
        )  # o : [batch, num_heads, channels_vs, len_q]

        # Reshape the output array.
        o = o.reshape(output_shape)  # o: [batch, num_heads, channels_vs, _h, _w]

        o = combine_heads(o)
        o = self.output_transfrom(o)
        return o, q


class ComputeQKV(nn.Module):
    def __init__(
        self,
        layer_type,
        in_channels,
        total_key_filters,
        total_value_filters,
        use_bias=True,
    ):
        """Perform qkv transformations and compute query, key and value.

        Args:
            inputs: a Tensor with shape [batch, channels, h, w]
            total_key_filters: an integer
            total_value_filters: an integer
            use_bias: a boolean deciding whether to use the bias term in qkv transformations
            layer_type: a string, type of this layer -- SAME, DOWN, UP

        Returns:
            q: a Tensor with shape [batch, total_key_filters, _h, _w]
            k: a Tensor with shape [batch, total_key_filters, h, w]
            v: a Tensor with shape [batch, total_value_filters, h, w]
        """
        super(ComputeQKV, self).__init__()

        if layer_type == "SAME":
            self.compute_q = Conv2d_gvtn(
                in_channels,
                total_key_filters,
                1,
                1,
                use_bias,
            )
        elif layer_type == "DOWN":
            self.compute_q = Conv2d_gvtn(
                in_channels,
                total_key_filters,
                3,
                2,
                use_bias,
            )
        elif layer_type == "UP":
            self.compute_q = Conv2dTrans_gvtn(
                in_channels,
                total_key_filters,
                3,
                2,
                use_bias,
            )

        self.compute_k = Conv2d_gvtn(
            in_channels,
            total_key_filters,
            1,
            1,
            use_bias,
        )
        self.compute_v = Conv2d_gvtn(
            in_channels,
            total_value_filters,
            1,
            1,
            use_bias,
        )

    def forward(self, x):
        q = self.compute_q(x)
        k = self.compute_k(x)
        v = self.compute_v(x)
        return q, k, v


class DotProductAttention(nn.Module):
    """Dot-product attention.
    Args:
        q: a Tensor with shape [batch, num_heads, channels_ks, len_q]
        k: a Tensor with shape [batch, num_heads, channels_ks, len_kv]
        v: a Tensor with shape [batch, num_heads, channels_vs, len_kv]
        dropout_rate: a float between 0.0 and 1.0. No dropout if dropout_rate = 0.0
        use_softmax: a boolean deciding whether to use softmax. Note that
        use_softmax = False will automatically set dropout_rate = 0.0
    Returns:
        A Tensor with shape [batch, heads, channels_vs, length_q]
    """

    def __init__(self, dropout_rate=0.0, use_softmax=True):
        super(DotProductAttention, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_softmax = use_softmax

        if not use_softmax:
            self.dropout_rate = 0.0
        else:
            self.dropout_rate = dropout_rate
            self.soft_max = nn.Softmax(dim=2)
            if self.dropout_rate > 0.0:
                self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v):
        if self.use_softmax:
            logits = torch.matmul(
                q.transpose(-2, -1), k
            )  # logits: [batch, num_heads, len_q, len_kv]
            weights = self.soft_max(logits)
            if self.dropout_rate > 0.0:
                weights = self.dropout(weights)
            return torch.matmul(
                v, weights.transpose(-2, -1)
            )  # output: [batch, num_heads, channels_vs, len_q]
        else:
            # to save computation, compute K^T * V first
            kv = torch.matmul(
                k, v.transpose(-2, -1)
            )  # kv: [batch, num_heads, channels_ks, channels_vs]

            # normalize kv
            kv = kv / torch.tensor(q.shape[-1], dtype=torch.float32)

            return torch.matmul(kv, q)  # output: [batch, num_heads, channels_vs, len_q]


def split_heads(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).

    Args:
        x: a Tensor with shape [batch, channels, h, w]
        num_heads: an integer

    Returns:
        A Tensor with shape [batch, num_heads, channels / num_heads, h, w, ]
    """
    batch, channels, h, w = x.shape
    ret = x.view(
        batch,
        num_heads,
        channels // num_heads,
        h,
        w,
    )
    return ret


def unfold(x):
    """Unfold the input tensor.

    Args:
        x: a Tensor with shape [batch, heads, channels per head,  h, w]

    Returns:
        A Tensor with shape [batch, heads, channels per head, h * w]
    """
    batch, heads, channels, h, w = x.shape
    ret = x.reshape(batch, heads, channels, h * w)
    return ret


def combine_heads(x):
    """Inverse of split_heads_2d.
    Args:
        x: a Tensor with shape [batch, num_heads, channels / num_heads, h, w]
    Returns:
        a Tensor with shape [batch, channels, h, w]
    """
    ret = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
    return ret
