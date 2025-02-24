#!/usr/bin/env python

import torch
import torch.nn as nn

from models.nn_modules import *


class UNet(nn.Module):
    def __init__(
        self,
        output_size,
        base_num_features,
        num_pool,
        down_kwargs,
        up_kwargs,
    ):
        super(UNet, self).__init__()

        self.output_size = output_size
        self.base_num_features = base_num_features
        self.num_pool = num_pool
        self.down_kwargs = down_kwargs
        self.up_kwargs = up_kwargs

        self.inc = StackConv(1, base_num_features, down_kwargs)
        self.down = nn.ModuleList(
            [
                Down(
                    base_num_features * (2**i),
                    base_num_features * (2 ** (i + 1)),
                    down_kwargs,
                )
                for i in range(num_pool)
            ]
        )

        self.up = nn.ModuleList(
            [
                Up(
                    base_num_features * (2**i),
                    base_num_features * (2 ** (i - 1)),
                    up_kwargs,
                )
                for i in range(num_pool, 0, -1)
            ]
        )

        self.outc = OutConv(base_num_features, 2)
        if output_size == [200, 200]:
            self.align = nn.Conv2d(
                2, 2, kernel_size=7, stride=1, padding=3
            )  # out 200*200.
        elif output_size == [100, 100]:
            self.align = nn.Conv2d(
                2, 2, kernel_size=5, stride=2, padding=2
            )  # out 100*100
        elif output_size == [50, 50]:
            self.align = torch.nn.Sequential(
                nn.Conv2d(2, 2, kernel_size=5, stride=2, padding=2),  # out 100*100
                nn.Conv2d(2, 2, kernel_size=3, stride=2, padding=1),  # out 50*50
            )
        else:
            raise ValueError(f"output resolution is {output_size}, and not supported!")

    def forward(self, x):
        x_down = []
        x_down.append(self.inc(x))
        for down in self.down:
            x_down.append(down(x_down[-1]))

        x = x_down[-1]
        for i, up in enumerate(self.up):
            x = up(x, x_down[-2 - i])

        x = self.outc(x)
        x = self.align(x)

        return x


class CEyeNet(UNet):
    def __init__(
        self,
        output_size,
        base_num_features,
        num_pool,
        down_kwargs,
        up_kwargs,
        compeye_kwargs,
    ):
        super(CEyeNet, self).__init__(
            output_size=output_size,
            base_num_features=base_num_features,
            num_pool=num_pool,
            down_kwargs=down_kwargs,
            up_kwargs=up_kwargs,
        )
        self.compeye_kwargs = compeye_kwargs

        if compeye_kwargs["pool_type"] == "conv":
            chs = [base_num_features * (2**i) for i in range(num_pool)]
            self.cesp = nn.ModuleList(
                [CESP(channels=ch, **compeye_kwargs) for ch in chs]
            )
        else:
            self.cesp = CESP(**compeye_kwargs)

    def forward(self, x):
        x_down = []
        x_down.append(self.inc(x))
        for down in self.down:
            x_down.append(down(x_down[-1]))

        x_tp = []
        if self.compeye_kwargs["pool_type"] == "conv":
            for i, xd in enumerate(x_down[:-1]):
                x_tp.append(self.cesp[i](xd))
        else:
            for xd in x_down[:-1]:
                x_tp.append(self.cesp(xd))

        x = x_down[-1]
        for i, up in enumerate(self.up):
            x = up(x, x_tp[-1 - i])

        x = self.outc(x)
        x = self.align(x)

        return x


class AsymUNet(UNet):
    """Unet with different encoder and decoder conv blocks."""

    def __init__(
        self,
        output_size=[200, 200],
        base_num_features=64,
        num_pool=4,
        num_conv=2,
        kernel_size=[3, 3],
        dilation=[1, 1],
        block_type="conv",
        up_dilation=[3, 1],
        up_num_conv=2,
        up_kernel_size=[3, 3],
        up_block_type="conv",
    ):
        super().__init__(
            output_size,
            base_num_features,
            num_pool,
            num_conv,
            kernel_size,
            dilation,
            block_type,
        )
        self.up_conv_kwargs = {
            "num_conv": up_num_conv,
            "kernel_size": up_kernel_size,
            "dilation": up_dilation,
            "block_type": up_block_type,
        }
        self.up = nn.ModuleList(
            [
                Up(
                    base_num_features * (2**i),
                    base_num_features * (2 ** (i - 1)),
                    self.up_conv_kwargs,
                )
                for i in range(num_pool, 0, -1)
            ]
        )
