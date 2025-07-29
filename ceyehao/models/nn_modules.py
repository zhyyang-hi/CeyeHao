import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        padding_mode="zeros",
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                bias=bias,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class StackConv(nn.Module):
    """[Convblock or Resblock] * num_conv"""

    def __init__(self, in_channels, out_channels, conv_kwargs):
        super().__init__()
        block_type = conv_kwargs["block_type"]
        num_conv = conv_kwargs["num_conv"]
        kernel_size = np.array(conv_kwargs["kernel_size"])
        dilation = np.array(conv_kwargs["dilation"])
        padding = dilation * (kernel_size - 1) // 2
        if block_type == "res":
            assert num_conv % 2 == 0, "num_conv must be even for resblock"
            self.stack_conv = nn.Sequential(
                ResBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size[0:2],
                    padding=padding[0:2],
                    dilation=dilation[0:2],
                    bias=False,
                ),
                *[
                    ResBlock(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size[2 * (i + 1) : 2 * (i + 1) + 2],
                        padding=padding[2 * (i + 1) : 2 * (i + 1) + 2],
                        dilation=dilation[2 * (i + 1) : 2 * (i + 1) + 2],
                        bias=False,
                    )
                    for i in range(num_conv // 2 - 1)
                ],
            )
        elif block_type == "conv":
            self.stack_conv = nn.Sequential(
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size[0],
                    padding=padding[0],
                    dilation=dilation[0],
                    bias=False,
                ),
                *[
                    ConvBlock(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size[i + 1],
                        padding=padding[i + 1],
                        dilation=dilation[i + 1],
                        bias=False,
                    )
                    for i in range(num_conv - 1)
                ],
            )

    def forward(self, x):
        return self.stack_conv(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.projection_shortcut = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=1,
            bias=bias,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, inputs):
        if self.in_channels == self.out_channels:
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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), StackConv(in_channels, out_channels, conv_kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then stack conv"""

    def __init__(self, in_channels, out_channels, conv_kwargs):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = StackConv(in_channels, out_channels, conv_kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CESP(nn.Module):
    """Compound Eye Skip Pathway.
    for conv pool, the channels must be specified.
    """

    def __init__(
        self, tile_factor, pool_type, tp_order, channels=None
    ):
        super().__init__()
        self.tile_factor = tile_factor
        self.tp_order = tp_order
        if pool_type == "avg":
            self.pool = nn.AvgPool2d(self.tile_factor)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(self.tile_factor)
        elif pool_type == "bilinear":
            self.pool = lambda x: F.interpolate(
                x,
                scale_factor=1 / self.tile_factor,
                mode="bilinear",
                antialias=True,
            )
        elif pool_type == "conv":
            assert channels is not None, "channels must be specified for conv pool"
            self.pool = nn.Conv2d(
                channels,
                channels,
                kernel_size=self.tile_factor,
                stride=self.tile_factor,
            )
        if self.tp_order == "pool_first":
            self.compeye = lambda x: self.tile(self.pool(x))
        elif self.tp_order == "tile_first":
            self.compeye = lambda x: self.pool(self.tile(x))

    def tile(self, x: torch.Tensor):
        return x.repeat(1, 1, self.tile_factor, self.tile_factor)

    def forward(self, x):
        size = x.size()
        x = self.compeye(x)
        # padding to the original size
        diffX = size[2] - x.size()[2]
        diffY = size[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x
