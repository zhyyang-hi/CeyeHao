#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gvtn_loss.py
@Desc    :   Adapted from https://github.com/zhengyang-wang/GVTNets
'''

import torch
import torch.nn.functional as F
from .basic_ops import *


class LossGvtn:
    def __init__(self, loss_type, probabilistic, offset) -> None:
        self.loss_type = loss_type
        self.probabilistic = probabilistic
        self.offset = offset
        if self.probabilistic:
            self.conv = Conv2d_gvtn(1, 1, 1, 1, False)
            self.compute_sigma = lambda x: F.softplus(self.conv(x)) + 1e-3
            if self.loss_type == "MSE":
                self.loss = lambda pred, label, sigma: torch.mean(
                    torch.div(torch.abs(pred - label), 2 * sigma**2) + torch.log(sigma)
                )
            elif self.loss_type == "MAE":
                self.loss = lambda pred, label, sigma: torch.mean(
                    torch.div(torch.abs(pred - label), sigma) + torch.log(sigma)
                )
        elif self.loss_type == "MSE":
            self.loss = torch.nn.MSELoss()
        elif self.loss_type == "MAE":
            self.loss = torch.nn.L1Loss()

    def __call__(self, outputs, label):
        penult, pred = outputs
        if self.probabilistic:
            sigma = self.compute_sigma(penult)
            return self.loss(pred, label, sigma)
        else:
            return self.loss(pred, label)
