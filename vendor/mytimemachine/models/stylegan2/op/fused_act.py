"""Portable inference implementation of rosinality's fused activation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=math.sqrt(2)):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=math.sqrt(2)):
    view_shape = (1, bias.shape[0]) + (1,) * (input.ndim - 2)
    return F.leaky_relu(input + bias.view(view_shape), negative_slope) * scale
