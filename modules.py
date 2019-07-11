import functools
import numpy as np

import torch
import torch.nn as nn


LEAKY_FACTOR = 0.2
MULT_FACTOR = 1


# TEST PASSED
class PixelUnShuffle(nn.Module):
    """
    Inverse process of pytorch pixel shuffle module
    """
    def __init__(self, down_scale):
        """
        :param down_scale: int, down scale factor
        """
        super(PixelUnShuffle, self).__init__()

        if not isinstance(down_scale, int):
            raise ValueError('Down scale factor must be a integer number')
        self.down_scale = down_scale

    def forward(self, input):
        """
        :param input: tensor of shape (batch size, channels, height, width)
        :return: tensor of shape(batch size, channels * down_scale * down_scale, height / down_scale, width / down_scale)
        """
        b, c, h, w = input.size()
        assert h % self.down_scale == 0
        assert w % self.down_scale == 0

        oc = c * self.down_scale ** 2
        oh = int(h / self.down_scale)
        ow = int(w / self.down_scale)

        output_reshaped = input.reshape(b, c, oh, self.down_scale, ow, self.down_scale)
        output = output_reshaped.permute(0, 1, 3, 5, 2, 4).reshape(b, oc, oh, ow)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Sequential(
            PixelUnShuffle(scale),
            nn.Conv2d(input_channels * (scale ** 2), output_channels, kernel_size=ksize, stride=1, padding=ksize//2)
        )

    def forward(self, input):
        return self.downsample(input)


class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels * (scale ** 2), kernel_size=1, stride=1, padding=ksize//2),
            nn.PixelShuffle(scale)
        )

    def forward(self, input):
        return self.upsample(input)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, channels, ksize=3,
                 use_instance_norm=False, affine=False):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.ksize = ksize
        padding = self.ksize // 2
        if use_instance_norm:
            self.transform = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(input_channels, channels, kernel_size=self.ksize, stride=1),
                nn.InstanceNorm2d(channels, affine=affine),
                nn.LeakyReLU(0.2),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size=self.ksize, stride=1),
                nn.InstanceNorm2d(channels)
            )
        else:
            self.transform = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(input_channels, channels, kernel_size=self.ksize, stride=1),
                nn.LeakyReLU(0.2),
                nn.ReflectionPad2d(padding),
                nn.Conv2d(channels, channels, kernel_size=self.ksize, stride=1),
            )

    def forward(self, input):
        return input + self.transform(input) * MULT_FACTOR


class NormalizeBySum(nn.Module):
    def forward(self, x):
        return x / torch.sum(x, dim=1, keepdim=True).clamp(min=1e-7)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class DSN(nn.Module):
    def __init__(self, k_size, input_channels=3, scale=4):
        super(DSN, self).__init__()

        self.k_size = k_size

        self.sub_mean = MeanShift(1)

        self.ds_1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(input_channels, 64, 5),
            nn.LeakyReLU(LEAKY_FACTOR)
        )

        self.ds_2 = DownsampleBlock(2, 64, 128, ksize=1)
        self.ds_4 = DownsampleBlock(2, 128, 128, ksize=1)

        res_4 = list()
        for idx in range(5):
            res_4 += [ResidualBlock(128, 128)]
        self.res_4 = nn.Sequential(*res_4)

        self.ds_8 = DownsampleBlock(2, 128, 256)

        self.kernels_trunk = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            UpsampleBlock(8 // scale, 256, 256, ksize=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU()
        )

        self.kernels_weight = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, k_size ** 2, 3)
        )

        self.offsets_trunk = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            UpsampleBlock(8 // scale, 256, 256, ksize=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU()
        )

        self.offsets_h_generation = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, k_size ** 2, 3),
            nn.Tanh()
        )

        self.offsets_v_generation = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, k_size ** 2, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.ds_1(x)
        x = self.ds_2(x)
        x = self.ds_4(x)
        x = x + self.res_4(x)
        x = self.ds_8(x)

        kt = self.kernels_trunk(x)
        k_weight = torch.clamp(self.kernels_weight(kt), min=1e-6, max=1)
        kernels = k_weight / torch.sum(k_weight, dim=1, keepdim=True).clamp(min=1e-6)

        ot = self.offsets_trunk(x)
        offsets_h = self.offsets_h_generation(ot)
        offsets_v = self.offsets_v_generation(ot)

        return kernels, offsets_h, offsets_v
