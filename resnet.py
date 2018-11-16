#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : resnet.py
# @Author: Piston Yang
# @Date  : 18-8-23
from mxnet.gluon import nn


class SubpixelConv2D(nn.HybridBlock):
    def __init__(self, in_channel, kernel_size, out_channel, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.out_channel = 4 * out_channel
        self.conv = nn.Conv2D(self.out_channel, kernel_size, in_channels=in_channel)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv(x)
        x = F.depth_to_space(x, 2)
        return x


class BottleneckResidualBlock(nn.HybridBlock):
    def __init__(self, in_channel, out_channel, resample=None, **kwargs):
        super(BottleneckResidualBlock, self).__init__(**kwargs)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.resample = resample
        with self.name_scope():
            if resample == 'down':
                self.conv_shortcut = nn.Conv2D(out_channel, kernel_size=1, strides=2, in_channels=in_channel)
                self.conv_1 = nn.Conv2D(in_channel // 2, kernel_size=1, in_channels=in_channel)
                self.conv_1b = nn.Conv2D(out_channel // 2, kernel_size=3, strides=2, padding=1,
                                         in_channels=in_channel // 2)
                self.conv_2 = nn.Conv2D(out_channel, kernel_size=1, use_bias=False, in_channels=out_channel // 2)

            elif resample == 'up':
                self.conv_shortcut = SubpixelConv2D(in_channel=in_channel, kernel_size=1, out_channel=out_channel)
                self.conv_1 = nn.Conv2D(in_channel // 2, kernel_size=1, in_channels=in_channel)
                self.conv_1b = nn.Conv2DTranspose(out_channel // 2, kernel_size=3, strides=2, padding=1,
                                                  output_padding=1, in_channels=in_channel // 2)
                self.conv_2 = nn.Conv2D(out_channel, kernel_size=1, use_bias=False, in_channels=out_channel // 2)

            elif resample is None:
                if in_channel != out_channel:
                    self.conv_shortcut = nn.Conv2D(out_channel, kernel_size=1, in_channels=in_channel)
                self.conv_1 = nn.Conv2D(in_channel // 2, kernel_size=1, in_channels=in_channel)
                self.conv_1b = nn.Conv2D(out_channel // 2, kernel_size=3, padding=1, in_channels=in_channel // 2)
                self.conv_2 = nn.Conv2D(out_channel, kernel_size=1, use_bias=False, in_channels=out_channel // 2)
            else:
                raise Exception('invalid resample value')

            self.bn = nn.BatchNorm()

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.in_channel == self.out_channel and self.resample is None:
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)

        x = F.Activation(x, act_type='relu')
        x = self.conv_1(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv_1b(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv_2(x)
        x = self.bn(x)

        return shortcut + 0.3 * x


class ResnetGenerator(nn.HybridBlock):
    def __init__(self, dim=64, **kwargs):
        super(ResnetGenerator, self).__init__(**kwargs)
        self.dim = dim
        with self.name_scope():
            self.pre_conv = nn.Conv2D(4 * 4 * 8 * dim, kernel_size=1)
            self.layer1 = self._make_layer(layers=6, channels=8, stage_index=1, resample='up')
            self.layer2 = self._make_layer(layers=6, channels=4, stage_index=2, resample='up')
            self.layer3 = self._make_layer(layers=6, channels=2, stage_index=3, resample='up')
            self.layer4 = self._make_layer(layers=6, channels=1, stage_index=4, resample='up')
            self.layer5 = self._make_layer(layers=5, channels=0.5, stage_index=5, use_resample=False)
            self.last_conv = nn.Conv2D(3, kernel_size=1, in_channels=self.dim // 2)

    def _make_layer(self, layers, channels, stage_index, resample=None, use_resample=True):
        ch_size = int(channels * self.dim)
        if channels >= 1:
            half_c = channels // 2 if channels // 2 > 0 else channels / 2
            half_size = int(half_c * self.dim)
        else:
            half_size = None

        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            for _ in range(layers):
                layer.add(BottleneckResidualBlock(ch_size, ch_size, resample=None))
            if use_resample:
                layer.add(BottleneckResidualBlock(ch_size, half_size, resample=resample))
        return layer

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.pre_conv(x)
        x = x.reshape((x.shape[0], 8 * self.dim, 4, 4))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.last_conv(x)
        x = F.Activation(x / 5., act_type='tanh')
        return x


class ResnetDiscriminator(nn.HybridBlock):
    def __init__(self, dim=64, **kwargs):
        super(ResnetDiscriminator, self).__init__(**kwargs)
        self.dim = dim
        with self.name_scope():
            self.pre_conv = nn.Conv2D(self.dim // 2, kernel_size=1)
            self.layer1 = self._make_layer(layers=5, channels=0.5, stage_index=1, resample='down')
            self.layer2 = self._make_layer(layers=6, channels=1, stage_index=2, resample='down')
            self.layer3 = self._make_layer(layers=6, channels=2, stage_index=3, resample='down')
            self.layer4 = self._make_layer(layers=6, channels=4, stage_index=4, resample='down')
            self.layer5 = self._make_layer(layers=6, channels=8, stage_index=5, use_resample=False)
            self.last_conv = nn.Conv2D(1, kernel_size=1)

    def _make_layer(self, layers, channels, stage_index, resample=None, use_resample=True):
        ch_size = int(channels * self.dim)
        double_size = ch_size * 2

        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            for _ in range(layers):
                layer.add(BottleneckResidualBlock(ch_size, ch_size, resample=None))
            if use_resample:
                layer.add(BottleneckResidualBlock(ch_size, double_size, resample=resample))
        return layer

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape((-1, 4 * 4 * 8 * self.dim, 1, 1))
        x = self.last_conv(x)
        x = F.flatten(x)
        return x
