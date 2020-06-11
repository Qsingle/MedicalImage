#-*- coding:utf8 -*-
#!/usr/bin/env python
'''
@Author:qiuzhongxi
@Filename:utils.py
@Date:2020/1/31
@Software:PyCharm

Some ops for unet and unet3d
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, strides=1, padding=0,bias=False):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, stride=strides, padding=padding,bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ksize, stride=strides,padding=padding,bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        net = self.conv(x)
        return net

class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride=1, padding=0, dilation=1,bias=True,activation=True):
        super(BasicConv,self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = activation

    def forward(self, x):
        net = self.conv(x)
        net = self.bn(net)
        if self.activation:
            net = F.leaky_relu(net,inplace=True)
        return net

class BasicConv3d(nn.Module):
    def __init__(self,in_ch, out_ch, ksize, stride, padding=0, dilation=1,activation=F.relu,bias=False):
        super(BasicConv3d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm3d(out_ch)
        )
        self.activation = activation

    def forward(self, x):
        net = self.net(x)
        if self.activation:
            net = self.activation(net, inplace=True)
        return net

class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1,padding=1,bias=False):
        super(DoubleConv3d, self).__init__()
        interch = out_ch if in_ch > out_ch else out_ch // 2
        self.net = nn.Sequential(
            BasicConv3d(in_ch, interch, ksize, stride, padding=padding, bias=bias),
            BasicConv3d(interch, out_ch, ksize, stride, padding=padding, bias=bias)
        )

    def forward(self, x):
        net = self.net(x)
        return net

class BasicBlock(nn.Module):
    def __init__(self,in_ch, ch, stride=1, downsample=False):
        super(BasicBlock,self).__init__()
        self.conv1 = BasicConv(in_ch, ch, 3, 1, padding=1,bias=False)
        self.conv2 = BasicConv(ch, ch, 3, stride, padding=1,bias=False)
        self.down = None

        if downsample:
            self.down = BasicConv(in_ch, ch, 1, stride,activation=False,bias=False)

    def forward(self,x):
        net = self.conv1(x)
        net = self.conv2(net)
        if self.down:
            x = self.down(x)
        net = x + net
        return net

class BottleBlock(nn.Module):
    def __init__(self,in_ch, ch, stride=1, downsample=False):
        super(BottleBlock,self).__init__()
        self.conv1 = BasicConv(in_ch, ch, 1, 1,bias=False)
        self.conv2 = BasicConv(ch, ch, 3, stride, padding=1,bias=False)
        self.conv3 = BasicConv(ch, ch*4, 1, 1,bias=False)
        self.down = None
        if downsample:
            self.down = BasicConv(in_ch, ch*4, 1, stride, activation=False,bias=False)

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        if self.down:
            x = self.down(x)
        net = x + net
        return net

def get_layers(n):
    n_layers = []
    if n == 18:
        n_layers = [2, 2, 2, 2]
    elif n == 32:
        n_layers = [3, 4, 6, 3]
    elif n == 50:
        n_layers = [3, 4, 6, 3]
    elif n == 101:
        n_layers = [3, 4, 23, 3]
    elif n == 152:
        n_layers = [3, 8, 36, 3]

    return n_layers

class hswish(nn.Module):
    def __init__(self):
        super(hswish, self).__init__()

    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def __init__(self):
        super(hsigmoid, self).__init__()

    def forward(self,x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    '''
    expand conv
    depthwise conv
    pointwise conv
    '''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        '''

        :param kernel_size:The kernel size of depthwise conv
        :param in_size:input channels
        :param expand_size:expand channel
        :param out_size:out channel
        :param nolinear: no linear activation function
        :param semodule:SeModule of block
        :param stride:stride of the block
        '''
        super(Block, self).__init__()
        self.se = semodule
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        if self.se is not None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

if __name__ == "__main__":
    import numpy as np
    x = np.random.normal(0,1,[1, 3, 64, 64,64])
    x = torch.Tensor(x)
    model = DoubleConv3d(3, 64, 3, 1)
    out = model(x)
    print(out.shape)