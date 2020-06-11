#-*- coding:utf8 -*-
#!/usr/bin/env python
'''
@Author:qiuzhongxi
@Filename:unet.py
@Date:2020/1/31
@Software:PyCharm

Implementation of Unet and Unet3D
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F


from .utils import BasicConv,DoubleConv
from .utils import DoubleConv3d, BasicConv3d
from .utils import BasicBlock,BottleBlock,get_layers
from .utils import Block,hswish,SeModule

class Unet(nn.Module):
    def __init__(self, in_ch, num_classes=2):
        super(Unet,self).__init__()
        self.conv1 = BasicConv(in_ch, 64, 3, stride=1, padding=1, bias=False)
        self.conv1_after = DoubleConv(64, 64, 3, strides=1, padding=1)
        self.down = nn.MaxPool2d(2, 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)

        self.down2_conv = DoubleConv(64, 128, 3, strides=1, padding=1)
        self.down3_conv = DoubleConv(128, 256, 3, strides=1, padding=1)
        self.down4_conv = DoubleConv(256, 512, 3, strides=1, padding=1)
        self.down5_conv = DoubleConv(512, 1024, 3, strides=1, padding=1)

        self.up6_conv = BasicConv(1024, 512, 1, stride=1,bias=False)
        self.up6_conv_after = DoubleConv(1024, 512, 3, strides=1, padding=1)
        self.up7_conv = BasicConv(512, 256, 1, stride=1, bias=False)
        self.up7_conv_after = DoubleConv(512, 256, 3, strides=1, padding=1)
        self.up8_conv = BasicConv(256, 128, 1, stride=1, bias=False)
        self.up8_conv_after = DoubleConv(256, 128, 3, strides=1, padding=1)
        self.up9_conv = BasicConv(128, 64, 1, stride=1, bias=False)
        self.up9_conv_after = DoubleConv(128, 64, 3, strides=1, padding=1)
        self.out_conv = BasicConv(64, num_classes, 1, 1, bias=False)

        self.num_classes = num_classes
    def forward(self, x):
        conv1 = self.conv1(x)
        down1_0 = self.conv1_after(conv1)
        down1 = self.down(down1_0)
        down2_0 = self.down2_conv(down1)
        down2 = self.down(down2_0)
        down3_0 = self.down3_conv(down2)
        down3 = self.down(down3_0)
        down4_0 = self.down4_conv(down3)
        down4 = self.down(down4_0)
        down5 = self.down5_conv(down4)
        up1 = self.up(down5)
        up1_conv = self.up6_conv(up1)
        up1_merge = torch.cat([down4_0, up1_conv],dim=1)
        up1_after = self.up6_conv_after(up1_merge)
        up2 = self.up(up1_after)
        up2_conv = self.up7_conv(up2)
        up2_merge = torch.cat([down3_0, up2_conv],dim=1)
        up2_after = self.up7_conv_after(up2_merge)
        up3 = self.up(up2_after)
        up3_conv = self.up8_conv(up3)
        up3_merge = torch.cat([down2_0, up3_conv],dim=1)
        up3_after = self.up8_conv_after(up3_merge)
        up4 = self.up(up3_after)
        up4_conv = self.up9_conv(up4)
        up4_merge = torch.cat([down1_0, up4_conv], dim=1)
        up4_after = self.up9_conv_after(up4_merge)
        out = self.out_conv(up4_after)

        return out

class Unet3d(nn.Module):
    def __init__(self, in_ch, num_classes=2):
        super(Unet3d,self).__init__()
        self.down = nn.MaxPool3d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.down1_0 = DoubleConv3d(in_ch, 64)
        self.down2_0 = DoubleConv3d(64, 128)
        self.down3_0 = DoubleConv3d(128, 256)
        self.down4 = DoubleConv3d(256, 512)
        self.up1 = DoubleConv3d(768, 256)
        self.up2 = DoubleConv3d(384, 128)
        self.up3 = DoubleConv3d(192, 64)
        self.out_conv = BasicConv3d(64, num_classes, ksize=1, stride=1)

    def forward(self, x):
        down1_0 = self.down1_0(x)
        down1 = self.down(down1_0)
        down2_0 = self.down2_0(down1)
        down2 = self.down(down2_0)
        down3_0 = self.down3_0(down2)
        down3 = self.down(down3_0)
        down4 = self.down4(down3)
        up1 = self.up(down4)
        merge1 = torch.cat([down3_0, up1], dim=1)
        up1_conv = self.up1(merge1)
        up2 = self.up(up1_conv)
        merge2 = torch.cat([down2_0, up2], dim=1)
        up2_conv = self.up2(merge2)
        up3 = self.up(up2_conv)
        merge3 = torch.cat([down1_0, up3], dim=1)
        up3_conv = self.up3(merge3)
        out = self.out_conv(up3_conv)
        return out


class ResUnet(nn.Module):
    def __init__(self, in_ch, num_classes, n_layers=18):
        super(ResUnet, self).__init__()
        ch = 64
        # self.conv1 = BasicConv(in_ch, ch, 3, 1, padding=1, bias=False)
        # self.conv2 = BasicConv(ch, ch, 3, 2, padding=1, bias=False)
        # self.conv3 = BasicConv(ch, ch, 3, 1, padding=1, bias=False)
        self.conv1 = BasicConv(in_ch, ch, 7, 1, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)

        self.n_blocks = get_layers(n_layers)
        if n_layers < 50:
            block = BasicBlock
        else:
            block = BottleBlock

        self.block1 = block(ch, ch, 1, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block1_x = block(in_cha, ch, 1)

        ch *= 2
        self.block2 = block(in_cha, ch, 2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block2_x = block(in_cha, ch, 1)

        ch *= 2
        self.block3 = block(in_cha, ch, 2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block3_x = block(in_cha, ch, 1)
        ch *= 2
        self.block4 = block(in_cha, ch, 2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block4_x = block(in_cha, ch, 1)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        #self.init_params()
        self.up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
        if n_layers < 50:
            self.center_conv = BasicConv(in_cha, 512, 1, stride=1, padding=0, bias=False)
            self.center_after = DoubleConv(512+256, 512, 3, strides=1, padding=1)
            self.up6_conv = BasicConv(512, 256, 1, stride=1, padding=0, bias=False)
            self.merge6_conv = DoubleConv(256+128, 256, 3, strides=1, padding=1)
            self.up7_conv = BasicConv(256, 128, 1, stride=1, padding=0, bias=False)
            self.merge7_conv = DoubleConv(128+64, 128, 3, strides=1, padding=1)
            self.up8_conv = BasicConv(128, 64, 1, stride=1, padding=0,bias=False)
            self.merge8_conv = DoubleConv(128, 64, 3, strides=1, padding=1, bias=False)
        else:
            self.center_conv = BasicConv(in_cha, 1024, 1, stride=1, padding=0, bias=False)
            self.center_after = DoubleConv(2048, 512,3, strides=1, padding=1)
            self.up6_conv = BasicConv(512, 256, 1, stride=1, padding=0, bias=False)
            self.merge6_conv = DoubleConv(512+256, 256, 3, strides=1, padding=1)
            self.up7_conv = BasicConv(256, 128, 1, stride=1, padding=0, bias=False)
            self.merge7_conv = DoubleConv(256+128, 64, 3, strides=1, padding=1)
            self.up8_conv = BasicConv(64, 64, 1, stride=1, padding=0, bias=False)
            self.merge8_conv = DoubleConv(128, 64, 3, strides=1, padding=1)


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        #first = self.conv3(self.conv2(self.conv1(x)))
        first = self.conv1(x)
        pool1 = self.pool1(first)
        down1_0 = self.block1(pool1)
        for i in range(1, self.n_blocks[0] - 1):
            down1_0 = self.block1_x(down1_0)
        down1 = self.block2(down1_0)
        down2_0 = self.block2_x(down1)
        for i in range(1, self.n_blocks[1] - 1):
            down2_0 =self.block2_x(down2_0)
        down2 = self.block3(down2_0)
        down3_0 = self.block3_x(down2)
        for i in range(1, self.n_blocks[2]-1):
            down3_0 = self.block3_x(down3_0)
        down3 = self.block4(down3_0)
        down4 = self.block4_x(down3)
        for i in range(1, self.n_blocks[3]-1):
            down4 = self.block4_x(down4)
        center = self.up(down4)
        center_conv = self.center_conv(center)
        merge = torch.cat([down3_0, center_conv], dim=1)
        center_after = self.center_after(merge)
        up6 = self.up(center_after)
        up6_conv = self.up6_conv(up6)
        merge6 = torch.cat([down2_0, up6_conv], dim=1)
        merge6_conv = self.merge6_conv(merge6)
        up7 = self.up(merge6_conv)
        up7_conv = self.up7_conv(up7)
        merge7 = torch.cat([down1_0, up7_conv], dim=1)
        merge7_conv = self.merge7_conv(merge7)
        up8 = self.up(merge7_conv)
        up8_conv = self.up8_conv(up8)
        merge8 = torch.cat([first, up8_conv], dim=1)
        merge8_conv = self.merge8_conv(merge8)
        out = self.out_conv(merge8_conv)
        return out

class MobileNetv3_Large_Unet(nn.Module):
    def __init__(self, in_ch, num_classes=3):
        super(MobileNetv3_Large_Unet, self).__init__()
        self.down1_0 = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1)
        )
        self.down1 = Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2)
        self.down2_0 = Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1)
        self.down2 = Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(72), 2)
        self.down3_0 = nn.Sequential(
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(120), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(120), 1)
        )
        self.down3 = Block(3, 40, 240, 80, hswish(), None, 2)
        self.down4_0 = nn.Sequential(
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(480), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(672), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(672), 1)
        )
        self.down4 = Block(5, 160, 672, 160, hswish(), SeModule(672), 2)
        self.down5 = nn.Sequential(
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),
            nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(960),
            hswish()
        )
        #self.init_params()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.up6_conv = BasicConv(960, 160, ksize=1, stride=1, padding=0)
        self.merge6_conv = DoubleConv(320, 512, 3, strides=1, padding=1)
        self.up7_conv = BasicConv(512, 40, ksize=1, stride=1, padding=0)
        self.merge7_conv = DoubleConv(80, 256, 3, strides=1, padding=1)
        self.up8_conv = BasicConv(256, 24, ksize=1, stride=1, padding=0)
        self.merge8_conv = DoubleConv(48, 128, 3, strides=1, padding=1)
        self.up9_conv = BasicConv(128, 16, ksize=1, stride=1, padding=0)
        self.merge9_conv = DoubleConv(32, 64, 3, strides=1, padding=1)
        self.up10_conv = DoubleConv(64, 64, 3, strides=1, padding=1)
        self.out_conv = BasicConv(64, num_classes, ksize=1, stride=1, padding=0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self,x):
        down1_0 = self.down1_0(x)
        down1 = self.down1(down1_0)
        down2_0 = self.down2_0(down1)
        down2 = self.down2(down2_0)
        down3_0 = self.down3_0(down2)
        down3 = self.down3(down3_0)
        down4_0 = self.down4_0(down3)
        down4 = self.down4(down4_0)
        down5 = self.down5(down4)
        up6 = self.up(down5)
        up6_conv = self.up6_conv(up6)
        merge6 = torch.cat([down4_0,up6_conv], dim=1)
        merge6_conv = self.merge6_conv(merge6)
        up7 = self.up(merge6_conv)
        up7_conv =self.up7_conv(up7)
        merge7 = torch.cat([down3_0, up7_conv], dim=1)
        merge7_conv = self.merge7_conv(merge7)
        up8 = self.up(merge7_conv)
        up8_conv = self.up8_conv(up8)
        merge8 = torch.cat([down2_0, up8_conv], dim=1)
        merge8_conv = self.merge8_conv(merge8)
        up9 = self.up(merge8_conv)
        up9_conv = self.up9_conv(up9)
        merge9 = torch.cat([down1_0, up9_conv], dim=1)
        merge9_conv = self.merge9_conv(merge9)
        up10_conv = self.up10_conv(self.up(merge9_conv))
        out = self.out_conv(up10_conv)
        return out

class MobileNetv3_Small_Unet(nn.Module):
    def __init__(self,in_ch, num_classes=4):
        super(MobileNetv3_Small_Unet, self).__init__()
        self.down1_0 = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )
        self.down1 = Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2)
        self.down2_0 = Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2)
        self.down2 = Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1)
        self.down3_0 = Block(5, 24, 96, 40, hswish(), SeModule(96), 1)
        self.down3 = Block(5, 40, 240, 40, hswish(), SeModule(240), 2)
        self.down4_0 = nn.Sequential(
            Block(5, 40, 240, 40, hswish(), SeModule(240), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(240), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(120), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(144), 1)
        )
        self.down4 = Block(5, 48, 288, 96, hswish(), SeModule(288), 2)
        self.down5 = nn.Sequential(
            Block(5, 96, 576, 96, hswish(), SeModule(576), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(576), 1),
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(576),
            hswish()
        )
        self.init_params()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.up6_conv = BasicConv(576, 48, ksize=1, stride=1, padding=0)
        self.merge6_conv = DoubleConv(96, 512, 3, strides=1, padding=1)
        self.up7_conv = BasicConv(512, 40, ksize=1, stride=1, padding=0)
        self.merge7_conv = DoubleConv(80, 256, 3, strides=1, padding=1)
        self.up8_conv = BasicConv(256, 24, ksize=1, stride=1, padding=0)
        self.merge8_conv = DoubleConv(40, 128, 3, strides=1, padding=1)
        self.up9_conv = BasicConv(128, 16, ksize=1, stride=1, padding=0)
        self.merge9_conv = DoubleConv(32, 64, 3, strides=1, padding=1)
        self.up10_conv = DoubleConv(64, 64, 3, strides=1, padding=1)
        self.out_conv = BasicConv(64, num_classes, ksize=1, stride=1, padding=0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        down1_0 = self.down1_0(x)
        down1 = self.down1(down1_0)
        down2_0 = self.down2_0(down1)
        down2 = self.down2(down2_0)
        down3_0 = self.down3_0(down2)
        down3 = self.down3(down3_0)
        down4_0 = self.down4_0(down3)
        down4 = self.down4(down4_0)
        down5 = self.down5(down4)
        up6 = self.up(down5)
        up6_conv = self.up6_conv(up6)
        merge6 = torch.cat([down4_0, up6_conv], dim=1)
        merge6_conv = self.merge6_conv(merge6)
        up7 = self.up(merge6_conv)
        up7_conv = self.up7_conv(up7)
        merge7 = torch.cat([down3_0, up7_conv], dim=1)
        merge7_conv = self.merge7_conv(merge7)
        up8 = self.up(merge7_conv)
        up8_conv = self.up8_conv(up8)
        merge8  = torch.cat([down1, up8_conv], dim=1)
        merge8_conv = self.merge8_conv(merge8)
        up9 = self.up(merge8_conv)
        up9_conv = self.up9_conv(up9)
        merge9  = torch.cat([down1_0, up9_conv],dim=1)
        merge9_conv = self.merge9_conv(merge9)
        up10_conv = self.up10_conv(self.up(merge9_conv))
        out = self.out_conv(up10_conv)
        return out

class UnetFPN(nn.Module):
    def __init__(self,in_ch, num_classes):
        super(UnetFPN,self).__init__()
        self.conv1 = BasicConv(in_ch, 64, 3, stride=1, padding=1, bias=False)
        self.conv1_after = DoubleConv(64, 64, 3, strides=1, padding=1)
        self.down = nn.MaxPool2d(2, 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)

        self.down2_conv = DoubleConv(64, 128, 3, strides=1, padding=1)
        self.down3_conv = DoubleConv(128, 256, 3, strides=1, padding=1)
        self.down4_conv = DoubleConv(256, 512, 3, strides=1, padding=1)
        self.down5_conv = DoubleConv(512, 1024, 3, strides=1, padding=1)

        self.up6_conv = BasicConv(1024, 512, 1, stride=1, bias=False)
        self.up6_conv_after = DoubleConv(1024, 512, 3, strides=1, padding=1)
        self.up7_conv = BasicConv(512, 256, 1, stride=1, bias=False)
        self.up7_conv_after = DoubleConv(512, 256, 3, strides=1, padding=1)
        self.up8_conv = BasicConv(256, 128, 1, stride=1, bias=False)
        self.up8_conv_after = DoubleConv(256, 128, 3, strides=1, padding=1)
        self.up9_conv = BasicConv(128, 64, 1, stride=1, bias=False)
        self.up9_conv_after = DoubleConv(128, 64, 3, strides=1, padding=1)


        self.out1 = BasicConv(512, num_classes, 1, stride=1, padding=0, bias=False)
        self.out2 = BasicConv(256, num_classes, 1, stride=1, padding=0, bias=False)
        self.out3 = BasicConv(128, num_classes, 1, stride=1, padding=0, bias=False)
        self.out4 = BasicConv(64, num_classes, 1, stride=1, padding=0, bias=False)
        #self.out_conv = BasicConv(256, num_classes, 1, stride=1, padding=0, bias=False)
        self.num_classes = num_classes

    def forward(self, x):
        conv1 = self.conv1(x)
        down1_0 = self.conv1_after(conv1)
        down1 = self.down(down1_0)
        down2_0 = self.down2_conv(down1)
        down2 = self.down(down2_0)
        down3_0 = self.down3_conv(down2)
        down3 = self.down(down3_0)
        down4_0 = self.down4_conv(down3)
        down4 = self.down(down4_0)
        down5 = self.down5_conv(down4)
        up1 = self.up(down5)
        up1_conv = self.up6_conv(up1)
        up1_merge = torch.cat([down4_0, up1_conv], dim=1)
        up1_after = self.up6_conv_after(up1_merge)
        up2 = self.up(up1_after)
        up2_conv = self.up7_conv(up2)
        up2_merge = torch.cat([down3_0, up2_conv], dim=1)
        up2_after = self.up7_conv_after(up2_merge)
        up3 = self.up(up2_after)
        up3_conv = self.up8_conv(up3)
        up3_merge = torch.cat([down2_0, up3_conv], dim=1)
        up3_after = self.up8_conv_after(up3_merge)
        up4 = self.up(up3_after)
        up4_conv = self.up9_conv(up4)
        up4_merge = torch.cat([down1_0, up4_conv], dim=1)
        up4_after = self.up9_conv_after(up4_merge)
        out1 = self.out1(up1_after)
        out2 = self.out2(up2_after)
        out2 = out2 + self.up(out1)
        out3 = self.out3(up3_after)
        out3 = out3 + self.up(out2)
        out4 = self.out4(up4_after) + self.up(out3)
        return out4

class UnetFPNV2(nn.Module):
    def __init__(self,in_ch, num_classes):
        super(UnetFPNV2,self).__init__()
        self.conv1 = BasicConv(in_ch, 64, 3, stride=1, padding=1, bias=False)
        self.conv1_after = DoubleConv(64, 64, 3, strides=1, padding=1)
        self.down = nn.MaxPool2d(2, 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)

        self.down2_conv = DoubleConv(64, 128, 3, strides=1, padding=1)
        self.down3_conv = DoubleConv(128, 256, 3, strides=1, padding=1)
        self.down4_conv = DoubleConv(256, 512, 3, strides=1, padding=1)
        self.down5_conv = DoubleConv(512, 1024, 3, strides=1, padding=1)

        self.up6_conv = BasicConv(1024, 512, 1, stride=1, bias=False)
        self.up6_conv_after = DoubleConv(1024, 512, 3, strides=1, padding=1)
        self.up7_conv = BasicConv(512, 256, 1, stride=1, bias=False)
        self.up7_conv_after = DoubleConv(512, 256, 3, strides=1, padding=1)
        self.up8_conv = BasicConv(256, 128, 1, stride=1, bias=False)
        self.up8_conv_after = DoubleConv(256, 128, 3, strides=1, padding=1)
        self.up9_conv = BasicConv(128, 64, 1, stride=1, bias=False)
        self.up9_conv_after = DoubleConv(128, 64, 3, strides=1, padding=1)


        self.out1 = BasicConv(512, num_classes, 1, stride=1, padding=0, bias=False)
        self.out2 = BasicConv(256, num_classes, 1, stride=1, padding=0, bias=False)
        self.out3 = BasicConv(128, num_classes, 1, stride=1, padding=0, bias=False)
        self.out4 = BasicConv(64 , num_classes, 1, stride=1, padding=0, bias=False)
        #self.out_conv = BasicConv(256, num_classes, 1, stride=1, padding=0, bias=False)
        self.out_conv = BasicConv(num_classes*2, num_classes, 1, 1)
        self.num_classes = num_classes

    def forward(self, x):
        conv1 = self.conv1(x)
        down1_0 = self.conv1_after(conv1)
        down1 = self.down(down1_0)
        down2_0 = self.down2_conv(down1)
        down2 = self.down(down2_0)
        down3_0 = self.down3_conv(down2)
        down3 = self.down(down3_0)
        down4_0 = self.down4_conv(down3)
        down4 = self.down(down4_0)
        down5 = self.down5_conv(down4)
        up1 = self.up(down5)
        up1_conv = self.up6_conv(up1)
        up1_merge = torch.cat([down4_0, up1_conv], dim=1)
        up1_after = self.up6_conv_after(up1_merge)
        up2 = self.up(up1_after)
        up2_conv = self.up7_conv(up2)
        up2_merge = torch.cat([down3_0, up2_conv], dim=1)
        up2_after = self.up7_conv_after(up2_merge)
        up3 = self.up(up2_after)
        up3_conv = self.up8_conv(up3)
        up3_merge = torch.cat([down2_0, up3_conv], dim=1)
        up3_after = self.up8_conv_after(up3_merge)
        up4 = self.up(up3_after)
        up4_conv = self.up9_conv(up4)
        up4_merge = torch.cat([down1_0, up4_conv], dim=1)
        up4_after = self.up9_conv_after(up4_merge)
        out1 = self.out1(up1_after)
        out2 = self.out2(up2_after)
        out2 = self.out_conv(torch.cat([self.up(out1), out2], dim=1))
        out3 = self.out3(up3_after)
        out3 = self.out_conv(torch.cat([self.up(out2), out3], dim=1))
        out4 = self.out_conv(torch.cat([self.up(out3), self.out4(up4_after)], dim=1))
        return out4

class MobileNetv3_Large_UnetFPN(nn.Module):
    def __init__(self, in_ch, num_classes=3):
        super(MobileNetv3_Large_UnetFPN, self).__init__()
        self.down1_0 = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1)
        )
        self.down1 = Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2)
        self.down2_0 = Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1)
        self.down2 = Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(72), 2)
        self.down3_0 = nn.Sequential(
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(120), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(120), 1)
        )
        self.down3 = Block(3, 40, 240, 80, hswish(), None, 2)
        self.down4_0 = nn.Sequential(
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(480), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(672), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(672), 1)
        )
        self.down4 = Block(5, 160, 672, 160, hswish(), SeModule(672), 2)
        self.down5 = nn.Sequential(
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),
            nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(960),
            hswish()
        )
        #self.init_params()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.up6_conv = BasicConv(960, 160, ksize=1, stride=1, padding=0)
        self.merge6_conv = DoubleConv(320, 512, 3, strides=1, padding=1)
        self.up7_conv = BasicConv(512, 40, ksize=1, stride=1, padding=0)
        self.merge7_conv = DoubleConv(80, 256, 3, strides=1, padding=1)
        self.up8_conv = BasicConv(256, 24, ksize=1, stride=1, padding=0)
        self.merge8_conv = DoubleConv(48, 128, 3, strides=1, padding=1)
        self.up9_conv = BasicConv(128, 16, ksize=1, stride=1, padding=0)
        self.merge9_conv = DoubleConv(32, 64, 3, strides=1, padding=1)

        self.out1 = BasicConv(512, num_classes, ksize=1, stride=1, padding=0)
        self.out2 = BasicConv(256, num_classes, ksize=1, stride=1, padding=0)
        self.out3 = BasicConv(128, num_classes, ksize=1, stride=1, padding=0)
        self.out4 = BasicConv(64, num_classes, ksize=1, stride=1, padding=0)
        self.out5 = BasicConv(64, num_classes, ksize=1, stride=1, padding=0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self,x):
        down1_0 = self.down1_0(x)
        down1 = self.down1(down1_0)
        down2_0 = self.down2_0(down1)
        down2 = self.down2(down2_0)
        down3_0 = self.down3_0(down2)
        down3 = self.down3(down3_0)
        down4_0 = self.down4_0(down3)
        down4 = self.down4(down4_0)
        down5 = self.down5(down4)
        up6 = self.up(down5)
        up6_conv = self.up6_conv(up6)
        merge6 = torch.cat([down4_0,up6_conv], dim=1)
        merge6_conv = self.merge6_conv(merge6)
        up7 = self.up(merge6_conv)
        up7_conv =self.up7_conv(up7)
        merge7 = torch.cat([down3_0, up7_conv], dim=1)
        merge7_conv = self.merge7_conv(merge7)
        up8 = self.up(merge7_conv)
        up8_conv = self.up8_conv(up8)
        merge8 = torch.cat([down2_0, up8_conv], dim=1)
        merge8_conv = self.merge8_conv(merge8)
        up9 = self.up(merge8_conv)
        up9_conv = self.up9_conv(up9)
        merge9 = torch.cat([down1_0, up9_conv], dim=1)
        merge9_conv = self.merge9_conv(merge9)

        out1 = self.out1(merge6_conv)
        out2 = self.up(out1) + self.out2(merge7_conv)
        out3 = self.up(out2) + self.out3(merge8_conv)
        out4 = self.up(out3) + self.out4(merge9_conv)
        out = self.up(out4)
        return out

class MobileNetv3_Small_UnetFPN(nn.Module):
    def __init__(self,in_ch, num_classes=4):
        super(MobileNetv3_Small_UnetFPN, self).__init__()
        self.down1_0 = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )
        self.down1 = Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2)
        self.down2_0 = Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2)
        self.down2 = Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1)
        self.down3_0 = Block(5, 24, 96, 40, hswish(), SeModule(96), 1)
        self.down3 = Block(5, 40, 240, 40, hswish(), SeModule(240), 2)
        self.down4_0 = nn.Sequential(
            Block(5, 40, 240, 40, hswish(), SeModule(240), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(240), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(120), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(144), 1)
        )
        self.down4 = Block(5, 48, 288, 96, hswish(), SeModule(288), 2)
        self.down5 = nn.Sequential(
            Block(5, 96, 576, 96, hswish(), SeModule(576), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(576), 1),
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(576),
            hswish()
        )
        self.init_params()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.up6_conv = BasicConv(576, 48, ksize=1, stride=1, padding=0)
        self.merge6_conv = DoubleConv(96, 512, 3, strides=1, padding=1)
        self.up7_conv = BasicConv(512, 40, ksize=1, stride=1, padding=0)
        self.merge7_conv = DoubleConv(80, 256, 3, strides=1, padding=1)
        self.up8_conv = BasicConv(256, 24, ksize=1, stride=1, padding=0)
        self.merge8_conv = DoubleConv(40, 128, 3, strides=1, padding=1)
        self.up9_conv = BasicConv(128, 16, ksize=1, stride=1, padding=0)
        self.merge9_conv = DoubleConv(32, 64, 3, strides=1, padding=1)
        self.up10_conv = DoubleConv(64, 64, 3, strides=1, padding=1)

        self.out1 = BasicConv(512, num_classes,ksize=1, stride=1, padding=0)
        self.out2 = BasicConv(256, num_classes, ksize=1, stride=1, padding=0)
        self.out3 = BasicConv(128, num_classes, ksize=1, stride=1, padding=0)
        self.out4 = BasicConv(64, num_classes, ksize=1, stride=1, padding=0)
        self.out5 = BasicConv(64, num_classes, ksize=1, stride=1, padding=0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        down1_0 = self.down1_0(x)
        down1 = self.down1(down1_0)
        down2_0 = self.down2_0(down1)
        down2 = self.down2(down2_0)
        down3_0 = self.down3_0(down2)
        down3 = self.down3(down3_0)
        down4_0 = self.down4_0(down3)
        down4 = self.down4(down4_0)
        down5 = self.down5(down4)
        up6 = self.up(down5)
        up6_conv = self.up6_conv(up6)
        merge6 = torch.cat([down4_0, up6_conv], dim=1)
        merge6_conv = self.merge6_conv(merge6)
        up7 = self.up(merge6_conv)
        up7_conv = self.up7_conv(up7)
        merge7 = torch.cat([down3_0, up7_conv], dim=1)
        merge7_conv = self.merge7_conv(merge7)
        up8 = self.up(merge7_conv)
        up8_conv = self.up8_conv(up8)
        merge8  = torch.cat([down1, up8_conv], dim=1)
        merge8_conv = self.merge8_conv(merge8)
        up9 = self.up(merge8_conv)
        up9_conv = self.up9_conv(up9)
        merge9  = torch.cat([down1_0, up9_conv],dim=1)
        merge9_conv = self.merge9_conv(merge9)
        up10 = self.up(merge9_conv)
        up10_conv = self.up10_conv(up10)

        out1 = self.out1(merge6_conv)
        out2 = self.out2(merge7_conv)
        out2 = out2 + self.up(out1)
        out3 = self.out3(merge8_conv)
        out3 = out3 + self.up(out2)
        out4 = self.out4(merge9_conv)
        out4 = out4 + self.up(out3)
        out5 = self.out5(up10_conv) + self.up(out4)
        return out5

class ResUnetFPN(nn.Module):
    '''
    An implementation of the ResNet Unet with Multi scale predict
    '''
    def __init__(self, in_ch, num_classes, n_layers=18):
        '''
        Initialize the Model
        :param in_ch: Input channels of the model, int
        :param num_classes: The number of the classes
        :param n_layers: The number of laryers for resnet
        '''
        super(ResUnetFPN, self).__init__()
        ch = 64
        # self.conv1 = BasicConv(in_ch, ch, 3, 1, padding=1, bias=False)
        # self.conv2 = BasicConv(ch, ch, 3, 2, padding=1, bias=False)
        # self.conv3 = BasicConv(ch, ch, 3, 1, padding=1, bias=False)
        self.conv1 = BasicConv(in_ch, ch, 7, 2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)

        self.n_blocks = get_layers(n_layers)
        if n_layers < 50:
            block = BasicBlock
        else:
            block = BottleBlock

        self.block1 = block(ch, ch, 1, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block1_x = block(in_cha, ch, 1)

        ch *= 2
        self.block2 = block(in_cha, ch, 2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block2_x = block(in_cha, ch, 1)

        ch *= 2
        self.block3 = block(in_cha, ch, 2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block3_x = block(in_cha, ch, 1)
        ch *= 2
        self.block4 = block(in_cha, ch, 2, downsample=True)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        self.block4_x = block(in_cha, ch, 1)
        if n_layers < 50:
            in_cha = ch
        else:
            in_cha = ch * 4
        #self.init_params()
        self.up = nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True)
        if n_layers < 50:
            self.center_conv = BasicConv(in_cha, 512, 1, stride=1, padding=0, bias=False)
            self.center_after = DoubleConv(512+256, 512, 3, strides=1, padding=1)
            self.up6_conv = BasicConv(512, 256, 1, stride=1, padding=0, bias=False)
            self.merge6_conv = DoubleConv(256+128, 256, 3, strides=1, padding=1)
            self.up7_conv = BasicConv(256, 128, 1, stride=1, padding=0, bias=False)
            self.merge7_conv = DoubleConv(128+64, 128, 3, strides=1, padding=1)
            self.up8_conv = BasicConv(128, 64, 1, stride=1, padding=0,bias=False)
            self.merge8_conv = DoubleConv(128, 64, 3, strides=1, padding=1, bias=False)
        else:
            self.center_conv = BasicConv(in_cha, 1024, 1, stride=1, padding=0, bias=False)
            self.center_after = DoubleConv(2048, 512,3, strides=1, padding=1)
            self.up6_conv = BasicConv(512, 256, 1, stride=1, padding=0, bias=False)
            self.merge6_conv = DoubleConv(512+256, 256, 3, strides=1, padding=1)
            self.up7_conv = BasicConv(256, 128, 1, stride=1, padding=0, bias=False)
            self.merge7_conv = DoubleConv(256+128, 64, 3, strides=1, padding=1)
            self.up8_conv = BasicConv(64, 64, 1, stride=1, padding=0, bias=False)
            self.merge8_conv = DoubleConv(128, 64, 3, strides=1, padding=1)
        self.up9_conv = DoubleConv(64, 64, 3, strides=1, padding=1)
        self.out1 = BasicConv(512, num_classes, 1, stride=1, padding=0, bias=False)
        self.out2 = BasicConv(256, num_classes, 1, stride=1, padding=0, bias=False)
        self.out3 = BasicConv(128, num_classes, 1, stride=1, padding=0, bias=False)
        self.out4 = BasicConv(64, num_classes, 1, stride=1, padding=0, bias=False)
        self.out5 = BasicConv(64, num_classes, 1, stride=1, padding=0, bias=False)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        #first = self.conv3(self.conv2(self.conv1(x)))
        first = self.conv1(x)
        pool1 = self.pool1(first)
        down1_0 = self.block1(pool1)
        for i in range(1, self.n_blocks[0] - 1):
            down1_0 = self.block1_x(down1_0)
        down1 = self.block2(down1_0)
        down2_0 = self.block2_x(down1)
        for i in range(1, self.n_blocks[1] - 1):
            down2_0 =self.block2_x(down2_0)
        down2 = self.block3(down2_0)
        down3_0 = self.block3_x(down2)
        for i in range(1, self.n_blocks[2]-1):
            down3_0 = self.block3_x(down3_0)
        down3 = self.block4(down3_0)
        down4 = self.block4_x(down3)
        for i in range(1, self.n_blocks[3]-1):
            down4 = self.block4_x(down4)
        center = self.up(down4)
        center_conv = self.center_conv(center)
        merge = torch.cat([down3_0, center_conv], dim=1)
        center_after = self.center_after(merge)
        up6 = self.up(center_after)
        up6_conv = self.up6_conv(up6)
        merge6 = torch.cat([down2_0, up6_conv], dim=1)
        merge6_conv = self.merge6_conv(merge6)
        up7 = self.up(merge6_conv)
        up7_conv = self.up7_conv(up7)
        merge7 = torch.cat([down1_0, up7_conv], dim=1)
        merge7_conv = self.merge7_conv(merge7)
        up8 = self.up(merge7_conv)
        up8_conv = self.up8_conv(up8)
        merge8 = torch.cat([first, up8_conv], dim=1)
        merge8_conv = self.merge8_conv(merge8)
        up9  = self.up(merge8_conv)
        up9_conv = self.up9_conv(up9)
        out1 = self.out1(center_after)
        out2 = self.out2(merge6_conv) + self.up(out1)
        out3 = self.out3(merge7_conv) + self.up(out2)
        out4 = self.out4(merge8_conv) + self.up(out3)
        out5 = self.out5(up9_conv) + self.up(out4)
        return out5

def get_paramaers(model:nn.Module):
    params = list(model.parameters())
    par_cnt = 0
    for par in params:
        l = 1
        for i in par.size():
            l *= i
        print("Size:{} Parameters num:{}".format(str(par.size()), l))
        par_cnt += l
    print("total parameters num:{} M".format(par_cnt / (1024 * 1024)))

if __name__ == "__main__":
    import numpy as np
    x = np.random.normal(0,1,[1, 3, 64, 64])
    x = torch.Tensor(x)
    #model =MobileNetv3_Large_Unet(3)
    #model = ResUnet(3,4)
    #model = ResUnetFPN(3, 3, 18)
    model = UnetFPNV2(3, 4)
    #model = MobileNetv3_Small_UnetFPN(3, 4)
    get_paramaers(model)
    out = model(x)
    print(out.shape)
