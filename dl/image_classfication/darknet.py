import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys

__all__ = ['DarkNet_19', 'DarkNet_53', 'DarkNet_Light', 'DarkNet_Tiny']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch // 2, 1),
                Conv_BN_LeakyReLU(ch // 2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_19(nn.Module):
    def __init__(self, num_classes=1000):
        print("Initializing the darknet19 network ......")

        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )

        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

        # self.conv_7 = nn.Conv2d(1024, 1000, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        C_4 = self.conv_4(x)
        C_5 = self.conv_5(self.maxpool_4(C_4))
        C_6 = self.conv_6(self.maxpool_5(C_5))

        # x = self.conv_7(C_6)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # return x
        return C_4, C_5, C_6


class DarkNet_53(nn.Module):

    def __init__(self, num_classes=1000):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=4)
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, targets=None):
        x = self.layer_1(x)
        x = self.layer_2(x)
        C_3 = self.layer_3(x)
        C_4 = self.layer_4(C_3)
        C_5 = self.layer_5(C_4)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return C_3, C_4, C_5


class DarkNet_Tiny(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet_Tiny, self).__init__()
        # backbone network : DarkNet-Tiny
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            Conv_BN_LeakyReLU(32, 32, 3, padding=1, stride=2)
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            Conv_BN_LeakyReLU(64, 64, 3, padding=1, stride=2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 128, 3, padding=1, stride=2),
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 256, 3, padding=1, stride=2),
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 512, 3, padding=1, stride=2),
        )

        # self.conv_6 = nn.Conv2d(512, 1000, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        C_3 = self.conv_3(x)
        C_4 = self.conv_4(C_3)
        C_5 = self.conv_5(C_4)
        # x = self.conv_6(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        return C_3, C_4, C_5


class DarkNet_Light(nn.Module):
    def __init__(self, num_classes=1000):
        super(DarkNet_Light, self).__init__()
        # backbone network : DarkNet_Light
        self.conv_1 = Conv_BN_LeakyReLU(3, 16, 3, 1)
        self.maxpool_1 = nn.MaxPool2d((2, 2), 2)  # stride = 2

        self.conv_2 = Conv_BN_LeakyReLU(16, 32, 3, 1)
        self.maxpool_2 = nn.MaxPool2d((2, 2), 2)  # stride = 4

        self.conv_3 = Conv_BN_LeakyReLU(32, 64, 3, 1)
        self.maxpool_3 = nn.MaxPool2d((2, 2), 2)  # stride = 8

        self.conv_4 = Conv_BN_LeakyReLU(64, 128, 3, 1)
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)  # stride = 16

        self.conv_5 = Conv_BN_LeakyReLU(128, 256, 3, 1)
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)  # stride = 32

        self.conv_6 = Conv_BN_LeakyReLU(256, 512, 3, 1)
        self.maxpool_6 = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d((2, 2), 1)  # stride = 32
        )

        self.conv_7 = Conv_BN_LeakyReLU(512, 1024, 3, 1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        C_4 = self.conv_5(x)  # stride = 16
        x = self.maxpool_5(C_4)
        x = self.conv_6(x)
        x = self.maxpool_6(x)
        C_5 = self.conv_7(x)  # stride = 32

        return C_4, C_5

