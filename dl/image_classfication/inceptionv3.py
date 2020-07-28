import torch.nn as nn
import torch


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.ReLU(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.brand1x1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)

        self.branch3x3db1_1 = conv_block(in_channle)