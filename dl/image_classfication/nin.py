import torch.nn as nn
import torch


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.AvgPool2d(x, kernel_size=x.size()[2:])


class MinBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(MinBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class NIN(nn.Module):

    def __init__(self, num_class):
        super(NIN, self).__init__()
        self.net = nn.Sequential(
            MinBlock(3, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MinBlock(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MinBlock(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(0.5),
            MinBlock(384, self.classes, kernel_size=3, stride=1, padding=1),
            GlobalAvgPool2d()
        )

    def forward(self, x):
        x = self.net(x)
        return torch.flatten(x, 1)
