import torch.nn as nn
import torch
from collections import OrderedDict

__all__ = ["DenseNet"]


class DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size,
                 drop_rate, memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size,
                                bn_size * growth_rate, kernel_size=1,
                                           stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 =nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            input = x
        else:
            input = torch.cat(input, 1)
        input = self.norm1(input)
        input = self.relu1(input)
        input = self.conv1(input)

        input = self.norm2(input)
        input = self.relu2(input)
        input = self.conv2(input)
        if self.drop_rate > 0:
            new_features = nn.Dropout(new_features, )