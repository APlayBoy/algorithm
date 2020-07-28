import torch.nn as nn
import torch

__all__ = ["ResNet", 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, 1, 1, False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1, 1, 1, False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        indentity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            indentity = self.downsample(x)

        out += indentity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                  padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


net_layers = {"resnet18": [[2, 2, 2, 2], BasicBlock, 1, 64],
              "resnet34": [[3, 4, 6, 3], BasicBlock, 1, 64],
              "resnet50": [[3, 4, 6, 3], Bottleneck, 1, 64],
              "resnet101": [[3, 4, 23, 3], Bottleneck, 1, 64],
              "resnet152": [[3, 8, 36, 3], Bottleneck, 1, 64],
              "resnet50_32x4d": [[3, 4, 6, 3], Bottleneck, 32, 4],
              "resnet101_32x8d": [[3, 4, 23, 3], Bottleneck, 32, 8],
              "wide_resnet50_2": [[3, 4, 6, 3], Bottleneck, 1, 64 * 2],
              "wide_resnet101_2": [[3, 4, 23, 3], Bottleneck, 1, 64 * 2]}


class ResNet(nn.Module):

    def __init__(self, net_type="resnet50", num_classes=1000, replace_stride_with_dilation=None):
        layers, block, groups, width_per_group =net_layers[net_type]
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dialation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear( 512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilate=self.dialation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


resnet18 = ResNet(net_type="resnet18")
resnet34 = ResNet(net_type="resnet34")
resnet50 = ResNet(net_type="resnet50")
resnet101 = ResNet(net_type="resnet101")
resnet152 = ResNet(net_type="resnet152")
