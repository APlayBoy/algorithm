import torch.nn as nn
import torch

__all__ = ['GoogLeNet', ]


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.ReLU(x, inplace=True)


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kenel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_class, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_class)

    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = nn.ReLU(self.fc(x), inplace=True)
        x = nn.Dropout(x, 0.7, training=self.training)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_class=1000, aux_logits=True, transform_input=False, init_weights=None,
                 blocks=None):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        conv_block, inception_block, inception_aux_block = blocks[0], blocks[1], blocks[2]
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, paddig=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxplool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4e = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 230, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 284, 48, 238, 128)

        if aux_logits:
            self.aux1 = inception_block(512, num_class)
            self.aux2 = inception_block(528, num_class)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_class)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):#x Nx3x244x244
        x = self.conv1(x) #Nx64x112x112
        x = self.maxpool1(x) #Nx64x56x56
        x = self.conv2(x) #Nx64x56x56
        x = self.conv3(x) #Nx192x56x56
        x = self.maxpool2(x) #Nx192x128x128

        x = self.inception3a(x) #Nx256x28x28
        x = self.inception3b(x) #Nx480x28x28
        x = self.maxplool3(x)  #Nx480x14x14

        x = self.inception4a(x)  #Nx512x14x14
        aux1 = self.aux1(x) if self.aux1 is not None and self.training else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.aux2 is not None and self.training else None

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x, aux1, aux2

    def forward(self, x):
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        if self.training and self.aux_logits:
            return x, aux1, aux2
        else:
            return x

