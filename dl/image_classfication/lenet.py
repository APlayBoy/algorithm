import torch
import torch.nn as nn

__all__ = ['LeNet', ]


class LeNet(nn.Module):

    def __init__(self):

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_features_ids = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 1, 2],
         [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2],
         [0, 1, 2, 4], [1, 2, 4, 5], [2, 3, 5, 0],
         [0, 1, 2, 3, 4, 5]]

        self.conv2 = [nn.Conv2d(len(ids), 1, 5) for ids in self.conv2_features_ids]
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Sigmoid(nn.Linear(25 * 16, 120))
        self.fc2 = nn.Sigmoid(nn.Linear(120, 84))

        self.fc3 = nn.Linear(84, 10)
        #TODO:径向基函数的实现

    def foward(self, x):
        conv1 = self.conv1(x)  #输入 batch_size * 28*28*1 输出28*28*1
        pool1 = self.pool1(conv1) #输入 batch_size * 28*28*6 输出batch_size*14*14*6

        # 输入 batch_size * 14*14*6 输出batch_size*10*10*16
        conv2 = [conv(pool1[ids]) for ids, conv in zip(self.conv2_features_ids, self.conv2)]
        conv2 = torch.stack(conv2, 0)
        pool2 = self.pool2(conv2) #输入batch_size * 10*10*16 输出batch_size*5*5*16
        conv2 = conv2.view(conv2.size(0), -1)#输入输出batch_size*5*5*16 输出batch_size*400

        fc1 = self.fc1(pool2)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)


if __name__ == '__main__':
    from torchstat import stat



