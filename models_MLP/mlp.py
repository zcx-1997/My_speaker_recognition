# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> mlp
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/21 下午9:39
@Description        ：
==================================================
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class MyMLP(nn.Module):

    def __init__(self, in_size):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(in_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256,256)
        self.fc5 = nn.Linear(256,630)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.fc4(x)
        out = self.fc5(x)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = MyMLP(13).to(device)
    summary(net, (1, 39))
