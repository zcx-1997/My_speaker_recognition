# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> rnn
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/21 下午4:00
@Description        ：
==================================================
"""

import torch
from torch import nn
from torchsummary import summary


class MyRNN(nn.Module):
    def __init__(self, input_size, num_hiddens, num_layers):
        super(MyRNN, self).__init__()
        self.hiddens = num_hiddens
        self.rnn = nn.RNN(input_size, self.hiddens, num_layers,
                          batch_first=True)
        self.fc1 = nn.Linear(self.hiddens, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 630)
        self.batchnorm = nn.BatchNorm1d(self.hiddens)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(256)
        # self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.rnn(x)  #torch.Size([2, 299, 64])
        x = x[:,-1,:].squeeze()  # torch.Size([2, 64])
        x = self.batchnorm(x)
        x = self.relu(self.batchnorm1(self.fc1(x)))
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.relu(self.batchnorm2(self.fc3(x)))
        x = self.relu(self.batchnorm2(self.fc4(x)))
        out = self.fc5(x)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.zeros((2, 299, 13))
    x = x.to(device)
    net = MyRNN(13, 64, 1).to(device)
    summary(net, (299,13))
    y = net(x)
    print(y.shape)
