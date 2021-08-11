# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> lstm
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/25 下午3:27
@Description        ：
==================================================
"""

import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F


class MyLSTM(nn.Module):
    def __init__(self, in_size, num_hiddens, num_layers):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(in_size, num_hiddens, num_layers,
                            dropout=0.5, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
        self.fc1 = nn.Linear(num_hiddens, 128)
        self.fc2 = nn.Linear(128, 462)
        # self.fc3 = nn.Linear(256, 462)

    def forward(self, x):
        y, _ = self.lstm(x)
        y = y[:,-1,:]  # 取最后一帧
        out1 = self.fc1(y)
        out2 = self.fc2(out1)
        return out1, out2

if __name__ == '__main__':
    x = torch.randn([6*10,299,13])

    net = MyLSTM(13,64,2)
    out1, out2 = net(x)
    print(out1.shape)
    print(out2.shape)

    summary(net.to(torch.device('cuda')), (299, 13))