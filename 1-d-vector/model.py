# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> model
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/11 上午9:12
@Description        ：
                    _ooOoo_    
                   o8888888o    
                   88" . "88    
                   (| -_- |)    
                    O\ = /O    
                ____/`---'\____    
                 .' \\| |// `.    
               / \\||| : |||// \    
             / _||||| -:- |||||- \    
               | | \\\ - /// | |    
             | \_| ''\---/'' | |    
              \ .-\__ `-` ___/-. /    
           ___`. .' /--.--\ `. . __    
        ."" '< `.___\_<|>_/___.' >'"".    
       | | : `- \`.;`\ _ /`;.`/ - ` : | |    
         \ \ `-. \_ __\ /__ _/ .-` / /    
 ======`-.____`-.___\_____/___.-`____.-'======    
                    `=---='    
 .............................................    
              佛祖保佑             永无BUG
==================================================
"""
import numpy as np
import torch
from torch import nn, autograd, optim
from matplotlib import pyplot as plt

import constants as c


class MyMLP(nn.Module):

    def __init__(self, in_size, out_size):
        super(MyMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc1 = nn.Linear(self.in_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=c.nmels, hidden_size=c.hidden_lstm,
                            num_layers=c.num_layers_lstm, batch_first=True)

        self.fc1 = nn.Linear(c.hidden_lstm, c.hidden_fc)
        self.fc2 = nn.Linear(c.hidden_fc, c.train_num_spks)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = self.tanh(x)
        x = x[:,-1,:]
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        return x
if __name__ == '__main__':
    x = torch.rand(64,100,40)
    net = MyLSTM()
    y = net(x)
    print(y.shape)
