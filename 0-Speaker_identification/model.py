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
from torch.nn import functional as F

class MyMLP(nn.Module):

    def __init__(self,in_size,out_size):
        super(MyMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc1 = nn.Linear(self.in_size,256)
        self.fc2 = nn.Linear(256, out_size)
        # self.fc3 = nn.Linear(256, out_size)
        self.batchnorm0 = nn.BatchNorm1d(in_size)
        self.batchnorm1 = nn.BatchNorm1d(256)
        # self.batchnorm2 = nn.BatchNorm1d(256)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        self.batchnorm0(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        # x = self.dropout(x)
        out = self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.batchnorm2(x)
        # out = self.fc3(x)
        return out



