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

class MyMLP(nn.Module):

    def __init__(self,in_size,out_size):
        super(MyMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc1 = nn.Linear(self.in_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out



