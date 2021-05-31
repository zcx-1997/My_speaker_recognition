#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/25 10:20
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn

x = torch.zeros(10,4,5)

net = nn.RNN(5,3,2)
out,h = net(x)
print(out.shape)
print(h.shape)

lstm = nn.LSTM(5,3,2)
out1,(h1,c1) = lstm(x)
for name, param in lstm.named_parameters():
    if 'bias' in name:
        nn.init.constant_(param, 0.0)
    elif 'weight' in name:
        nn.init.constant_(param, 1.0)
print(out1.shape)
print(h1.shape)
print(c1.shape)

x = torch.zeros(4,10,5)
lstm2 = nn.LSTM(5,3,2,batch_first=True)
for name, param in lstm2.named_parameters():
    if 'bias' in name:
        nn.init.constant_(param, 0.0)
    elif 'weight' in name:
        nn.init.constant_(param, 1.0)
out2,(h2,c2) = lstm2(x)
print(out2.shape)
print(h2.shape)
print(c2.shape)

print(h1==h2)
print(c1==c2)

