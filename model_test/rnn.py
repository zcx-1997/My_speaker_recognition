#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/25 10:20
    Author  : 春晓
    Software: PyCharm
"""
import torch
from torch import nn

x = torch.zeros(10, 4, 5)   # (seq_len,batch,fea_len)

print("======================== RNN ======================")
net = nn.RNN(5, 3, 1)       # (fea_len,units,num_layer)
out, h = net(x)             # (seq_len,batch,units) (num_layer,batch,units)
print('out:', out.shape)    # torch.Size([4, 10, 3])
print('h:  ', h.shape)      # torch.Size([1, 4, 3])
# print(h.squeeze().shape)    #torch.Size([4, 3])

print("batch_first")
net = nn.RNN(5, 3, 2, batch_first=True)
x = x.permute(1, 0, 2)      # torch.Size([4, 10, 5])
out, h = net(x)
print('out:', out.shape)    # torch.Size([4, 10, 3])
print('h:  ', h.shape)      # torch.Size([2, 4, 3])

# lstm = nn.LSTM(5,3,2)     # (fea_len,units,num_layer)
# out1,(h1,c1) = lstm(x)    # (seq_len,batch,units) (num_layer,batch,units)
# print(out1.shape)         # (10,4,3)
# print(h1.shape)           # (2,4,3)
# print(c1.shape)           # (2,4,3)
# print(out1[-1]==h1[-1])   # True
#
# x = torch.zeros(4,10,5)                       # (batch,seq_len,fea_len)
# lstm2 = nn.LSTM(5,3,2,batch_first=True)       # (fea_len,units,num_layer)
#
# out2,(h2,c2) = lstm2(x)                       # (batch,seq_len,units) (num_layer,batch,units)
# print(out2.shape)                             # (4,10,3)
# print(h2.shape)                               # (2,4,3)
# print(c2.shape)                               # (2,4,3)
# print(out2[:,-1,:]==h2[-1])                   # True
#
#
# print(h1==h2)
# print(c1==c2)
# print(h1)
# print(h2)
# print(out2[:,-1,:].shape)
