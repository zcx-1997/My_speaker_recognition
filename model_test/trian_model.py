#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/25 9:21
    Author  : 春晓
    Software: PyCharm
"""
import torch
import random
from torch import nn

from hparam import hparam as hp
from utils.utils import get_centroids,get_cossim,calc_loss

class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(hp.data.nmels,hp.model.lstm_hidden,num_layers=hp.model.num_layer,    # (40,64,1)
                            dropout=0.5,batch_first=True)
        # for name,param in self.lstm.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param,0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_uniform_(param)
        self.linear1 = nn.Linear(hp.model.lstm_hidden,hp.model.linear_hidden)   # (64,128)
        self.linear2 = nn.Linear(hp.model.linear_hidden,hp.data.train_speaker)  # (128,567)

    def forward(self,x):  # torch.Size([6*10, 180, 40])
        y,_ = self.lstm(x)  # (6*10,180,64)
        y = y[:,-1,:]  # 取最后时间T的输出  torch.Size([6*10, 64])
        out1 = self.linear1(y)  # torch.Size([60, 128])
        out2 = self.linear2(out1)  # (60,567)
        # out = y / torch.norm(y,dim=1).unsqueeze(1)
        return out1,out2

class GE2ELoss(nn.Module):

    def __init__(self,device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device),requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device),requires_grad=True)
        self.device = device

    def forward(self,x):
        torch.clamp(self.w, 1e-6)  # 限制w的最小值
        centroids = get_centroids(x)
        # centroids = x.mean(dim=1)
        cossim = get_cossim(x,centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss,_ = calc_loss(sim_matrix)
        return loss


if __name__ == '__main__':

    x = torch.randn([6*10,180,40])

    mylstm = MyLSTM()
    out1,out2 = mylstm(x)
    print(out1.shape)
    print(out2.shape)




