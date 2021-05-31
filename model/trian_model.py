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
        self.lstm = nn.LSTM(hp.data.nmels,hp.model.lstm_hidden,num_layers=hp.model.num_layer,batch_first=True)
        for name,param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param,0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        self.dense = nn.Linear(hp.model.lstm_hidden,hp.model.linear_hidden)

    def forward(self,x):
        y,_ = self.lstm(x)  # torch.Size([20, 183, 64])
        y = y[:,y.size(1)-1]  # 取最后一帧  torch.Size([20, 64])
        y = self.dense(y)  #torch.Size([20, 128])

        out = y / torch.norm(y,dim=1).unsqueeze(1)
        return out

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

    x = torch.randn([20,183,40])

    mylstm = MyLSTM()
    out = mylstm(x)
    print(out.shape)




