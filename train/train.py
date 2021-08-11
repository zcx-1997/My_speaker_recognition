#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/24 15:55
    Author  : 春晓
    Software: PyCharm
"""
import os
import random
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime

from hparam import hparam as hp
from data_process.data_load import TIMITUnProcessed,TIMITPreprocessed
from model_test.trian_model import MyLSTM,GE2ELoss,get_centroids,get_cossim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data,net,loss,optimizer,epochs):

    # ge2e_loss = GE2ELoss(device)
    # optimizer = torch.optim.SGD([{'params':net.parameters()},{'params':ge2e_loss.parameters()}],lr=hp.train.lr)

    # train
    net.train()
    for epoch in range(epochs):
    # for epoch in range(1):
        total_loss = 0
        for i,(x,y) in enumerate(data):
            x.to(device); y.to(device)  # (6,10,180,40)  (6,10)
            x = torch.reshape(x, (x.size(0) * x.size(1), x.size(2), x.size(3)))  # (6*10,180,40)
            y = torch.reshape(y, (y.size(0) * y.size(1),))  # (6*10,)

            optimizer.zero_grad()
            embedding, y_hat = net(x)  # (6*10,128) (6*10,567)
            l = loss(y_hat, y)
            # loss = ge2e_loss(logits)
            l.backward()
            # 梯度剪切，将梯度限制在一个阈值内，防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(net.parameters(),3.0)
            # torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(),1.0)
            optimizer.step()

            total_loss += l

        print("Epoch:%d,Loss:%f"%(epoch+1,total_loss/len(train_loader)))

    # # test
    # if hp.data.data_preprocessed:
    #     test_db = TIMITPreprocessed(train=False)
    # else:
    #     test_db = TIMITUnProcessed(train=False)
    # test_loader = DataLoader(test_db, batch_size=hp.test.batch_size,shuffle=True,drop_last=True)
    #
    # net.eval()
    # avg_EER = 0
    # for i, x in enumerate(test_loader):
    #
    #     x = x.to(device)  # (4,6,160,40)
    #     enroll_x, veri_x = torch.split(x, int(x.size(1)/2), dim=1)  # (4,3,160,40)
    #     enroll_x = torch.reshape(enroll_x,(enroll_x.size(0)*enroll_x.size(1),enroll_x.size(2),enroll_x.size(3)))
    #     veri_x = torch.reshape(veri_x, (veri_x.size(0) * veri_x.size(1), veri_x.size(2), veri_x.size(3)))
    #
    #     enroll_logits = net(enroll_x)
    #     veri_logits = net(veri_x)
    #
    #     enroll_logits = torch.reshape(enroll_logits,(hp.test.batch_size,hp.test.num_utters//2,enroll_logits.size(1)))
    #     veri_logits = torch.reshape(veri_logits, (hp.test.batch_size, hp.test.num_utters // 2, veri_logits.size(1)))
    #
    #     enroll_centroids = get_centroids(enroll_logits)
    #     sim_matrix = get_cossim(veri_logits,enroll_centroids)
    #
    #     # EER
    #     diff,threshold = 1,0
    #     EER_FAR,EER_FRR,EER = 0,0,0
    #
    #     for thres in [0.01*i+0.5 for i in range(50)]:
    #         sim_matrix_thres = sim_matrix>thres
    #
    #         FAR = (sum([sim_matrix_thres[i].float().sum() - sim_matrix_thres[i, :, i].float().sum() for i in range(int(hp.test.batch_size))])
    #                / (hp.test.batch_size - 1.0) / (float(hp.test.num_utters / 2)) / hp.test.batch_size)
    #
    #         FRR = (sum([hp.test.num_utters / 2 - sim_matrix_thres[i, :, i].float().sum() for i in range(int(hp.test.batch_size))])
    #                / (float(hp.test.num_utters / 2)) / hp.test.batch_size)
    #
    #         # Save threshold when FAR = FRR (=EER)
    #         if diff > abs(FAR - FRR):
    #             diff = abs(FAR - FRR)
    #             EER = (FAR + FRR) / 2
    #             threshold = thres
    #             EER_FAR = FAR
    #             EER_FRR = FRR
    #     avg_EER += EER
    #     print("EER:%f (threshod:%f,FAR:%f,FRR:%f)"%(EER,threshold,EER_FAR,EER_FRR))
    # avg_EER = avg_EER/(len(test_loader))
    # print("avg_EER:%f,"%avg_EER)


def test():
    pass

if __name__ == '__main__':

    if hp.data.data_preprocessed:
        train_db = TIMITPreprocessed()
    else:
        train_db = TIMITUnProcessed()  # error
    train_loader = DataLoader(train_db,batch_size=hp.train.batch_size,shuffle=True,drop_last=True)
    # TIMITUnProcessed: [batch,10,180,40]
    # for (x,labels) in train_loader:
    #     print(x.shape)
    #     print(labels)
    #     break

    epochs = hp.train.epochs
    lr = hp.train.lr
    # epochs = 10
    # lr = 0.1


    net = MyLSTM().to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(),lr)
    #
    # for x,y in train_loader:
    #     x.to(device);y.to(device)  #(6,10,180,40)  (6,10)
    #     x = torch.reshape(x, (x.size(0) * x.size(1), x.size(2), x.size(3)))  # (6*10,180,40)
    #     _, y_hat = net(x)  # (6*10,567)
    #     y = torch.reshape(y,(y.size(0)*y.size(1),))  # (6*10,)
    #     l = loss(y_hat,y)

    train(train_loader,net,loss,optimizer,epochs)

