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
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_process.data_load import TIMITUnProcessed,TIMITPreprocessed
from model.trian_model import MyLSTM,GE2ELoss,get_centroids,get_cossim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train():

    if hp.data.data_preprocessed:
        train_db = TIMITPreprocessed()
    else:
        train_db = TIMITUnProcessed()
    train_loader = DataLoader(train_db,batch_size=hp.train.batch_size,shuffle=True,drop_last=True)

    # TIMITUnProcessed: 116,[4,5,183,40]
    # TIMITPreprocessed: 141,[4,5,160,40]

    net = MyLSTM().to(device)

    ge2e_loss = GE2ELoss(device)
    optimizer = torch.optim.SGD([{'params':net.parameters()},{'params':ge2e_loss.parameters()}],lr=hp.train.lr)

    # train
    net.train()
    for epoch in range(hp.train.epochs):
    # for epoch in range(1):
        total_loss = 0
        for i,x in enumerate(train_loader):
            x = x.to(device)  # (4,5,160,40)
            x = torch.reshape(x,(x.size(0)*x.size(1),x.size(2),x.size(3)))  # (4*5,160,40)
            optimizer.zero_grad()
            logits = net(x)  # (20,128)

            logits = torch.reshape(logits,(hp.train.batch_size,hp.train.num_utters,logits.size(1)))  # (4,5,128)


            loss = ge2e_loss(logits)
            loss.backward()
            # 梯度剪切，将梯度限制在一个阈值内，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(),1.0)
            optimizer.step()

            total_loss += loss

        print("Epoch:%d,Loss:%f"%(epoch+1,total_loss/len(train_loader)))

    # test
    if hp.data.data_preprocessed:
        test_db = TIMITPreprocessed(train=False)
    else:
        test_db = TIMITUnProcessed(train=False)
    test_loader = DataLoader(test_db, batch_size=hp.test.batch_size,shuffle=True,drop_last=True)

    net.eval()
    avg_EER = 0
    for i, x in enumerate(test_loader):

        x = x.to(device)  # (4,6,160,40)
        enroll_x, veri_x = torch.split(x, int(x.size(1)/2), dim=1)  # (4,3,160,40)
        enroll_x = torch.reshape(enroll_x,(enroll_x.size(0)*enroll_x.size(1),enroll_x.size(2),enroll_x.size(3)))
        veri_x = torch.reshape(veri_x, (veri_x.size(0) * veri_x.size(1), veri_x.size(2), veri_x.size(3)))

        enroll_logits = net(enroll_x)
        veri_logits = net(veri_x)

        enroll_logits = torch.reshape(enroll_logits,(hp.test.batch_size,hp.test.num_utters//2,enroll_logits.size(1)))
        veri_logits = torch.reshape(veri_logits, (hp.test.batch_size, hp.test.num_utters // 2, veri_logits.size(1)))

        enroll_centroids = get_centroids(enroll_logits)
        sim_matrix = get_cossim(veri_logits,enroll_centroids)

        # EER
        diff,threshold = 1,0
        EER_FAR,EER_FRR,EER = 0,0,0

        for thres in [0.01*i+0.5 for i in range(50)]:
            sim_matrix_thres = sim_matrix>thres

            FAR = (sum([sim_matrix_thres[i].float().sum() - sim_matrix_thres[i, :, i].float().sum() for i in range(int(hp.test.batch_size))])
                   / (hp.test.batch_size - 1.0) / (float(hp.test.num_utters / 2)) / hp.test.batch_size)

            FRR = (sum([hp.test.num_utters / 2 - sim_matrix_thres[i, :, i].float().sum() for i in range(int(hp.test.batch_size))])
                   / (float(hp.test.num_utters / 2)) / hp.test.batch_size)

            # Save threshold when FAR = FRR (=EER)
            if diff > abs(FAR - FRR):
                diff = abs(FAR - FRR)
                EER = (FAR + FRR) / 2
                threshold = thres
                EER_FAR = FAR
                EER_FRR = FRR
        avg_EER += EER
        print("EER:%f (threshod:%f,FAR:%f,FRR:%f)"%(EER,threshold,EER_FAR,EER_FRR))
    avg_EER = avg_EER/(len(test_loader))
    print("avg_EER:%f,"%avg_EER)


def test():
    pass


train()