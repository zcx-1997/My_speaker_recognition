# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> train
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/11 下午3:59
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
import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import constants as c
from data_loader import MLPTrianDataset
from model import MyMLP

def accuracy(y_hat, y):
    """计算准确率。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum()) / len(y)

def train(device):
    epochs = 1000
    lr = 0.001
    print("start")
    print("load data...")
    train_db = MLPTrianDataset()
    train_loader = DataLoader(train_db, batch_size=64, shuffle=True,
                              drop_last=True)

    print('data ok')
    net = MyMLP(40 * 40, 462)
    net = net.to(device)
    print("net ok")

    centerion = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr)

    writer = SummaryWriter()
    print("start train ...")
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        total_acc = 0
        for step_id, (x, y) in enumerate(train_loader):
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.to(device)
            y_hat = net(x)
            loss = centerion(y_hat, y.long())
            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("loss", loss, epoch*step_id)

            total_loss += loss
            total_acc += accuracy(y_hat, y)

        avg_loss = total_loss/len(train_loader)
        avg_acc = total_acc/len(train_loader)
        writer.add_scalar("avg_loss",avg_loss, epoch)
        writer.add_scalar("avg_acc", avg_acc, epoch)
        if (epoch+1) % 1 == 0:
            print("Epoch{}, avg_loss={:.5f}".format(epoch+1,avg_loss))

    writer.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on: ", device)
    train(device)
