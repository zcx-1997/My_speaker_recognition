# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> train_MLP
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/25 下午7:57
@Description        ：
==================================================
"""
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_process.timit_loader import TIMIT_Uttr, TIMIT_RAW
from models_MLP.mlp import MyMLP

#计算准确率
def sum_right(y_hat, y):
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum())

def accuracy(y_hat, y):
    """计算准确率。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum()) / len(y)


def train_epochs(net, train_loader, test_loader, epochs, lr,device):

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(),lr)
    net = net.to(device)

    for epoch in range(epochs):
        # train
        net.train()
        total_loss = 0
        total_acc = 0
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x[:,:40,:]
            x = x.reshape(x.size(0), -1)
            y_hat = net(x)
            loss = criterion(y_hat, y)
            opt.zero_grad()
            loss.backward()
            # 梯度裁剪,防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.0)
            opt.step()

            total_loss += loss
            total_acc += accuracy(y_hat, y)

        if (epoch+1) % 1 == 0:
            print("epoch{}: loss={:.5f}, acc={:.4f}".format(epoch+1,
                    total_loss/len(train_loader),total_acc/len(train_loader)))

        # eval
        net.eval()
        test_loss = 0
        test_acc, num = 0, 0
        for x,y in train_loader:
        # for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x[:,:40,:]
            x = x.reshape(x.size(0), -1)
            y_hat = net(x)
            loss = criterion(y_hat, y)

            test_loss += loss.item()
            test_acc += sum_right(y_hat,y)
            num += len(y)


        if (epoch+1) % 10 == 0:
            print("test: loss={:.5f}, acc={:.4f}".format(
                test_loss/len(test_loader),test_acc/num))


def predicts(net,x,y):
    ''' 单值预测 '''
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    x, y = x.to(device), y.to(device)
    x = x[:, :40, :]
    x = x.reshape(x.size(0), -1)
    y_hat = net(x)
    print("predict:",torch.argmax(y_hat))
    print("label:", y)
    if torch.argmax(y_hat) == y:
        return 1
    else:
        return 0



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on:", device)

    epochs = 50
    lr = 0.01
    batch_size =1

    train_db = TIMIT_Uttr()
    test_db = TIMIT_Uttr(train=False)
    train_loader = DataLoader(train_db, batch_size, shuffle=True)
    test_loader = DataLoader(test_db, batch_size, shuffle=True)
    # print(len(train_loader))
    # x, y = next(iter(train_loader))
    # print(x.shape)
    # print(y.shape)

    net = MyMLP(40*39).to(device)

    train_epochs(net,train_loader,test_loader,epochs,lr,device)

    num = 0
    for _ in range(1000):
        i = random.randint(1,4000)
        x, y = TIMIT_Uttr()[i]
        num +=predicts(net,x,y)
    print("acc:%f"%(num/1000))