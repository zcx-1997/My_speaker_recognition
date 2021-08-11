# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> train_timit
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/20 下午9:50
@Description        ：
==================================================
"""
import torch
import torchsummary
from torch import nn
from torch.utils.data import DataLoader
from data_process.timit_loader import TIMIT_Uttr, TIMIT_Frame,TIMIT_RAW
from models_RNN.rnn import MyRNN
from models_MLP.mlp import MyMLP
from models_RNN.lstm import MyLSTM

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
            x = torch.reshape(x, (x.size(0) * x.size(1), x.size(2), x.size(3)))
            y = torch.reshape(y, (y.size(0) * y.size(1),))
            # x = x.reshape(x.shape[0], -1)
            # TIMIT_Frame
            # x, y = x.to(device), y.to(device)
            embedding, y_hat = net(x)
            loss = criterion(y_hat, y)
            opt.zero_grad()
            loss.backward()
            # 梯度裁剪,防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.0)
            opt.step()

            total_loss += loss.item()
            total_acc += accuracy(y_hat, y)

        if (epoch+1) % 1 == 0:
            print("epoch{}: loss={:.5f}, acc={:.4f}".format(epoch+1,
                    total_loss/len(train_loader),total_acc/len(train_loader)))

        # eval
        net.eval()
        test_loss = 0
        test_acc = 0
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            x = torch.reshape(x, (x.size(0) * x.size(1), x.size(2), x.size(3)))
            y = torch.reshape(y, (y.size(0) * y.size(1),))
            embedding, y_hat = net(x)
            loss = criterion(y_hat, y)

            test_loss += loss.item()
            test_acc += accuracy(y_hat,y)

        if (epoch+1) % 10 == 0:
            print("test: loss={:.5f}, acc={:.4f}".format(
                test_loss/len(test_loader),test_acc/len(test_loader)))


def predicts(net,x,y):
    if len(x.size()) == 2:
        x = x.unsqueeze(0)
    x, y = x.to(device), y.to(device)
    embedding, y_hat = net(x)
    print("predict:",torch.argmax(y_hat))
    print("label:", y)



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on:", device)

    epochs = 100
    lr = 0.01
    batch_size = 4

    train_db = TIMIT_RAW()
    test_db = TIMIT_RAW(train=False)
    train_loader = DataLoader(train_db, batch_size, shuffle=True)
    test_loader = DataLoader(test_db, batch_size, shuffle=True)

    # train_db = TIMIT_Frame()
    # test_db = TIMIT_Frame(train=False)
    # train_loader = DataLoader(train_db, batch_size, shuffle=True)
    # test_loader = DataLoader(test_db, batch_size, shuffle=True)

    # net = MyMLP(39*299)
    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init;.normal_(m.weight, std=0.01)
    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init.xavier_uniform_(m.weight)

    # net.apply(init_weights)

    print("============ train ... =============")
    net = MyLSTM(13,64,2)
    train_epochs(net,train_loader,test_loader, epochs,lr,device)

    # x,y  = next(iter(train_loader))  #  (b, f, 39) (b,)
    # y = y.unsqueeze(1).expand(-1, x.shape[1]).reshape(-1)
    # # (b,)-->(b,1)-->(b,f)-->(b*f,)
    # y = y.to(device)
    # x = x.reshape(-1, x.shape[-1]).to(device)  # (b*f,39)

    print("============ predict =============")
    x, y = train_db[100]
    x, y = x[0], y[0]
    predicts(net,x,y)

    x, y = test_db[44]
    x, y = x[0], y[0]
    predicts(net, x, y)
