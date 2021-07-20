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
from torch import nn
from torch.utils.data import DataLoader
from data_process.timit_loader import MyDataset

epochs = 10
lr = 0.1
batch_size = 16

net = nn.Sequential(
    nn.Linear(10*13, 128),nn.ReLU(),
    nn.Linear(128,256),nn.ReLU(),
    nn.Linear(256,630)
)

loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(),lr)

def accuracy(y_hat, y):
    """计算准确率。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 结果是一个包含0（错）和1（对）的张量
    return float(cmp.type(y.dtype).sum()) / len(y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on:", device)
net = net.to(device)

train_db = MyDataset()
test_db = MyDataset(train=False)
train_loader = DataLoader(train_db,batch_size,shuffle=True)
test_loader = DataLoader(test_db,batch_size,shuffle=True)
for epoch in range(epochs):
    total_loss = 0
    total_acc = 0
    for x, y in train_loader:
        x = x[:,:,:10,:]
        x = x.reshape(x.shape[0]*x.shape[1],-1).to(device)  #torch.Size([128, 3887])
        y = y.reshape(y.shape[0]*y.shape[1]).to(device)  #torch.Size([128])
        y = y-1
        logits = net(x)
        l = loss(logits,y)
        opt.zero_grad()
        l.backward()
        opt.step()

        total_acc += accuracy(net(x),y)
        total_loss += l
    print("Epoch{}, loss={:.5f}, acc={:.5f}".format((epoch+1),total_loss/len(train_loader),
                                            total_acc/len(train_loader)))