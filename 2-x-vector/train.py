# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> train
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/1 下午9:39
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
import os
import time

import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import constants as c
from data_loader import TIMITDataset

from tdnn import TDNN

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


def train(device):

    checkpoint_dir = r'checkpoints'
    log_file = os.path.join(checkpoint_dir,'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)

    epochs = 200
    lr = 0.1

    train_db = TIMITDataset()
    train_loader = DataLoader(train_db, batch_size=64, shuffle=True,
                              drop_last=True)

    tdnn1 = TDNN(input_dim=40,output_dim=128, context_size=5, dilation=1)
    tdnn2 = TDNN(input_dim=128,output_dim=256,context_size=3,dilation=2)

    net = nn.Sequential(tdnn1,tdnn2,nn.Linear(256, 462))
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr)
    writer = SummaryWriter()

    print("start train ...")
    for epoch in range(epochs):
        print("train:")
        net.train()
        total_loss, rights, num_sample = 0, 0 ,0
        for step_id, (x, y) in enumerate(train_loader):
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3]).to(device)
            y = y.reshape(64, 1).expand(64, 10).reshape(-1)
            y = y.to(device)
            y_hat = net(x)
            y_hat = y_hat.mean(dim=1)

            loss = criterion(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("loss", loss, epoch * step_id)

            total_loss += loss
            rights += sum_right(y_hat, y)
            num_sample += len(y)

        avg_loss = total_loss / len(train_loader)
        acc = rights / num_sample
        writer.add_scalar("avg_loss", avg_loss, epoch)
        writer.add_scalar("avg_acc", acc, epoch)

        if (epoch + 1) % 1 == 0:
            message = "Epoch{}, avg_loss={:.4f}, acc={:.4f}, time={}".format(epoch + 1,
                                                                 avg_loss,acc,
                                                                 time.ctime())
            print(message)
            if checkpoint_dir is not None:
                with open(log_file, 'a') as f:
                    f.write(message+'\n')

        print("eval:")
        net.eval()
        test_db = TIMITDataset(train=False)
        test_loader = DataLoader(test_db, batch_size=64, shuffle=True,drop_last=True)
        veri_acc, rights, num_sample = 0, 0, 0
        for step_id, (enroll_data, veri_data) in enumerate(test_loader):
            enroll_data = enroll_data.reshape(enroll_data.shape[0] * enroll_data.shape[1], enroll_data.shape[2], enroll_data.shape[3]).to(device)
            print(enroll_data.shape)
            veri_data = veri_data.reshape(veri_data.shape[0], veri_data.shape[1], veri_data.shape[2]).to(device)
            # enroll_embeddings = net(enroll_data)
            # print(enroll_data.shape)
            enroll_embeddings = net(enroll_data).mean(dim=1).reshape(64,7,462).mean(dim=1)

            veri_embeddings = net(veri_data).mean(dim=1)
            cossims = torch.cosine_similarity(enroll_embeddings, veri_embeddings,dim=1)
            for cossim in cossims:
                if cossim > 0.5:
                    rights += 1
                    num_sample += len(veri_data)
        acc = rights /num_sample
        print("acc={:.5f}".format(acc))


        if checkpoint_dir is not None and (epoch + 1) % 100 == 0:
            net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(epoch + 1) + ".pth"
            ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
            torch.save(net.state_dict(), ckpt_model_path)
            net.to(device).train()
    writer.close()

    # save model
    net.eval().cpu()
    save_model_filename = "final_epoch_" + str(epoch + 1) + ".model"
    save_model_path = os.path.join(checkpoint_dir, save_model_filename)
    torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

def test(model_path):
    tdnn1 = TDNN(input_dim=40, output_dim=128, context_size=5, dilation=1)
    tdnn2 = TDNN(input_dim=128, output_dim=256, context_size=3, dilation=2)

    net = nn.Sequential(tdnn1, tdnn2, nn.Linear(256, 462))
    net.load_state_dict(torch.load(model_path))
    data = TIMITDataset(train=False)
    num_right = 0

    for _ in range(10000):
        i = random.randint(1, len(data)-1)
        enroll_data, test_data = data[i]
        test_data = test_data.unsqueeze(0)
        enroll_embedding = net(enroll_data).mean(dim=1).mean(dim=0)
        # print(enroll_embedding.shape)  #torch.Size([462])
        test_embedding = net(test_data).mean(dim=1).mean(dim=0)
        print(test_embedding.shape)
        cossim = torch.cosine_similarity(enroll_embedding,test_embedding, dim=0)

        if cossim > 0.5:
            num_right += 1
    return num_right / 10000

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Training on ", device)
    # train(device)

    model_path = r'./checkpoints/final_epoch_200.model'
    acc = test(model_path)
    print("final_acc=%f "% acc)