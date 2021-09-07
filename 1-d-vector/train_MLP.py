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
import os
import time

import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import constants as c
from data_loader import TIMITDataset_MLP, TIMITDataset_Veri
from model import MyMLP

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

    print("load data...")
    train_db = TIMITDataset_MLP()
    train_loader = DataLoader(train_db, batch_size=c.batch_train, shuffle=True,drop_last=True)
    print('data ok')

    net = MyMLP(c.context * c.nmels, c.train_num_spks)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), c.lr)
    writer = SummaryWriter()

    print("start train ...")
    for epoch in range(c.train_epochs):
        net.train()
        total_loss, rights, num_sample = 0, 0 ,0
        for step_id, (x, y) in enumerate(train_loader):
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.to(device)
            y_hat = net(x)
            loss = criterion(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("step_loss", loss, epoch * step_id)

            total_loss += loss
            rights += sum_right(y_hat, y)
            num_sample += len(y)

        avg_loss = total_loss / len(train_loader)
        acc = rights / num_sample
        writer.add_scalar("epoch_loss", avg_loss, epoch)
        writer.add_scalar("train_acc", acc, epoch)

        if (epoch + 1) % 1 == 0:
            message = "Epoch{}, avg_loss={:.4f}, acc={:.4f}, time={}".format(epoch + 1,
                                                                 avg_loss,acc,
                                                                 time.ctime())
            print(message)
            if checkpoint_dir is not None:
                with open(log_file, 'a') as f:
                    f.write(message+'\n')

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
    net = MyMLP(c.context * c.nmels, c.train_num_spks)
    net.load_state_dict(torch.load(model_path))
    data = TIMITDataset_Veri()
    num_right = 0

    for epoch in range(c.test_epochs):
        data = TIMITDataset_Veri()
        for i in range(len(data)):
            enroll_data, veri_data = data[i]
            if len(enroll_data.shape) == 3:
                enroll_data = enroll_data.reshape(enroll_data.shape[0], -1)
                veri_data = veri_data.reshape(-1)
                e_embedding = net(enroll_data).mean(dim=0)
                v_embedding = net(veri_data)
                cossim = torch.cosine_similarity(e_embedding, v_embedding, dim=0)
                if cossim > 0.5:
                    num_right += 1
    acc = num_right / (c.test_epochs*len(data))
    return acc

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on: ", device)
    train(device)
