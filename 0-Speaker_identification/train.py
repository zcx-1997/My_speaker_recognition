# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> train
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/13 上午9:18
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
from data_loader import SI_Dataset
from model import MyMLP


# 计算准确率
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
    log_file = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)

    epochs = 1000
    lr = 0.001
    batch_size = 64

    print("load data...")
    train_db = SI_Dataset()
    train_loader = DataLoader(train_db, batch_size, shuffle=True,
                              drop_last=True)
    test_db = SI_Dataset(train=False)
    test_loader = DataLoader(test_db, batch_size, shuffle=True,
                             drop_last=True)
    print('data ok')

    net = MyMLP(40 * 40, 630)
    net = net.to(device)
    print("net ok")

    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)
    writer = SummaryWriter()

    print("start train ...")
    for epoch in range(epochs):

        net.train()
        total_loss, train_rights, train_num_simple = 0, 0, 0
        for step_id, (x, y) in enumerate(train_loader):
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.to(device)
            y_hat = net(x)
            loss = criterion(y_hat, y.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            # scheduler.step()

            writer.add_scalar("train_step_loss", loss, epoch * step_id)

            total_loss += loss
            train_rights += sum_right(y_hat, y)
            train_num_simple += len(y)

        train_loss = total_loss / len(train_loader)
        train_acc = train_rights / train_num_simple
        writer.add_scalar("train_avg_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)

        net.eval()
        total_loss, test_rights, test_num_simple = 0, 0, 0
        for step_id, (x, y) in enumerate(test_loader):
            x = x.reshape(x.shape[0], -1).to(device)
            y = y.to(device)
            y_hat = net(x)
            loss = criterion(y_hat, y.long())
            writer.add_scalar("test_step_loss", loss, epoch * step_id)
            total_loss += loss
            test_rights += sum_right(y_hat, y)
            test_num_simple += len(y)

        test_loss = total_loss / len(test_loader)
        test_acc = test_rights / test_num_simple
        writer.add_scalar("test_avg_loss", test_loss, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)

        if (epoch + 1) % 10 == 0:
            message = "Epoch{}, train_loss={:.4f}, train_acc={:.4f}, time={}\n test_loss={:.4f},test_acc={:4f}".format(
                epoch + 1, train_loss, train_acc, time.ctime(), test_loss,
                test_acc)
            print(message)
            if checkpoint_dir is not None:
                with open(log_file, 'a') as f:
                    f.write(message + '\n')

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


def predict(net, x, y):
    x = x.reshape(-1)
    print(x.shape)
    y_hat = net(x)
    print("predict:", torch.argmax(y_hat))
    print("label:", y.long())


def predicts(net, x, y):
    x = x.reshape(-1)
    y_hat = net(x)
    if torch.argmax(y_hat) == y.long():
        return 1
    else:
        return 0


def test(model_path, dataset):
    net = MyMLP(40 * 40, 630)
    # net = net.to(device)
    net.load_state_dict(torch.load(model_path))
    i = random.randint(1, len(dataset))
    x, y = dataset[i]
    print(x.shape)
    predict(net, x, y)
    #
    # num = 0
    # for _ in range(1000):
    #     i = random.randint(1, len(dataset))
    #     x, y = dataset[i]
    #     print(x.shape)
    #     num += predicts(net, x, y)
    # print("acc:%f" % (num / 1000))


if __name__ == '__main__':
    print("=============== trian ===================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on: ", device)
    train(device)
    # 训练集上效果好（0.95），在测试集上效果差（0.20）

    # print("=============== test ===================")
    # model_path = r'checkpoints/final_epoch_1000.model'
    # print('train-dataset')
    # train_data = SI_Dataset()
    # net = MyMLP(40 * 40, 630)
    # # net = net.to(device)
    # net.load_state_dict(torch.load(model_path))
    # x, y = train_data[0]
    # predict(net, x, y)
    #
    # test(model_path, train_data)
    #
    # print('test-dataset')
    # test_data = SI_Dataset(train=False)
    # test(model_path, test_data)
