# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> data_loader
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/20 下午8:19
@Description        ：
==================================================
"""
import glob
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self,train=True):
        train_path = r'../data/TIMIT_TI/ti_train'
        test_path = r'../data/TIMIT_TI/ti_test'
        if train:
            self.path = train_path
        else:
            self.path = test_path
        self.speaker = os.listdir(self.path)
        self.features = []
        # for spk in self.speaker:
        #     au_paths = os.listdir(os.path.join(self.path,spk))

    def __getitem__(self, item):
        item = str(item+1)
        if item in self.speaker:
            au_paths = glob.glob(os.path.join(self.path,item,'*.npy'))
            features,labels = [], []
            for path in au_paths:
                feature = np.load(path)
                label = int(item)
                features.append(feature)
                labels.append(label)
            # sample = {'features':features,'label':label}
        return torch.tensor(features,dtype=torch.float32), torch.tensor(labels)

    def __len__(self):
        return len(self.speaker)


if __name__ == '__main__':
    mydataset = MyDataset()
    x, y = mydataset[0]
    print(x.shape)
    print(y.shape)
    print(y)
    train_loader = DataLoader(mydataset,batch_size=2, shuffle=True)
    print(len(train_loader))
    x,y = next(iter(train_loader))
    print(x.shape)
    print(x.dtype)
    print(y.shape)
    print(y)

    testData = MyDataset(train=False)
    test_loader = DataLoader(testData,batch_size=2,shuffle=True)
    x,y = next(iter(test_loader))
    print(x.shape)
    print(y.shape)
    print(y)
