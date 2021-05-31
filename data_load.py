#!/usr/bin/python3
"""
    Time    : 2021/4/13 20:24
    Author  : 春晓
    Software: PyCharm
"""

import glob
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from audio_process.mfcc_extract import mfcc_40, audio_load

class TimitDataset(Dataset):

    def __init__(self,train=True):
        if train:
            self.path = r'data/TIMIT/train_wav/*/*/*.WAV'
            self.filenames = glob.glob(self.path)
            self.framelen = 180
        else:
            self.path = r'data/TIMIT/test_wav/*/*/*.WAV'
            self.filenames = glob.glob(self.path)
        # self.speaker = glob.glob(os.path.dirname(self.path))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        audio,sr = audio_load(filename)
        mfcc = mfcc_40(audio,sr,winfunc=np.hamming)
        label = os.path.dirname(filename)[-5:]
        mfcc_db = torch.tensor(mfcc[:180])
        # label = torch.tensor(label)
        return [mfcc_db,label]

def my_collate(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return data, targets


if __name__ == '__main__':
    train_db = TimitDataset()
    # x,y = DataLoader(train_db)
    #
    # print(x.shape)
    # print(y)
    # print(len(train_db))
    # x,y = train_db[0]
    # print(x,type(x))
    # print(y,type(y))

    train_data,labels = DataLoader(train_db,shuffle=True,batch_size=10,collate_fn=my_collate(10))
    print(len(train_data),train_data.shape)
    print(labels)

    # test_db = TimitDataset(train=False)
    # x,y = test_db[0]
    # print(x,x.shape)
    # print(y)
    # print(len(test_db))

