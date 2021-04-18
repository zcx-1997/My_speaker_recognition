#!/usr/bin/python3
"""
    Time    : 2021/4/13 20:24
    Author  : 春晓
    Software: PyCharm
"""

import glob
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from audio_progress.mfcc_extract import mfcc_40, audio_load

class TimitDataset(Dataset):

    def __init__(self,train=True):
        if train:
            self.path = r'data/TIMIT/train_wav/*/*/*.WAV'
            self.filenames = glob.glob(self.path)
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

        return mfcc,label

if __name__ == '__main__':
    train_db = TimitDataset()
    x,y = DataLoader(train_db,batch_size=10)

    print(x.shape)
    print(y)
    print(len(train_db))

    test_db = TimitDataset(train=False)
    x,y = test_db[0]
    print(x,x.shape)
    print(y)
    print(len(test_db))

