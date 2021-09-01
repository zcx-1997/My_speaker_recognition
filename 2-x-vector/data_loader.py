# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> data_loader
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/1 下午7:25
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
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class TIMITDataset(Dataset):

    def __init__(self, train=True, num_frames=160, shuffle=True):

        self.train = train
        self.num_frames = num_frames
        self.shuffle = shuffle
        if self.train:
            self.path = r'../0-TIMIT/train'
        else:
            self.path = r'../0-TIMIT/test'
        self.spks_list = os.listdir(self.path)

    def __getitem__(self, idx):
        self.spk_name = self.spks_list[idx]
        np_spk = np.load(os.path.join(self.path, self.spk_name))

        if self.shuffle:
            indices = random.sample(range(0, 10), 10)
            utters = np_spk[indices]
        else:
            utters = np_spk
        utters = utters[:, :self.num_frames]
        utters = torch.tensor(utters, dtype=torch.float32)
        return utters

    def __len__(self):
        return len(self.spks_list)

if __name__ == '__main__':
    trian_db = TIMITDataset()
    print(len(trian_db))
    train_x = trian_db[0]
    print(train_x.shape)