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
        spk_feats = np.load(os.path.join(self.path, self.spk_name))
        spk_id = int(self.spk_name[:-4])
        if self.shuffle:
            indices = random.sample(range(0, 10), 10)
            spk_feats = spk_feats[indices]
        else:
            spk_feats = spk_feats

        if self.train:
            spk_feats = spk_feats[:, :self.num_frames]
            spk_feats = torch.tensor(spk_feats, dtype=torch.float32)
            label = torch.tensor(spk_id)
            return spk_feats, label
        else:
            spk_feats = spk_feats[:, :self.num_frames]
            spk_feats = torch.tensor(spk_feats, dtype=torch.float32)
            indices = random.sample(range(0, 10), 8)
            enroll_feats = spk_feats[indices[1:]]
            veri_feats = spk_feats[indices[0]]
            return enroll_feats, veri_feats


    def __len__(self):
        return len(self.spks_list)

class TIMITDataset_Veri(Dataset):
    def __init__(self):
        self.root_dir = r'data/test_tisv'
        self.spks_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.spks_list)

    def __getitem__(self, idx):
        spk = self.spks_list[idx]
        spk_path = os.path.join(self.root_dir, spk)
        spk_feats = np.load(spk_path)
        spk_feats = spk_feats[:, 100:140]
        spk_feats = torch.tensor(spk_feats, dtype=torch.float32)
        indices = random.sample(range(0, 10), 8)
        enroll_feats = spk_feats[indices[1:]]
        veri_feats = spk_feats[indices[0]]
        return  enroll_feats, veri_feats



if __name__ == '__main__':
    print("=================== TRAIN ==================")
    trian_db = TIMITDataset()
    print(len(trian_db))
    train_x, train_y = trian_db[0]
    print(train_x.shape)
    print(train_y)

    train_loader = DataLoader(trian_db,batch_size=4,shuffle=True,drop_last=True)
    x,y = next(iter(train_loader))
    print(x.shape)
    print(y)

    print("=================== TEST ==================")
    test_db = TIMITDataset(train=False)
    print(len(test_db))
    enroll_data, veri_data = test_db[0]
    print(enroll_data.shape)
    print(veri_data.shape)

    test_loader = DataLoader(test_db,batch_size=4,shuffle=True,drop_last=True)
    enroll_data, veri_data = next(iter(test_loader))
    print(enroll_data.shape)
    print(veri_data.shape)

