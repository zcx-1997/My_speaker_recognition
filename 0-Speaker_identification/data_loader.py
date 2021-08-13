# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> data_loader
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/13 上午8:58
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

import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader

import constants as c

class SI_Dataset(Dataset):
    ''' return features(shape:[40,40]) and label(shape:[]) '''
    def __init__(self,train=True):
        self.train = train
        if self.train:
            self.root_dir = r'data/train_si'
        else:
            self.root_dir = r'data/test_si'
        self.shuffle = random.shuffle
        self.spks_list = os.listdir(self.root_dir)

        self.shuffle(self.spks_list)
        self.frames_lables = []
        for spk_name in self.spks_list:
            spk_id = int(spk_name[:-4])
            spk_path = os.path.join(self.root_dir, spk_name)
            spk_feats = np.load(spk_path)
            spk_feats = spk_feats[:, :c.num_frames]

            indices = torch.arange(0, 160, 40)

            for utter in spk_feats:
                for i in indices:
                    frame = utter[i:i + 40]
                    feat = torch.tensor(frame, dtype=torch.float32)
                    label = torch.tensor(spk_id, dtype=torch.float32)
                    self.frames_lables.append((feat, label))

    def __getitem__(self, idx):
        feat, label = self.frames_lables[idx]
        return feat, label  # torch.Size([40, 40])  torch.Size([])

    def __len__(self):  # 18480
        # lens = len(self.spks_list) * num_utters * 4
        if self.train:
            num_spks = len(self.spks_list)
            num_utters = num_spks * c.num_utters_trian
            lens = num_utters * int(c.num_frames / 40)
        else:
            num_spks = len(self.spks_list)
            num_utters = num_spks * c.num_utters_test
            lens = num_utters * int(c.num_frames / 40)
        return lens


if __name__ == '__main__':
    train_db = SI_Dataset()
    print(len(train_db))
    feat, label = train_db[0]
    print(feat.shape)
    print(label)

    feat, label = train_db[1]
    print(feat.shape)
    print(label.shape)

    train_loader = DataLoader(train_db,batch_size=64,shuffle=True,drop_last=True)
    print(len(train_loader))

    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)

    print("=========================")
    test_db = SI_Dataset(train=False)
    print(len(test_db))
    feat, label = test_db[0]
    print(feat.shape)
    print(label)

    test_loader = DataLoader(test_db,batch_size=64,shuffle=True,drop_last=True)
    print(len(test_loader))

    x, y = next(iter(test_loader))
    print(x.shape)
    print(y.shape)