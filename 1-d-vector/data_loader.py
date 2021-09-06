# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> data_loader
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/11 上午11:21
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


class TIMITDataset(Dataset):
    def __init__(self):
        self.root_dir = r'data/train_tisv'
        self.shuffle = random.shuffle
        self.spks_list = os.listdir(self.root_dir)
        self.shuffle(self.spks_list)
        self.frames_labels = []
        self._load_data()

    def __getitem__(self, idx):
        feat, label = self.frames_labels[idx]
        return feat, label  # torch.Size([40, 40])  torch.Size([])

    def __len__(self):
        lens = len(self.frames_labels)
        # num_spks = len(self.spks_list)
        # num_utters = num_spks * c.num_utters_trian
        # lens = num_utters * int(c.num_frames / 40)
        return lens

    def _load_data(self):
        for spk_name in self.spks_list:
            spk_id = int(spk_name[:-4])
            spk_path = os.path.join(self.root_dir, spk_name)
            spk_feats = np.load(spk_path)
            spk_feats = spk_feats[:, :c.num_frames]

            for utter_feats in spk_feats:
                for i, frame_feat in enumerate(utter_feats):
                    if i >= 30 and i < len(utter_feats)-10:
                        stacked_frame_feat = utter_feats[i-30:i+11]
                        label = spk_id
                        self.frames_labels.append((stacked_frame_feat, label))


            # indices = torch.arange(0, 160, 40)
            #
            # for utter in spk_feats:
            #     for i in indices:
            #         frame = utter[i:i + 40]
            #         feat = torch.tensor(frame, dtype=torch.float32)
            #         label = torch.tensor(spk_id).long()
            #         self.frames_lables.append((feat, label))

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
        return  spk_feats[indices[1:]], spk_feats[indices[0]]



if __name__ == '__main__':
    data = TIMITDataset()
    print(len(data))
    feat, label = data[0]
    print(feat.shape)
    print(label)

    feat, label = data[1]
    print(feat.shape)
    print(label)

    data_loader = DataLoader(data,batch_size=64,shuffle=True,drop_last=True)
    print(len(data_loader))

    x, y = next(iter(data_loader))
    print(x.shape)
    print(y)
    print(y.shape)

    # testdata = TIMITDataset_Veri()
    # enroll, test = testdata[0]
    # print(enroll.shape)
    # print(test.shape)
