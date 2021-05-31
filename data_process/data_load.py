#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/24 15:58
    Author  : 春晓
    Software: PyCharm
"""

import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils.utils import mfccs_and_spec
from audio_process.ectract_speech_features import extract_mfcc,wav_load

class TIMITUnProcessed(Dataset):

    def __init__(self,train=hp.training):
        if train:
            self.path = hp.data.train_path_unprocessed
            self.num_utters = hp.train.num_utters
        else:
            self.path = hp.data.test_path_unprocessed
            self.num_utters = hp.test.num_utters
        self.speakers = glob.glob(os.path.dirname(self.path))
        random.shuffle(self.speakers)

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*.wav')
        random.shuffle(wav_files)
        wav_files = wav_files[:self.num_utters]

        utters_features = []
        for f in wav_files:
            audio,sr = wav_load(f)
            mfcc_features = extract_mfcc(audio,sr)
            mfcc_features = mfcc_features[:160]
            # _,mel_db,_ = mfccs_and_spec(f,wav_process = True)
            utters_features.append(mfcc_features)
        return torch.Tensor(utters_features)  # fbank


class TIMITPreprocessed(Dataset):

    def __init__(self,train=hp.training,shuffle=True,utter_start=0):

        if train:
            self.path = hp.data.train_path
            self.num_utters = hp.train.num_utters
        else:
            self.path = hp.data.test_path
            self.num_utters = hp.test.num_utters

        self.file_list = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.shuffle:
            selected_file = random.sample(self.file_list,1)[0]  # 字符串，无[0]是只包含一个字符串的数组
        else:
            selected_file = self.file_list[idx]

        utters = np.load(os.path.join(self.path,selected_file))

        if self.shuffle:
            utter_idx = np.random.randint(0,utters.shape[0],self.num_utters) # 返回num_utters个[0,utters.shape[0]]的值
            utterances = utters[utter_idx]
        else:
            utterances = utters[self.utter_start:self.utter_start+self.num_utters]

        # utterances = utterances[:,:,:160]
        # utterances = torch.Tensor(np.transpose(utterances,axes=(0,2,1)))
        return utterances


if __name__ == '__main__':

    print('TIMIT-TRAIN')
    timit_train = TIMITUnProcessed()
    print(len(timit_train))
    print(timit_train[0].shape)
    print(timit_train[1].shape)
    print(timit_train[2].shape)

    print('TIMIT-TEST')
    timit_test = TIMITUnProcessed(train=False)
    print(len(timit_test))
    print(timit_test[0].shape)
    print(timit_test[1].shape)
    print(timit_test[2].shape)

    print('TIMIT-P-TRAIN')
    timit_p_train = TIMITPreprocessed()
    print(len(timit_p_train),timit_p_train[0].shape)

    print('\nTIMIT-P-TEST')
    timit_p_test = TIMITPreprocessed(train=False)
    print(len(timit_p_test),timit_p_test[0].shape)