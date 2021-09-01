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
import random
import torch
import numpy as np
import torchaudio
from python_speech_features import logfbank, mfcc
from torch.utils.data import Dataset, DataLoader

class TIMIT_Uttr(Dataset):
    def __init__(self,train=True):
        self.train = train
        train_path = r'../data/TIMIT_SI/train_si'
        test_path = r'../data/TIMIT_SI/test_si'
        if train:
            self.path = train_path
        else:
            self.path = test_path
        self.speaker = os.listdir(self.path)

    def __getitem__(self, idx):
        if self.train:
            # speaker:630,  utt_spk:7
            spk_id = str(idx // 7)
            utt_id = int(idx % 7)
        else:
            # speaker:630,  utt_spk:3
            spk_id = str(idx // 3)
            utt_id = int(idx % 3)

        if spk_id in self.speaker:
            au_paths = glob.glob(os.path.join(self.path,spk_id,'*.npy'))
            au_path = au_paths[utt_id]
            feature = np.load(au_path)
            label = int(spk_id)
        return torch.tensor(feature,dtype=torch.float32), torch.tensor(label)

    def __len__(self):
        return len(self.speaker)


class TIMIT_Frame(Dataset):
    def __init__(self,train=True):
        self.train = train
        train_path = r'../data/TIMIT_SI/train_si'
        test_path = r'../data/TIMIT_SI/test_si'
        if train:
            self.path = train_path
        else:
            self.path = test_path
        self.speaker = os.listdir(self.path)

    def __getitem__(self, idx):
        utter_id = int(idx // 299)
        frame_id = int(idx % 299)
        audio_paths = glob.glob(os.path.join(self.path,'*','*.npy'))
        random.shuffle(audio_paths)
        au_path = audio_paths[utter_id]
        features = np.load(au_path)
        spk = os.path.dirname(au_path)
        feature = features[frame_id]
        label = int(os.path.split(spk)[-1])
        return torch.tensor(feature,dtype=torch.float32), torch.tensor(label)

    def __len__(self):
        return len(self.speaker)


class TIMIT_RAW(Dataset):
    '''直接从TIMIT原数据集（解压后）中读取数据'''
    def __init__(self,train=True):
        if train:
            self.path = r'../data/TIMIT/train_wav/*/*/*.WAV'
        else:
            self.path = r'../data/TIMIT/test_wav/*/*/*.WAV'
        self.spks = glob.glob(os.path.dirname(self.path))
        random.shuffle(self.spks)

    def __len__(self):
        ''' 返回说话人的个数 '''
        return len(self.spks)

    def __getitem__(self, idx):
        spk = self.spks[idx]
        wav_files = glob.glob(spk+'/*.WAV')
        random.shuffle(wav_files)

        features, labels = [], []
        for wav_path in wav_files:
            audio, sr = torchaudio.load(wav_path)
            # 将语音长度固定成3秒
            time = audio.shape[1] / sr
            if time >= 3:
                audio = audio[:, 0:sr * 3].reshape(-1, sr * 3)
            else:
                zeros = torch.zeros(sr * 3 - audio.shape[1]).reshape(1, -1)
                audio = torch.cat((audio, zeros), dim=1)
            # fbank_feat = logfbank(audio,sr)
            mfcc_feat = mfcc(audio,sr)[:160]
            label = idx

            features.append(mfcc_feat)
            labels.append(label)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels)


if __name__ == '__main__':

    rawData_train = TIMIT_RAW()
    print(len(rawData_train))  #462
    x,y  = rawData_train[0]
    print(rawData_train[0][0].shape)  # x torch.Size([10, 299, 13])
    print(rawData_train[0][1].shape)  # y torch.Size([10])

    train_loader = DataLoader(rawData_train, batch_size=6, shuffle=True)
    print(len(train_loader))
    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
    print(y)

    rawData_train = TIMIT_Uttr()
    print(len(rawData_train))  #462
    x,y  = rawData_train[0]
    print(rawData_train[0][0].shape)  # x torch.Size([10, 299, 13])
    print(rawData_train[0][1].shape)  # y torch.Size([10])

    train_loader = DataLoader(rawData_train, batch_size=6, shuffle=True)
    print(len(train_loader))
    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
    print(y)




''' 
print("================= train data ===================")
    trainData = TIMIT_Uttr()
    x, y = trainData[7]
    print(x.shape)
    print(y.shape)
    # print(x)
    print(y)
    train_loader = DataLoader(trainData, batch_size=2, shuffle=True)
    print(len(train_loader))
    x,y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
    print(y)

    print("================= test data ===================")
    testData = TIMIT_Uttr(train=False)
    x,y = testData[3]
    print(x.shape)
    print(y.shape)
    # print(x)
    print(y)
    test_loader = DataLoader(testData,batch_size=2,shuffle=True)
    x,y = next(iter(test_loader))
    print(x.shape)
    print(y.shape)
    print(y)

    print("==============0-TIMIT frame================")
    frameData = TIMIT_Frame()
    x,y = frameData[0]
    print(x.shape)
    print(y.shape)
    print(y)
    frame_loader = DataLoader(frameData, batch_size=256,shuffle=True)
    x,y = next(iter(frame_loader))
    print(x.shape)
    print(y.shape)
    print(y)
'''





