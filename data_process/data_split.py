# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> data_spkit
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/7/20 下午2:37
@Description        ：
==================================================
"""

import os
import glob
import numpy as np

import torch
import torchaudio
from python_speech_features import logfbank, mfcc, delta


def TIMIT_ti(src_dir,tar_dir,train=True):

    spks = glob.glob(os.path.dirname(src_dir))
    i = 0
    for spk in spks:
        i += 1
        print("{}th speaker process...".format(i))
        train_dir = os.path.join(tar_dir, 'ti_train', str(i))
        test_dir = os.path.join(tar_dir,'ti_test', str(i))
        os.makedirs(train_dir,exist_ok=True)
        os.makedirs(test_dir,exist_ok=True)
        audio_paths = glob.glob(os.path.join(spk, '*.WAV'))
        if train:
            audio_paths = audio_paths[:8]
        else:
            audio_paths = audio_paths[8:]
        j = 0
        for audio_path in audio_paths:
            j += 1
            audio, sr = torchaudio.load(audio_path)

            time = audio.shape[1] / sr
            if time >= 3:
                audio = audio[:, 0:sr * 3].reshape(-1, sr * 3)
                time = audio.shape[1] / sr
            else:
                zeros = torch.zeros(sr * 3 - audio.shape[1]).reshape(1, -1)
                audio = torch.cat((audio, zeros), dim=1)
                time = audio.shape[1] / sr
            # print('time', time)  # 3
            mfcc_f = mfcc(audio, sr, appendEnergy=True)
            if train:
                np.save(train_dir + '/feature' + str(j) + '.npy', mfcc_f)
            else:
                np.save(test_dir + '/feature' + str(j) + '.npy', mfcc_f)
    print("done")

if __name__ == '__main__':
    src_dir = r'../data/TIMIT/*/*/*/*.WAV'
    tar_dir = r'../data/TIMIT_TI/'

    print("ti train dataset ...")
    TIMIT_ti(src_dir,tar_dir)

    print("ti test dataset ...")
    TIMIT_ti(src_dir,tar_dir,train=False)
