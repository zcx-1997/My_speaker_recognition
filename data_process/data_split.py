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


def TIMIT_SI(src_dir, tar_dir, train=True):
    spks = glob.glob(os.path.dirname(src_dir))
    i = 0
    for spk in spks:
        print("{}th speaker process...".format(i))
        train_dir = os.path.join(tar_dir, 'si_train', str(i))
        test_dir = os.path.join(tar_dir, 'si_test', str(i))
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        audio_paths = glob.glob(os.path.join(spk, '*.WAV'))
        if train:
            audio_paths = audio_paths[:7]
        else:
            audio_paths = audio_paths[7:]
        j = 0
        for audio_path in audio_paths:
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
            mfcc_13 = mfcc(audio, sr, appendEnergy=True)
            mfcc_d1 = delta(mfcc_13, 1)
            mfcc_d2 = delta(mfcc_13, 2)
            mfcc_f = np.hstack((mfcc_13, mfcc_d1, mfcc_d2))
            if train:
                np.save(train_dir + '/' + str(j) + '.npy', mfcc_f)
            else:
                np.save(test_dir + '/' + str(j) + '.npy', mfcc_f)
            j += 1
        i += 1
    print("done")

def TIMIT_SV(train=True):
    if train:
        src_dir = '../data/TIMIT/train_wav/*/*/*.WAV'
        tar_dir = '../data/TIMIT_SV/train_sv'

    else:
        src_dir = '../data/TIMIT/test_wav/*/*/*.WAV'
        tar_dir = '../data/TIMIT_SV/test_sv'

    os.makedirs(tar_dir, exist_ok=True)
    spks = glob.glob(os.path.dirname(src_dir))

    for i,spk in enumerate(spks):
        print('%dth speaker processing...' % i)
        spk_features = []
        wav_paths = glob.glob(os.path.join(spk, '*.WAV'))
        for wav_path in wav_paths:
            audio, sr = torchaudio.load(wav_path)
            time = audio.shape[1] / sr
            if time >= 3:
                audio = audio[:, 0:sr * 3].reshape(-1, sr * 3)
                time = audio.shape[1] / sr
            else:
                zeros = torch.zeros(sr * 3 - audio.shape[1]).reshape(1, -1)
                audio = torch.cat((audio, zeros), dim=1)
                time = audio.shape[1] / sr
            mfcc_feat = mfcc(audio, sr, appendEnergy=True)
            spk_features.append(mfcc_feat)
        np.save( tar_dir + '/' + str(i) + '.npy', spk_features)
    print("done")



if __name__ == '__main__':
    src_dir = r'../data/TIMIT/*/*/*/*.WAV'
    tar_dir = r'../data/TIMIT_SI/'

    # speaker identification
    print("tisi train dataset ...")
    TIMIT_SI(src_dir, tar_dir)

    print("tisi test dataset ...")
    TIMIT_SI(src_dir, tar_dir, train=False)

    # # speaker verification
    # print("tisv train dataset ...")
    # TIMIT_SV()
    #
    # print("tisv test dataset ...")
    # TIMIT_SV(train=False)