# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> wav_process
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/12 下午9:19
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
import torchaudio
from python_speech_features import fbank, mfcc

import constants as c

def extractFeatures(src_dir, tar_dir):
    print("start text independent utterance feature extraction")

    train_dir = os.path.join(tar_dir, 'train_si')
    test_dir = os.path.join(tar_dir, 'test_si')
    os.makedirs(train_dir, exist_ok=True)
    print("train data dir ok!")
    os.makedirs(test_dir, exist_ok=True)
    print("test data dir ok!")

    spks_list = glob.glob(os.path.dirname(src_path))
    for i, spk_dir in enumerate(spks_list):
        print("%dth speaker processing..." % i)
        wav_names = os.listdir(spk_dir)
        random.shuffle(wav_names)

        train_wavs = wav_names[:7]
        test_wavs = wav_names[7:]
        train_spec, test_spec = [], []

        for wav_name in train_wavs:  # 'SI1768.WAV'
            wav_path = os.path.join(spk_dir, wav_name)
            audio, sr = torchaudio.load(wav_path)
            time = audio.shape[1] / sr
            if time >= c.fix_time_utter:
                audio = audio[:, 0:sr * c.fix_time_utter].reshape(-1,
                                                                  sr * c.fix_time_utter)
                time = audio.shape[1] / sr
            else:
                zeros = torch.zeros(
                    sr * c.fix_time_utter - audio.shape[1]).reshape(1, -1)
                audio = torch.cat((audio, zeros), dim=1)
                time = audio.shape[1] / sr
            # print('time', time)  # 3

            fbank_feat, energy = fbank(audio, nfilt=40)
            logfbank_feat = np.log(fbank_feat)
            logfbank_feat[:, 0] = np.log(energy)
            train_spec.append(logfbank_feat)
        train_spec = np.array(train_spec)
        print(train_spec.shape)
        print("save", train_dir)
        np.save(os.path.join(train_dir, "%d.npy" % i), train_spec)


        for wav_name in test_wavs:
            wav_path = os.path.join(spk_dir, wav_name)
            audio, sr = torchaudio.load(wav_path)
            time = audio.shape[1] / sr
            if time >= c.fix_time_utter:
                audio = audio[:, 0:sr * c.fix_time_utter].reshape(-1,
                                                                  sr * c.fix_time_utter)
                time = audio.shape[1] / sr
            else:
                zeros = torch.zeros(
                    sr * c.fix_time_utter - audio.shape[1]).reshape(1, -1)
                audio = torch.cat((audio, zeros), dim=1)
                time = audio.shape[1] / sr
            # print('time', time)  # 3

            fbank_feat, energy = fbank(audio, nfilt=40)
            logfbank_feat = np.log(fbank_feat)
            logfbank_feat[:, 0] = np.log(energy)
            test_spec.append(logfbank_feat)
        test_spec = np.array(test_spec)
        print(test_spec.shape)
        print("save", test_dir)
        np.save(os.path.join(test_dir, "%d.npy" % i), test_spec)


if __name__ == '__main__':
    src_path = r'/home/zcx/datasets/TIMIT/TIMIT_WAV/*/*/*/*.wav'
    tar_dir = r'data'
    extractFeatures(src_path, tar_dir)
    # extractFeatures()