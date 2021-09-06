# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> voxceleb1_wav_process
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/6 下午4:54
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
import time
import numpy as np

import torchaudio
from python_speech_features import fbank, mfcc


def extract_and_save_features(src_dir, tar_dir):

    print("start text independent utterance feature extraction", time.ctime())

    train_src_dir = os.path.join(src_dir, 'vox1_dev_wav', 'wav')
    test_src_dir = os.path.join(src_dir, 'vox1_test_wav', 'wav')

    train_dir = os.path.join(tar_dir, 'train_10')
    # test_dir = os.path.join(tar_dir, 'test_for_train_100')
    os.makedirs(train_dir, exist_ok=True)
    print("train data dir ok!")
    # os.makedirs(test_dir, exist_ok=True)
    # print("test data dir ok!")

    # train data
    print(time.ctime())
    print("==================== train data ======================")
    train_spks_list = os.listdir(train_src_dir)
    for i, spk_name in enumerate(train_spks_list):
        print("\n%dth speaker processing..." % i)
        spk_id = spk_name[3:].lstrip('0')
        if int(spk_id) <= 10:
            wav_dir = os.path.join(train_src_dir, spk_name, '*', '*')
            wavs_list = glob.glob(wav_dir)
            spk_feats = []
            for j, wav_path in enumerate(wavs_list):
                audio, sr = torchaudio.load(
                    wav_path)  # audio:tensor,[1,n]; sr=16000
                audio = audio.reshape(-1)  # audio:tensor,[n,]
                # time = len(audio) / sr
                # print('utter_time: ', time)
                fbank_feat, energy = fbank(audio, nfilt=40)
                logfbank_feat = np.log(fbank_feat)
                logfbank_feat[:, 0] = np.log(energy)
                # print(logfbank_feat.shape)
                spk_feats.append(logfbank_feat)
            spk_np = np.array(spk_feats,dtype=object)
            np.save(os.path.join(train_dir, "%s.npy" % spk_name), spk_np)

    # # test data
    # print(time.ctime())
    # print("==================== test data ======================")
    # test_spks_list = os.listdir(test_src_dir)
    # for i, spk_name in enumerate(test_spks_list):
    #     print("\n%dth speaker processing..." % i)
    #     # spk_id = spk_name[3:].lstrip('0')
    #     wav_dir = os.path.join(test_src_dir, spk_name, '*', '*')
    #     wavs_list = glob.glob(wav_dir)
    #     spk_utter = []
    #     for wav_path in wavs_list:
    #         audio, sr = torchaudio.load(
    #             wav_path)  # audio:tensor,[1,n]; sr=16000
    #         audio = audio.reshape(-1)  # audio:tensor,[n,]
    #         # time = len(audio) / sr
    #         # print('utter_time: ', time)
    #         fbank_feat, energy = fbank(audio, nfilt=40)
    #         logfbank_feat = np.log(fbank_feat)
    #         logfbank_feat[:, 0] = np.log(energy)
    #         # print(logfbank_feat.shape)
    #         spk_utter.append(logfbank_feat)
    #     spk_np = np.array(spk_utter,dtype=object)
    #     np.save(os.path.join(test_dir, "%s.npy" % spk_name), spk_np)
if __name__ == '__main__':
    # src_dir = r'/home/zcx/datasets/VoxCeleb'
    # tar_dir = r'Voxceleb1'
    # extract_and_save_features(src_dir, tar_dir)


    spk_path = r'Voxceleb1/train_10/id10010.npy'
    spk_feats = np.load(spk_path,allow_pickle=True)
    wav1 = spk_feats[0]
    print(spk_feats.shape)
    print(wav1.shape)

