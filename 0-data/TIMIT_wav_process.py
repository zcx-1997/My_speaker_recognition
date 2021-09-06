# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> wav_process
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/6 下午3:46
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
import torch
import torchaudio
import numpy as np
from python_speech_features import fbank, logfbank, mfcc

def extract_and_save_features(src_path, tar_dir):
    """
    Extract features, the features are saved as numpy file.
    The time per utterance is fixed by 3s and 299 frames from each utterance
    """

    # set constants
    train_num_spks = 462
    fix_time_utter = 3

    print("start text independent utterance feature extraction")

    train_dir = os.path.join(tar_dir, 'train')
    test_dir = os.path.join(tar_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    print("train data dir ok!")
    os.makedirs(test_dir, exist_ok=True)
    print("test data dir ok!")

    spks_list = glob.glob(os.path.dirname(src_path))
    total_num_spks = len(spks_list)

    test_num_spks = total_num_spks - train_num_spks
    print("total speaker number : %d" % total_num_spks)
    print("train : %d, test : %d" % (train_num_spks, test_num_spks))
    print("==========================================")

    for i, spk_dir in enumerate(spks_list):
        # eg. spk_dir = '/home/zcx/datasets/0-TIMIT/TIMIT_WAV/train_wav/DR3/MCAL0'
        print("%dth speaker processing..." % i)
        spk_spec = []
        for wav_name in os.listdir(spk_dir):  # 'SI1768.WAV'
            wav_path = os.path.join(spk_dir, wav_name)
            audio, sr = torchaudio.load(wav_path)
            time = audio.shape[1] / sr
            if time >= fix_time_utter:
                audio = audio[:, 0:sr * fix_time_utter].reshape(-1, sr * fix_time_utter)
                time = audio.shape[1] / sr
            else:
                zeros = torch.zeros(sr * fix_time_utter - audio.shape[1]).reshape(1, -1)
                audio = torch.cat((audio, zeros), dim=1)
                time = audio.shape[1] / sr
            # print('time', time)  # 3

            fbank_feat, energy = fbank(audio, nfilt=40)
            logfbank_feat = np.log(fbank_feat)
            logfbank_feat[:, 0] =  np.log(energy)
            spk_spec.append(logfbank_feat)
        spk_spec = np.array(spk_spec)
        print(spk_spec.shape)

        # save spectrogram as numpy file
        if i < train_num_spks:
            np.save(os.path.join(train_dir, "%d.npy" % i), spk_spec)
        else:
            np.save(os.path.join(test_dir, "%d.npy" % i), spk_spec)


if __name__ == '__main__':
    src_path = r'/home/zcx/datasets/TIMIT/TIMIT_WAV/*/*/*/*.wav'
    tar_dir = r'./TIMIT'

    extract_and_save_features(src_path, tar_dir)
