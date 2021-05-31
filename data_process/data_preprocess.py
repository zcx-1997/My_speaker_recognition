#!/usr/bin/env python    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/24 11:35
    Author  : 春晓
    Software: PyCharm
"""

import os
import glob
import librosa
import numpy as np

from hparam import hparam as hp

audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))
# print(audio_path)  # 如：'../data/TIMIT\\TEST\\DR1\\FAKS0'

def save_spectrogram_tisv():

    print('='*5 +'start text independent utterance feature extraction'+'='*5)

    os.makedirs(hp.data.train_path,exist_ok=True)
    os.makedirs(hp.data.test_path,exist_ok=True)

    utter_min_len = (hp.data.tisv_frame*hp.data.hop_frame + hp.data.len_frame) * hp.data.sr
    print(utter_min_len)

    num_total_speakers = len(audio_path)
    num_train_speakers = (num_total_speakers // 10) * 9
    num_test_speakers = num_total_speakers - num_train_speakers
    print('total speakers num: %d' % num_total_speakers)
    print('train: %d; test: %d'%(num_train_speakers,num_test_speakers))

    for i,folder in enumerate(audio_path):
        print('%dth speaker processing...'%i)
        utter_spec = []
        # os.listdir(folder) # 将folder目录下的所有文件构建成一个列表
        for utter_name in os.listdir(folder):
            if utter_name[-4:] == '.WAV':
                utter_path = os.path.join(folder,utter_name)
                audio,sr = librosa.load(utter_path,hp.data.sr)

                # print(audio.shape)
                intervals = librosa.effects.split(audio,top_db=30)
                # print(intervals.shape)
                # print(intervals)

                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:
                        audio_part = audio[interval[0]:interval[1]]
                        s = librosa.stft(y=audio_part,n_fft=hp.data.nfft,win_length=int(hp.data.len_frame*hp.data.sr),hop_length=int(hp.data.hop_frame*hp.data.sr))
                        s = np.abs(s)**2
                        mel_basis = librosa.filters.mel(sr,n_fft=hp.data.nfft,n_mels=hp.data.nmels)
                        s = np.log10(np.dot(mel_basis,s)+1e-6)
                        utter_spec.append(s[:,:hp.data.tisv_frame])
                        utter_spec.append(s[:,-hp.data.tisv_frame:])

        utterances_spec = np.array(utter_spec)
        print(utterances_spec.shape)

        if i < num_train_speakers:
            np.save(os.path.join(hp.data.train_path,'speaker%d.npy'%i),utterances_spec)
        else:
            np.save(os.path.join(hp.data.test_path,'speaker%d.npy'%i),utterances_spec)

if __name__ == '__main__':
    save_spectrogram_tisv()


