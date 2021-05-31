#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/31 15:17
    Author  : 春晓
    Software: PyCharm
"""

import numpy as np
import librosa
import scipy.io.wavfile as wavfile
from python_speech_features import logfbank,mfcc,delta,fbank

def wav_load(filename):
    audio,sr = librosa.load(filename,sr=None)  # librosa对采样值进行了缩放
    # sr,audio = wavfile.read(filename)
    return audio,sr

def extract_mfcc(audio,sr,energy=True):
    mfcc_feat = mfcc(audio, sr, appendEnergy=energy)
    if energy:
        _, energy = fbank(audio, sr)
        energy = np.log(energy)
        energy = energy.reshape((mfcc_feat.shape[0], 1))

    mfcc_feat_d = delta(mfcc_feat,2)
    mfcc_feat_d2 = delta(mfcc_feat_d,2)
    mfcc_features = np.hstack((mfcc_feat,mfcc_feat_d,mfcc_feat_d2))
    mfcc_features = np.hstack((energy, mfcc_features))
    return mfcc_features

def extract_logfbank(audio,sr):
    logfbank_feat = logfbank(audio,sr,nfilt=40)
    # logfbank_feat_d = delta(logfbank_feat,2)
    # logfbank_feat_dd = delta(logfbank_feat_d,2)
    # logfbank_feat = np.hstack((logfbank_feat,logfbank_feat_d,logfbank_feat_dd))
    return logfbank_feat

if __name__ == '__main__':
    filename = '../data/TIMIT_Transformed/train_wav/DR1/FCJF0/SA1.WAV'
    audio,sr = wav_load(filename)
    mfccs = extract_mfcc(audio,sr)
    logfbanks = extract_logfbank(audio,sr)
    print(mfccs.shape)
    print(logfbanks.shape)





