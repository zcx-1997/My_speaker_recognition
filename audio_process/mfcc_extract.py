#!/usr/bin/python3
"""
    Time    : 2021/4/13 10:08
    Author  : 春晓
    Software: PyCharm
"""
import scipy.io.wavfile as wavfile
import librosa
import python_speech_features
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def audio_load(wav_path, mode=0):

    '''
    加载音频
    :param wav_path: 音频路径
    :param mode:
                0：scipy.io.wavfile.read()
                1:librosa.load()
    :return: audio, sr
    '''

    if mode:
        audio, sr = librosa.load(wav_path, sr=None)
    else:
        sr, audio = wavfile.read(wav_path)

    return audio, sr

def pre_emphasis(audio,preemph=0.97):
    '''
    预加重
    :param audio:待处理的 audio
    :param sr: 采样率
    :param preemph: 预加重系数，[0.95,1]
    :return: 预加重的 audio
    '''
    preemp_audio = np.append(audio[0], audio[1:] - preemph * audio[:-1])
    return preemp_audio

def mfcc_13(audio,sr,energy=True,winfunc=None):
    '''
    提取 13维的 mfcc
    :param audio: audio
    :param sr: 采样率
    :param energy: True 是否将 mfcc的第一维替换成能量值
    :param winfunc: 加窗函数
    :return: (n_frames,13)
    '''
    if  winfunc is None:
        mfcc_feat = python_speech_features.mfcc(audio, samplerate=sr, appendEnergy=energy)
    else:
        mfcc_feat = python_speech_features.mfcc(audio, sr, appendEnergy=energy, winfunc=winfunc)
    return mfcc_feat

def mfcc_39(audio,sr,energy=True,winfunc=None):
    '''
    39维的 mfcc特征 mfcc+一阶差分+二阶差分 (三组数据第一维的值均被替换为了能量值)
    :param audio: audio
    :param sr:采样率
    :param energy:True，是否替换成能量
    :param winfunc:加窗函数
    :return:(n_frames,39)
    '''
    mfcc = mfcc_13(audio,sr,energy=energy,winfunc=winfunc)
    mfcc_delta = python_speech_features.delta(mfcc,2)
    mfcc_delta2 = python_speech_features.delta(mfcc_delta,2)
    mfcc = np.hstack((mfcc,mfcc_delta,mfcc_delta2))
    return mfcc

def mfcc_40(audio,sr,winfunc=None):
    '''
    40维的 mfcc特征 能量值+mfcc+一阶差分+二阶差分
    :param audio: audio
    :param sr: 采样率
    :param winfunc: 加窗函数
    :return: (n_frames,40)
    '''
    mfcc = mfcc_39(audio,sr,energy=False,winfunc=winfunc)

    if  winfunc is None:
        _, energy = python_speech_features.fbank(audio, sr)
    else:
        _, energy = python_speech_features.fbank(audio, sr, winfunc=winfunc)
    energy = np.log(energy)
    energy = energy.reshape((mfcc.shape[0],1))
    # print(energy.shape)
    mfcc = np.hstack((energy,mfcc))
    return mfcc

if __name__ == '__main__':

    print('===========python_speech_features')
    wav_path = r'../data/TIMIT/train_wav/DR6/FAPB0/SA1.WAV'
    audio, sr = audio_load(wav_path)
    print('采样率，音频类型，shape，最大值，最小值,中值', sr, audio.dtype, audio.shape, np.max(audio), np.min(audio), np.median(audio))
    _,energy = python_speech_features.fbank(audio,sr,winfunc=np.hamming)
    energy = np.log(energy)
    print("energy",energy[:5])

    print("13维的mfcc(能量值)：")
    mfcc13 = mfcc_13(audio,sr,winfunc=np.hamming)
    print(mfcc13.shape)
    print(mfcc13[:5])

    print("39维的mfcc(能量值)：")
    mfcc39 = mfcc_39(audio,sr,winfunc=np.hamming)
    print(mfcc39.shape)
    print(mfcc39[:5])

    print("40维的mfcc(能量值+mfcc+一阶差分+二阶差分)：")
    mfcc40 = mfcc_40(audio,sr,winfunc=np.hamming)
    print(mfcc40.shape)
    print(mfcc40[:5])




    # print('===========librosa')
    # wav_path = r'../data/TIMIT/train_wav/DR6/FAPB0/SA1.WAV'
    # audio, sr = audio_load(wav_path,mode=1)
    # print('采样率，音频类型，shape，最大值，最小值,中值', sr, audio.dtype, audio.shape, np.max(audio), np.min(audio), np.median(audio))
    #
    # mfcc = librosa.feature.mfcc(audio,sr=sr,n_mfcc=40)
    # print(mfcc.shape)
    # print(mfcc[:5])
