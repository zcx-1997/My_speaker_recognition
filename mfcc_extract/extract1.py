#!/usr/bin/python3
"""
    Time    : 2021/4/6 20:51
    Author  : 春晓
    Software: PyCharm
"""
import numpy as np
import scipy.io.wavfile as wavfile
from python_speech_features import mfcc
from matplotlib import pyplot as plt

import librosa

wav_path = 'data/S0002/01.wav'
sr,audio = wavfile.read(wav_path)
print("音频采样率：",sr)
print("音频转化后的数据类型：",type(audio),"\nshape：",audio.shape,"\n前5个数：",audio[1:5])

au_duration = librosa.get_duration(filename=wav_path)
print("音频持续时间(s)：",au_duration)
print("数组长度=采样率*持续时间：%.2f = %d * %.2f" % (sr*au_duration,sr,au_duration))
# plt.figure()
# plt.plot(audio)
# plt.show()

mfcc_feat = mfcc(audio)

print("提取到的mfcc特征")
print("shape:",mfcc_feat.shape,"\n数据类型:",type(mfcc_feat))
print("前5个值：",mfcc_feat)
