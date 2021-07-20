#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/31 11:28
    Author  : 春晓
    Software: PyCharm
"""
import torch
import librosa
import torchaudio
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

print("=================================================")
print("torchaudio:")
wav_path = '../data/TIMIT/train_wav/DR1/FCJF0/SA1.WAV'
audio, sr = torchaudio.load(wav_path)
print("audio shape:{}, audio type:{}\nsr:{}".format(audio.shape, type(audio),
                                                    sr))
print("mean:{:.5f}, max:{:.5f}, min:{:.5f}".format(audio.mean(), audio.max(),
                                                   audio.min()))
# audio shape:torch.Size([1, 46797]), audio type:<class 'torch.Tensor'>
# sr:16000

# a = torch.tensor([[0.0,1,2,3,4,5]])
# print(a.shape)   # torch.Size([1, 6])
# print(a.t().shape)  # torch.Size([6, 1])

plt.figure()
plt.plot(audio.t().numpy())
plt.show()

print("=================================================")
print("scipy.io.wavfile:")
sr2, audio2 = wavfile.read(wav_path)
print("audio shape:{}, audio type:{}\nsr:{}".format(audio2.shape, type(audio2),
                                                    sr2))
print("mean:{:.5f}, max:{:.5f}, min:{:.5f}".format(audio2.mean(), audio2.max(),
                                                   audio2.min()))
# audio shape:(46797,), audio type:<class 'numpy.ndarray'>
# sr:16000
plt.figure()
plt.plot(audio2)
plt.show()

print("=================================================")
print("librosa:")
audio3, sr3 = librosa.load(wav_path, sr=None)
print("audio shape:{}, audio type:{}\nsr:{}".format(audio3.shape, type(audio3),
                                                    sr3))
print("mean:{:.5f}, max:{:.5f}, min:{:.5f}".format(audio3.mean(), audio3.max(),
                                                   audio3.min()))
# audio shape:(46797,), audio type:<class 'numpy.ndarray'>
# sr:16000
plt.figure()
plt.plot(audio3)
plt.show()
