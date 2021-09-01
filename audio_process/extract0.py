#!/usr/bin/python3
# 提取 mfcc：numpy和 python_speech_features
"""
    Time    : 2021/4/9 10:26
    Author  : 春晓
    Software: PyCharm
"""

import scipy.io.wavfile as wavfile
import librosa
from python_speech_features import delta
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

#from utils import sphere2wav

# 音频文件装换
# train_path = r'data/0-TIMIT/TRAIN/*/*/*.WAV'
# sphere2wav(train_path)
#
# test_path = r'data/0-TIMIT/TEST/*/*/*.WAV'
# sphere2wav(test_path,train=False)

# 加载音频文件
wav_path = r'../mfcc_extract/data/0-TIMIT/train_wav/DR6/FAPB0/SA1.WAV'
# audio,sr = librosa.load(wav_path,sr=None)

sr, audio = wavfile.read(wav_path)

au_duration = librosa.get_duration(filename=wav_path)
print("音频时长(s)：", au_duration)

print("音频采样率：", sr)
print("音频转化后的数据类型：", type(audio), "\nshape：", audio.shape, "\n前5个数：", audio[:5])
print("数组长度=采样率*音频时长：%.2f = %d * %.2f" % (sr * au_duration, sr, au_duration))

# ************** 采用 numpy 手动计算 mfcc *******************
# 1.预加重 preemphasis,输出预加重之后的 signal: 用方程实现：y(n) = x(n) - a*x(n-1)
pre_emphasis = 0.97
preemp_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
pre_aud_len = len(preemp_audio)
# print(preemp_audio[:5])
# print(pre_aud_len)

# 2.分帧加窗,输出 frames
winlen = 0.025
winstep = 0.01
frame_len, frame_step = int(winlen * sr), int(winstep * sr)
print('帧长：', frame_len, '帧移：', frame_step)
if pre_aud_len <= frame_len:
    num_frames = 1
else:
    num_frames = int(np.ceil((pre_aud_len - frame_len) / frame_step)) + 1
print('帧数：', num_frames)

# 分帧之后，最后一帧可能需要补0
pad_audio_len = int((num_frames - 1) * frame_step + frame_len)
# print(pad_audio_len-pre_aud_len)
zeros = np.zeros(pad_audio_len - pre_aud_len)
pad_audio = np.concatenate((preemp_audio, zeros))

# ind = np.tile(np.arange(0,frame_len),(num_frames,1))
# print(ind.shape) # shape: (num_frames,frame_len)

# ind2 = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
# print(ind2.shape)  # shape: (num_frames,frame_len)

indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step),
                                                                      (frame_len, 1)).T
# 每一个帧内元素的偏置(0,1,2) + 每帧的起始点(0,1,2,3,4) = 所有帧元素的位置((0,1,2),(1,2,3),(2,3,4),(3,4,5),(4,5,6))
# print(indices.shape)   # shape: (num_frames,frame_len)


frames = pad_audio[np.mat(indices).astype(np.int32, copy=False)]
# print(type(indices))  # <class 'numpy.ndarray'>
# print(type(np.mat(indices))) # <class 'numpy.matrix'>
# print(np.mat(indices).dtype) # int32

# print(pad_audio.shape)  # (59120,)
# print(frames.shape)  # (368, 400)

# 加上汉明窗
frames *= np.hamming(frame_len)

# 3.快速傅里叶变换并取功率谱(pow_spect)
# 傅里叶变换并取绝对值，得到频谱的幅度值
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
# 取功率谱
pow_spec = (1.0 / NFFT) * np.square(mag_frames)

# 计算能量，用来替换 mfcc中的第一个系数
energy = np.sum(pow_spec, 1)  # 对每一帧的能量谱进行求和
energy = np.where(energy == 0, np.finfo(float).eps, energy)  # 对能量为0的地方调整为eps，这样便于进行对数处理
energy = np.log(energy)
print(energy[:5])


# 4.mel滤波
nfilt = 26

# 计算梅尔三角间距滤波器，该滤波器在第一个频率和第三个频率处为0，在第二个频率处为1
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # 赫兹转梅尔
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 梅尔转赫兹
bin = np.floor((NFFT + 1) * hz_points / sr)
fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])  # left
    f_m = int(bin[m])  # center
    f_m_plus = int(bin[m + 1])  # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

# 滤波器和功率谱进行点乘
feat = np.dot(pow_spec, fbank.T)
feat = np.where(feat == 0, np.finfo(float).eps, feat)  # 不能出现0，保持数值稳定

# 5.计算 log映射,得到 fbank features

fbank_feat = np.log(feat)  # python_speech_features
# fbank_feat = 10 * np.log10(fbank_feat)  # psf 还提供了一种方式：log_S = 10 * log10(S)
# librosa使用另一种方式：S_db = 10 * log10(S) - 10 * log10(ref)  , ref=1。0
print("fbank:", fbank_feat.shape)

# 6.进行离散余弦变换 DCT,取前 13个系数作为 mfcc_feature(是否加入能量),得到 mfcc_feature

num_ceps = 13
mfcc = dct(fbank_feat, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
print("基本的mfcc:", mfcc.shape)

# 归一化倒谱提升
(nframes, ncoeff) = mfcc.shape  # shape:(368,13)
n = np.arange(ncoeff)
cep_lifter = 22
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift

print("归一化倒谱提升之后的 mfcc:", mfcc.shape)

# 是否加入能量
appendEnergy = 0
if appendEnergy:
    mfcc[:, 0] = np.log(energy)  # 只取 2-13个系数，第一个用能量的对数来代替
    print("加入（替换）能量的mfcc:", mfcc.shape)
# 不知道这是什么意思
# fbank_feat -= (np.mean(fbank_feat, axis=0) + 1e-8)
# mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

print('mfcc:', mfcc.shape)
# print(mfcc)

# ************** 采用 python_speech_feature 计算 mfcc *******************
import python_speech_features

winlen = 0.025
winstep = 0.01

hamming_len = winlen*sr
mfcc_feat = python_speech_features.mfcc(audio,samplerate=sr)
print("psf提取到的mfcc：",mfcc_feat.shape)

# 一阶差分和二阶差分
mfcc_feat_d1 = delta(mfcc, 1)
print('mfcc_1:', mfcc_feat_d1.shape)
mfcc_feat_d2 = delta(mfcc, 2)
print('mfcc_2:', mfcc_feat_d2.shape)
mfcc = np.hstack((mfcc, mfcc_feat_d1, mfcc_feat_d2))
print('结合一阶和二阶差分的mfcc:',mfcc.shape)

# 40维的 mfcc()
energy1 = energy.reshape((num_frames,1))
# print(energy)
# print(energy1.shape)
# print(energy)
# print(energy1)
mfcc_40 = np.hstack((mfcc,energy1))
print('40维（mfcc-一阶差分-二阶差分-能量）:',mfcc_40.shape)
