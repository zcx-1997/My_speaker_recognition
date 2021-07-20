#!/usr/bin/python3

# 音频格式转换

"""
    Time    : 2021/4/13 9:23
    Author  : 春晓
    Software: PyCharm
"""

import glob
import os
from sphfile import SPHFile


def sphere2wav(wav_path, tar_dir):
    '''
    将sphere格式的音频文件转换成wav格式
    :param wav_path: 要处理的音频路径
    :param tar_dir: 处理之后的音频存储目录
    :return: wav格式的音频
    '''
    sph_files = glob.glob(wav_path)
    # print(len(sph_files))
    for i in sph_files:
        tar_path = tar_dir + '/' + i.split('/', 1)[1]

        if os.path.exists(os.path.dirname(tar_path)):
            sph = SPHFile(i)
            sph.write_wav(tar_path)
        else:
            os.makedirs(os.path.dirname(tar_path))
            sph = SPHFile(i)
            sph.write_wav(tar_path)
    print("sphere2wav is done")


if __name__ == '__main__':
    train_wav_path = r'/home/zcx/datasets/TIMIT/TRAIN/*/*/*.WAV'
    train_tar_dir = r'/home/zcx/datasets/TIMIT/train_wav'
    sphere2wav(train_wav_path, train_tar_dir)

    test_wav_path = r'/home/zcx/datasets/TIMIT/TEST/*/*/*.WAV'
    test_tar_dir = r'/home/zcx/datasets/TIMIT/test_wav'
    sphere2wav(test_wav_path, test_tar_dir)
