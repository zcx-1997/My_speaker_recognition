# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> data_loader
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/14 下午2:47
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
import torch
import torchaudio
from torch.utils.data import DataLoader
# 数据集：librispeech

train_db = torchaudio.datasets.LIBRISPEECH('data','dev-clean',download=True)
print(len(train_db))
print(train_db[0])
print(train_db[1])
print(train_db[0][0])
print(train_db[0][0].shape)
print(train_db[1][0].shape)

train_loader = DataLoader(train_db,batch_size=2)
print(len(train_loader))
# data = next(iter(train_loader))
# print(data[0])
# print(data[1])

audio,sr = torchaudio.load(r'data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac')
print(audio[:2],sr)
