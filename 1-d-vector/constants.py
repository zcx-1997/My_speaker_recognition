# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> constants
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/11 上午10:38
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
# data
sr = 16000
frame_win = 0.025  # 400
frame_hop = 0.01   # 160
num_frames = 160
nfft = 512
nmels = 40
fix_time_utter = 3 # 单位：s

# train
train_num_spks = 462
num_utters_trian = 10
batch_train = 4

# errollement and test
num_utters_test = 7
batch_test = 4
# model
hidden_lstm = 768
num_layers_lstm = 3
hidden_fc = 256