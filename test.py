# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> test
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/2 下午5:20
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

y = torch.arange(2)
print(y)
y = y.view(2,1)
print(y)
y1 = y.repeat(1,2).view(2*2)
print(y1)

x = torch.arange(12).view(2,2,3)
print(x)
x1= x.reshape(x.shape[0]*x.shape[1],x.shape[2])
print(x1)