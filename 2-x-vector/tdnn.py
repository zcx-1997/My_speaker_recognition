# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_speaker_recognition -> tdnn
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/9/1 下午7:35
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
import torch.nn as nn
import torch.nn.functional as F

from data_loader import TIMITDataset


class TDNN(nn.Module):

    def __init__(
            self,
            input_dim=3,
            output_dim=3,
            context_size=3,
            dilation=1,
            stride=1,
            batch_norm=True,
            dropout_p=0.0
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        # for name, param in self.kernel.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.constant_(param, 1.0)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # x.shape = (batch, in_dim*context_size, new_t)
        x = x.transpose(1, 2)
        # x.shape = (batch, new_t, in_dim*context_size)
        x = self.kernel(x)
        # x.shape = (batch, new_t, output_dim)

        x = self.nonlinearity(x)
        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)
        return x

if __name__ == '__main__':
    # train_db = TIMITDataset()
    # x = train_db[0]
    # print(x.shape)

    x = torch.arange(0,1*5*3).view(1,5,3).float()
    print(x)


    tdnn = TDNN()
    y = tdnn(x)
    print(y.shape)
    print(y)


    tdnn1 = TDNN(input_dim=3,context_size=3,dilation=1,output_dim=3)
    tdnn2 = TDNN(input_dim=3,context_size=3,dilation=1,output_dim=1)

    net = nn.Sequential(tdnn1,tdnn2)
    y1 = net(x)
    print(y1.shape)
    print(y1)
