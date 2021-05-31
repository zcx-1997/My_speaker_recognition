#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/5/24 17:10
    Author  : 春晓
    Software: PyCharm
"""

import yaml

def load_hparam(filename):
    stream = open(filename,'r')
    docs = yaml.load_all(stream)  #<generator object load_all at 0x000001FBB3A05DD0>
    hparam_dict = dict()

    for doc in docs:
        for k,v in doc.items():
            hparam_dict[k] = v

    return hparam_dict


class Dotdict(dict):
    '''可以使用dictname.key的形式访问字典'''

    def __init__(self,dct=None):
        dct = dict() if not dct else dct

        for k,v in dct.items():
            if hasattr(v,'keys'):  # 如果对象v有属性’keys‘，即v是字典
                v = Dotdict(v)
            self[k] = v

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Hparam(Dotdict):

    def __init__(self,file = '../config/config.yaml'):
        super(Hparam, self).__init__()

        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)

        for k,v in hp_dotdict.items():
            setattr(self,k,v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

hparam = Hparam()


if __name__ == '__main__':
    hparam = Hparam()
    print(hparam.training)
    print(hparam.data.train_path)
