#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/19 16:16
# @Author  : Micky
# @Site    : 
# @File    : 隐藏层的梯度下降.py
# @Software: PyCharm

import numpy as np
# todo 隐藏层的正向传播练习
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def hidden_work():
    # 网络大小
    n_input = 4
    n_hidden = 3
    n_output = 2

    np.random.seed(214)
    # 产生四个随机数
    X = np.random.randn(4)

    weights_input_to_hidden = np.random.normal(scale=1/n_input**0.5,size=(n_input,n_hidden))


if __name__ == '__main__':
    hidden_work()