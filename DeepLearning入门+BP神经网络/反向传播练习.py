#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 10:40
# @Author  : Micky
# @Site    : 
# @File    : 反向传播练习.py
# @Software: PyCharm

"""
sigmoid函数练习
1. 实现sigmoid激活函数
2. 计算神经网路输出

sigmoid函数输出：
sigmoid(x) = 1/(1+e-x)

网络输出：
y = f(h) = sigmoid(∑i wi*xi+b)
"""

import numpy as np
"""
sigmoid函数练习
"""
def f1():
    def sigmoid(x):
        # todo-编写代码: 完成sigmoid函数
        return

    inputs = np.array([0.7, -0.3])
    weights = np.array([0.1, 0.8])
    bias = -0.1

    # todo-编写代码: 计算输出
    output = None

    print('Output:')
    print(output)

"""
解答上述问题
"""
def f1Answer():
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))
    inputs = np.asarray([0.7,-0.3])
    weights = np.asarray([0.1,0.8])
    bias = -0.1

    output = sigmoid(np.dot(inputs,weights) + bias)
    print('output：')
    print(output)

"""
梯度下降代码：(只执行1次反向传播)

Δw = α(步长)* δ * X
1. 定义sigmoid激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
2. 定义sigmoid激活函数的导数
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
3. 权重更新的学习率
learnrate = 0.5
4. 输入和权重的线性组合
h = x[0] * weights[0] + x[1] * weights[1]
或者h = np.dot(x,weights)
5. 神经网络输出
y = sigmoid(h)
6. 输出误差：
error = y - nn_output
输出的梯度：
output_grad = sigmoid_prime(h)
error_term = error * output_grad
# 梯度下降
delta_w = learnrate * error_term * x
"""

def f2():
    # 定义sigmoid激活函数
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    # 定义sigmoid激活函数的导数
    def sigmoid_prime(x):
        return sigmoid(x) * (1 - sigmoid(x))

    learnrate = 0.5
    x = np.array([1, 2, 3, 4])
    y = np.array(0.5)

    # 初始化权重值
    w = np.array([0.5, -0.5, 0.3, 0.1])

    ### 1. 正向传播
    h = np.dot(x,w)
    # 神经网络输出
    nn_output = sigmoid(h)
    # 反向传播
    error = nn_output - y
    error_term = error * sigmoid_prime(h)
    del_w = learnrate * error_term * x
    print('Neural Network output:')
    print(nn_output)
    print('Amount of Error:')
    print(error)
    print('Change in Weights:')
    print(del_w)

if __name__ == '__main__':
    f2()