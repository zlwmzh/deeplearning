#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/17 21:14
# @Author  : Micky
# @Site    : 
# @File    : AND_NOT感知器练习.py
# @Software: PyCharm

"""
 AND感知器练习
 output = 1, if weight1 * input1 + weight2 * input2 + bias > 0
 or
 output = 2, if weight1 * input2 + weight2 * input2 +bias < 0

 当output1 = 1 and output2 = 1的时候，output = 1
"""

import pandas as pd
def and1():

    # 需要手动设置weight1，weight2, and bias来实现上述目标。
    weight1 = 1.0
    weight2 = 1.0
    bias = -1.3

    # Input and outputs
    # 测试的输入样本
    test_inputs = [(0,0),(0,1),(1,0),(1,1)]
    # 输出值
    correct_outputs = [False,False,False,True]

    # 用来存储信息
    outputs = []
    # 开始预测
    for test_input, correct_output in zip(test_inputs,correct_outputs):
        # 预测样本
        linear_combination = weight1 * test_input[0] +weight2 * test_input[1] + bias
        # 判断预测值linear_combination是否大于等于0，如果大于，则为True，否则为False,用int转为0和1的形式
        output = int(linear_combination >= 0)
        if output == correct_output:
            is_correct_string = '正确'
        else:
            is_correct_string = '错误'
        outputs.append([test_input[0],test_input[1],linear_combination,output,is_correct_string])

    # 统计预测错误的个数
    num_wrong = len([output[4] for output in outputs if output[4] =='错误'])
    # 将outputs转换为DataFrame，定义列名，可视化
    output_frame = pd.DataFrame(outputs,columns=['input1','input2','线性预测值','激活输出值','是否正确'])
    if not num_wrong:
        print('Nice，全对了!!!')
    else:
        print('错了{}个'.format(num_wrong))
    print(output_frame)

"""
OR 感知器
OR只关注一个input,如果input为1，则输出为1

从AND感知器到OR感知器的方法是：
1. 提升权重
2. 减少偏执项的幅度
"""
def or1():
    # 需要手动设置weight1，weight2, and bias来实现上述目标。
    weight1 = 1.0
    weight2 = 1.0
    bias = -1.3

    # Input and outputs
    # 测试的输入样本
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # 输出值
    correct_outputs = [False, True, True, True]

    # 用来存储信息
    outputs = []
    # 开始预测
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        # 预测样本
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        # 判断预测值linear_combination是否大于等于0，如果大于，则为True，否则为False,用int转为0和1的形式
        output = int(linear_combination >= 0)
        if output == correct_output:
            is_correct_string = '正确'
        else:
            is_correct_string = '错误'
        outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

    # 统计预测错误的个数
    num_wrong = len([output[4] for output in outputs if output[4] == '错误'])
    # 将outputs转换为DataFrame，定义列名，可视化
    output_frame = pd.DataFrame(outputs, columns=['input1', 'input2', '线性预测值', '激活输出值', '是否正确'])
    if not num_wrong:
        print('Nice，全对了!!!')
    else:
        print('错了{}个'.format(num_wrong))
    print(output_frame)
if __name__ == '__main__':
    and1()