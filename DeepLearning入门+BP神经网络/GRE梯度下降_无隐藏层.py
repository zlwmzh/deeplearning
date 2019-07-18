#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 11:43
# @Author  : Micky
# @Site    : 
# @File    : GRE梯度下降_无隐藏层.py
# @Software: PyCharm


"""
案例：研究生学院录取数据，用梯度下降训练一个网络。
数据有三个输入特征：GRE 分数、GPA 分数和本科院校排名（从 1 到 4）。排名 1 代表最好，排名 4 代表最差。
"""
import pandas as pd
import numpy as np

# 设置pandas显示的参数
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

admissions = pd.read_csv(filepath_or_buffer='../datas/11.csv')

def data_explore(admissions):
    print(admissions.head())
    print(admissions.info())       # 查看数据集中是否有空值
    print(admissions.describe())   # 再次可以看到是否有空值，以及值范围，需要考虑做数据变换。
    print(admissions['admit'].value_counts())  # 查看数据是否均衡


"""
1、数据清理
rank 是类别特征，其中的数字并不表示任何相对的值。排名第 2 并不是排名第 1 的两倍；
排名第 3 也不是排名第 2 的 1.5 倍。因此，我们需要用哑变量 来对 rank 进行编码。
把数据分成 4 个新列，用 0 或 1 表示。排名为 1 的行对应 rank_1 列的值为 1 ，其余三列的值为 0；
排名为 2 的行对应 rank_2 列的值为 1 ，其余三列的值为 0，以此类推。

把 GRE 和 GPA 数据标准化，变成均值为 0，标准偏差为 1。因为 sigmoid 函数会挤压很大或者很小的输入。
很大或者很小输入的梯度为 0，这意味着梯度下降的步长也会是 0。
"""
def data_transform(admissions):
    """
    一. rank表示学校等级，需要用亚编码
    1、 用pd.get_dummies 将rank列，转成哑变量，新变量名前缀为：prefix='rank'
    2、 将进行过亚编码的Rank列与原来的数据进行拼接
    3、因为融合后的数据集中包含原来的rank列，所以我们需要移除
    """
    # concat函数是在pandas底下的方法，可以将数据根据不同的轴作简单的融合
    data = pd.concat([admissions,pd.get_dummies(admissions['rank'],prefix='rank')],axis=1)
    data = data.drop('rank',axis = 1)

    """
    二. gre 和gpa标准化
    x* = x-x_average / 标准差
    """
    for filed in ['gre','gpa']:
       mean,std = data[filed].mean(),data[filed].std()
       # loc基于列值，iloc基于index索引
       data.loc[:,filed] = (data[filed] - mean)/std

    """
    三. 数据拆分：训练集合测试集划分
    np.random.choice，随机选择数据集中90% 数据的index
    """
    # 分为90%的训练数据，10%的测试数据
    np.random.seed(214)
    # replace 是否放回
    train_sample_index = np.random.choice(data.index,size=int(len(data)*0.9),replace=False)
    # test_sample = data.iloc[test_sample_index,:]
    train_sample,test_sample= data.ix[train_sample_index],data.drop(train_sample_index)

    """
    四. 特征属性和目标属性分类
    """
    train_features,train_targets = train_sample.drop('admit',axis = 1),train_sample['admit']
    test_features,test_targets = test_sample.drop('admit',axis = 1),test_sample['admit']
    return train_features,train_targets,test_features,test_targets

"""
GRE无隐藏层--梯度下降编程练习
*** todo 建模伪代码如图 ***
"""
def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

def gre_work(feature,target):
    np.random.seed(214)
    sample_count,feature_count = feature.shape

    # 初始化权重 一般都是这种初始化方法
    weights = np.random.normal(scale=1/feature_count **0.5,size=feature_count)

    last_loss = None
    # 神经网络超参数设置
    epochs = 1000
    # 学习率
    learnrate = 0.5
    for e in range(epochs):
        #定义一个del_w矩阵，没执行一次正向和方向都重置
        del_w = np.zeros(weights.shape)
        for x, y in zip(feature.values, target):
            # 遍历所有的特征属性和目标属性
            # 计算输出output
            output = np.dot(x,weights)
            # 计算误差error
            error = sigmoid(output) - y
            # 计算error_term
            error_term = sigmoid(output)*(1-sigmoid(output))
            #计算总变化del_w
            del_w += error * error_term * x
        weights -= learnrate * del_w
        # 打印mse = 误差平方和/样本数
        if e % (epochs / 10) == 0:
            output = sigmoid(np.dot(feature,weights))
            loss = np.mean((output - y)**2)
            if last_loss and last_loss < loss:
                #上次的误差比这次的小，提示警告
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss

    # 计算测试数据的正确率
    test_out = sigmoid(np.dot(test_x,weights))
    predictions = test_out > 0.5
    # 计算想等情况下的均值就可以
    accuracy = np.mean(predictions == test_y)
    print("Prediction accuracy: {}".format(accuracy))

if __name__ == '__main__':
    train_x,train_y,test_x,test_y = data_transform(admissions)
    # print(test_x)
    gre_work(train_x,train_y)