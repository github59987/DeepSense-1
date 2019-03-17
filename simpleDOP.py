#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: simpleDOP
@time: 2019/3/16 22:04
@desc:
'''
import numpy as np
from tensorflow_own.Routine_operation import SaveFile, LoadFile
def data_fft(p_prior):
    '''
    对原始数据按照DeepSense模型要求进行数据预处理
    :param p_prior: str, 数据路径前缀
    :return: None
    '''
    # scalar
    CLASS_NUM = 6
    WIDTH = 100
    ALL_NUM = 400000 #训练集设置400000，测试集设置200000
    F = 100
    rng = np.random.RandomState(0)
    # 导入数据
    dataset_fin = np.zeros(shape=([1]), dtype=np.float32)
    label_fin = np.zeros(shape=([1]), dtype=np.float32)
    dataset_fft_flat = np.zeros(shape=([1]), dtype=np.float32)
    for i in range(CLASS_NUM):
        print('正在导入第%s个模式数据' % (i + 1))
        p = p_prior + r'DeepSenseing\ICT-DataSet\Label_%s.txt' % (i + 1)
        dataset_per = np.zeros(shape=([1]), dtype=np.float32)
        label_per = np.zeros(shape=([ALL_NUM//WIDTH, 6]), dtype=np.float32)
        with open(p, 'r') as file:
            dataset = np.loadtxt(file, delimiter=',', skiprows=0)[:ALL_NUM, :]
            # 删除中间3列线性加速度
            dataset = np.delete(dataset, [3, 4, 5, 9, 10, 11, 12, 13], axis=1)
            dataset_transpose = dataset.T #(6, ALL_NUM)
            # print(dataset_transpose.shape)
            #对各列做傅里叶变换取模
            for i in range(0, ALL_NUM-WIDTH, WIDTH):
                dataset_fft = np.abs(np.fft.fft(a=dataset[:, i:i+WIDTH], n=2*F, axis=1))
                dataset_fft_flat = dataset_fft.reshape(1, -1)#(1, 6*2*F)
                dataset_per = np.vstack((dataset_per, dataset_fft)) if dataset_per.any() else dataset_fft
            label_per[:, i] = 1
            label_fin = np.vstack((label_fin, label_per)) if label_fin.any() else label_per
            dataset_fin = np.vstack((dataset_fin, dataset_fft_flat)) if dataset_fin.any() else dataset_fft_flat
    rng.shuffle(dataset_fin) #(6*ALL_NUM//WIDTH, 6*2*F)
    rng.shuffle(label_fin) #(6*ALL_NUM//WIDTH, 1)
    print('总特征维度为:', dataset_fin.shape)
    print('总标签维度为:', label_fin.shape)
    p_train = p_prior + r'DeepSenseing\deepsense-DataSet\train.pickle'
    p_test = p_prior + r'DeepSenseing\deepsense-DataSet\test.pickle'
    SaveFile(data=(dataset_fin, label_fin), savepickle_p=p_train)  # 存储训练或测试数据

if __name__ == '__main__':
    p_prior = r'F:\\'
    # 检验存储数据成pickle文件的数据
    data_fft(p_prior= p_prior)
    data, label = LoadFile(p= p_prior+r'DeepSenseing\deepsense-DataSet\train.pickle')
    print(data.shape, label.shape)
    print(data.dtype, label.dtype)

