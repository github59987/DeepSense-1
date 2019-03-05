#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: data_operation
@time: 2019/3/4 17:04
@desc:
'''
import tensorflow as tf
import numpy as np
from tensorflow_own.Routine_operation import SaveFile, LoadFile
from tensorflow_own.TFrecord_operation import FileoOperation
def data_fft(p_prior):
    '''
    对原始数据按照DeepSense模型要求进行数据预处理
    :param p_prior: str, 数据路径前缀
    :return: None
    '''
    # scalar
    CLASS_NUM = 6
    WIDTH = 10
    PER_LENGTH = 100
    ALL_NUM = 1000
    F = 100
    # 矩阵初始化区域
    feature_all = np.zeros(shape=([1]), dtype=np.float64)
    label_all = np.zeros(shape=([1]), dtype=np.float64)
    dataset_fin = np.zeros(shape=[1], dtype=np.float64)
    sub_dataset_2_fft_all = np.zeros(shape=([1]), dtype=np.float64)
    dataset_per_mode = np.zeros(shape=[1], dtype=np.float64)
    label = np.zeros(shape=(ALL_NUM // PER_LENGTH, CLASS_NUM), dtype=np.float64)
    # 导入数据
    for i in range(CLASS_NUM):
        print('正在导入第%s个模式数据' % (i+1))
        p = p_prior + r'DeepSenseing\ICT DataSet\Label_%s.txt' % (i + 1)
        with open(p, 'r') as file:
            dataset = np.loadtxt(file, delimiter=',', skiprows=0)[:ALL_NUM, :]
            # 删除中间3列线性加速度
            dataset = np.delete(dataset, [3, 4, 5, 12, 13], axis=1)
            # print('正在对第%s个模式数据进行转置' % i)
            dataset_transpose = dataset.T
            # print('转之后数据维度为:', dataset_transpose.shape)
            # 原始数据形式：(acc_x, acc_y, acc_z, gry_x, gry_y, gry_z, mag_x, mag_y, mag_z, :)
            # 转换为:(:, d, 2f, T, K)
            for per_data in range(0, ALL_NUM, PER_LENGTH):
                # print('正在对第%s个模式的%s-%s个数据进行操作' % (i, per_data, per_data+PER_LENGTH))
                # 从原始数据集中截取单个数据长度
                sub_dataset_1 = dataset_transpose[:, per_data:per_data + PER_LENGTH]
                # 从单个数据中横向截取出每个width长度子数据并计算fft
                for sub_per_data in range(0, PER_LENGTH, WIDTH):
                    sub_dataset_2 = sub_dataset_1[:, sub_per_data: sub_per_data + WIDTH]
                    # 对各行进行fft
                    sub_dataset_2_fft = np.abs(np.fft.fft(a=sub_dataset_2, n=2 * F, axis=1))  # (d, 2f)
                    # 变换维度
                    sub_dataset_2_fft = sub_dataset_2_fft[:, :, np.newaxis]
                    # 组合T个时间段的fft
                    sub_dataset_2_fft_all = sub_dataset_2_fft if sub_per_data == 0 else \
                        np.concatenate((sub_dataset_2_fft_all, sub_dataset_2_fft), axis=2)  # (d, 2f, T)
                    # print(sub_dataset_2_fft_all.shape)
                # print('正在对K个传感器的三维坐标进行分割重组')
                # 将K个传感器的三维坐标进行分割
                k1, k2, k3 = sub_dataset_2_fft_all[0:3, :, :], sub_dataset_2_fft_all[3:6, :, :], sub_dataset_2_fft_all[
                                                                                                 6:9, :, :]
                # 变换维度
                k1, k2, k3 = k1[:, :, :, np.newaxis], k2[:, :, :, np.newaxis], k3[:, :, :, np.newaxis]  # (d, 2f, T, 1)
                # sub_dataset_fft = np.concatenate((k1, k2, k3), axis= 3) #组合三个传感器
                sub_dataset_fft = np.concatenate((k1, k2), axis=3)  # 组合acc和gry两个传感器 (d, 2f, T, K)
                # 变换维度
                sub_dataset_fft = sub_dataset_fft[np.newaxis, :, :, :, :]
                # 单类交通模式特征
                dataset_fin = sub_dataset_fft if per_data == 0 else \
                    np.concatenate((dataset_fin, sub_dataset_fft), axis=0)  # (:, d, 2f, T, K)
                # print('第%s个模式特征维度为: ' % i, dataset_fin.shape)
            label[:, i] = 1
            # 最终所有交通模式数据特征和标签
            feature_all = dataset_fin if i == 0 else \
                np.concatenate((feature_all, dataset_fin), axis=0)
            label_all = label if i == 0 else np.vstack((label_all, label))
            print('总特征维度为:', feature_all.shape)
            print('总标签维度为:', label_all.shape)

    p_train = p_prior + r'DeepSenseing\deepsense DataSet\train.pickle'
    SaveFile(data= (feature_all, label_all), savepickle_p= p_train)

def save_TFRecord(p_prior):
    '''
    将数据按照TFRecord格式进行存储
    :param p_prior: 数据存储、导入路径前缀
    :return: None
    '''
    #建立TFRecord格式类转换对象
    fileoperation = FileoOperation(
        p_in= p_prior + r'DeepSenseing\deepsense DataSet\train.pickle',
        filename= p_prior + r'DeepSenseing\deepsense DataSet\output.tfrecords-%.5d-of-%.5d',
        num_shards= 5,
        instance_per_shard= 10*6//5,
        read_in_fun= LoadFile
    )
    fileoperation.file2TFRecord()

if __name__ == '__main__':
    p_prior = r'F:\\'
    # 检验存储数据成pickle文件的数据
    data_fft(p_prior= p_prior)
    data, label = LoadFile(p= p_prior+r'DeepSenseing\deepsense DataSet\train.pickle')
    print(data.shape, label.shape)
    print(data.dtype, label.dtype)
    save_TFRecord(p_prior= p_prior)














