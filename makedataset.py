#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: makedataset
@time: 2019/3/11 18:46
@desc:
'''
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
    T = 10
    ALL_NUM = 10000
    F = 100
    # 矩阵初始化区域
    label_all = np.zeros(shape=([1]), dtype=np.float32)
    dataset_all = np.zeros(shape=[1], dtype=np.float64)
    # 导入数据
    for i in range(CLASS_NUM):
        print('正在导入第%s个模式数据' % (i+1))
        p = p_prior + r'DeepSenseing\ICT DataSet\Label_%s.txt' % (i + 1)
        # 矩阵初始化区域
        per_group_dataset = np.zeros(shape=[1], dtype=np.float32)
        dataset_fin = np.zeros(shape=[1], dtype=np.float32)
        label = np.zeros(shape=(ALL_NUM // WIDTH // T, CLASS_NUM), dtype=np.float64)
        with open(p, 'r') as file:
            dataset = np.loadtxt(file, delimiter=',', skiprows=0)[:ALL_NUM, :]
            # 删除中间3列线性加速度
            dataset = np.delete(dataset, [3, 4, 5, 12, 13], axis=1)
            # print('正在对第%s个模式数据进行转置' % i)
            dataset_transpose = dataset.T
            # print('转之后数据维度为:', dataset_transpose.shape)
            # 原始数据形式：(acc_x, acc_y, acc_z, gry_x, gry_y, gry_z, mag_x, mag_y, mag_z, :)
            #在axis=1的维度上按照WIDTH截取总数据集长度并在axis=0的维度上拼接,拼接后维度：(ALL_NUM//WIDTH*9, WIDTH)
            dataset_convert = np.concatenate(np.split(ary=dataset_transpose,
                                                      indices_or_sections=ALL_NUM//WIDTH, axis=1), axis=0)
            #在axis=0维度上分别计算各个传感器各个维度在各个时间片WIDTH内的fft变换幅值
            for time_sheet in np.split(ary=dataset_convert, indices_or_sections=ALL_NUM//WIDTH, axis=0): #ALL_NUM//WIDTH个(9, WIDTH)
                per_width_allsense = np.abs(np.fft.fft(a=time_sheet, n=2*F, axis=1)) #(9, 2*F)
                #flat二维矩阵并组合所有flat后的数据
                dataset_fin = np.vstack((dataset_fin, np.reshape(a=per_width_allsense, newshape=(1, -1)))) if \
                    dataset_fin.any() else np.reshape(a=per_width_allsense, newshape=(1, -1)) #(:, 9*2*F),最终总维度为:(ALL_NUM//WIDTH, 9*2*F)
            for per_data in np.split(ary=dataset_fin, indices_or_sections=ALL_NUM//WIDTH//T, axis=0): #T是单个训练样本包含多少个WIDTH数量
                #flat(T, 9*2*F)维数据并在axis=0维进行拼接
                per_data_flat = np.reshape(a=per_data, newshape=(1, -1)) #(1, 9*2*F*T)
                per_group_dataset = np.vstack((per_group_dataset, per_data_flat)) if per_group_dataset.any() else per_data_flat #(:, 9*2*F*T)
        #组合各组标签的特征数据
        dataset_all = np.vstack((dataset_all, per_group_dataset)) if dataset_all.any() else per_group_dataset
        label[:, i] = 1
        label_all = label if i == 0 else np.vstack((label_all, label))
    # 对数据特征和标签进行shuffle
    rng = np.random.RandomState(0)
    rng.shuffle(dataset_all)
    rng.shuffle(label_all)
    print('总特征维度为:', dataset_all.shape)
    print('总标签维度为:', label_all.shape)
    p_train = p_prior + r'DeepSenseing\deepsense DataSet\train.pickle'
    SaveFile(data=(dataset_all, label_all), savepickle_p=p_train)

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
        instance_per_shard= 100*6//5,
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
