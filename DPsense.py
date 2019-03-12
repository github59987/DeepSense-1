#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: DPsense
@time: 2019/3/12 19:13
@desc:
'''
import tensorflow as tf
import numpy as np
from tensorflow_own.AllNet import CNN, RNN, FNN
from tensorflow_own.TestEvaluation import Evaluation
from tensorflow_own.TFrecord_operation import FileoOperation
from tensorflow_own.Routine_operation import SaveFile, LoadFile, Summary_Visualization
# CONST
CONV_LEN_1 = 3
CONV_LEN_2 = 3  # 4
CONV_LEN_3 = 4  # 5
CONV_MEG_1 = 8
CONV_MEG_2 = 6
CONV_MEG_3 = 4
CONV_KEEP_PROB = 0.8
WIDTH = 10
T = 100 // 10
OUT_NUM = 64
epoch = 1000000
DATASET_NUM = 600
BATCH_SIZE = 50
TRAIN_STEPS = (DATASET_NUM // BATCH_SIZE) * epoch #总的数据量/batch_size及为需要分几次读入所有样本

def sub_conv_bn_dropout(CNN_object, op, is_training, strides):
    '''
    对单层数据进行卷积、bn、dropout处理
    :param CNN_object: CNN对象节点
    :param op: Tensor, 待处理张量
    :param is_training: bool, 指示当前状态
    :param strides: iterable, 卷积核各个维度的步伐
    :return: dropout处理后的张量节点
    '''
    conv = CNN_object.convolution(input=op, padding='VALID', strides=strides)
    conv_bn = CNN_object.batch_normoalization(input=conv, is_training=is_training)
    # bn形状为:(batch_size, size1, size2, size3/T)
    conv_relu = tf.nn.relu(conv_bn)
    axis = conv_relu.get_shape().as_list()
    conv_dropout = tf.nn.dropout(x=conv_relu, keep_prob=CONV_KEEP_PROB,
                                 noise_shape=[axis[0], 1, 1, axis[-1]])
    return conv_dropout

def deepsense_model():
    '''
    DeepSense模型
    :return: None
    '''
    ds_graph = tf.Graph()
    with ds_graph.as_default():
        # 建立摘要对象
        summary_visalization = Summary_Visualization()
        with tf.name_scope('dataset'):
            # 输入维度为batch_size*d*2f*T*k维度的单个传感器输入数据
            #建立TFRecord文件对象
            fileoperation = FileoOperation(
                ftype= tf.float64,#tf.float64
                ttype= tf.float64,
                fshape= (3, 200, 10, 2),
                tshape= (6),
                batch_size= BATCH_SIZE,
                capacity= 600+60*50,
                batch_fun= 'shuffle'
            )
            file_train = r'F:\DeepSenseing\deepsense DataSet\output.tfrecords-*'
            feature_batch, label_batch = fileoperation.ParseDequeue(files=file_train)
            feature_batch, label_batch = tf.cast(feature_batch, tf.float32), tf.cast(label_batch, tf.float32)
            # print(feature_batch.shape)
            #将单个feature_batch矩阵flat、按单个特征向量长度为WIDTH进行重组
            feature_batch_new = tf.reshape(tensor=feature_batch, shape=(-1, WIDTH), name='feature_batch_new')
            print('单个feature_batch按照WIDTH特征宽度重组后的数据维度为: %s' % feature_batch_new.shape)
            #按传感器类比分为三组数据
            input_acc, input_gry, input_mag = tf.split(value=feature_batch, num_or_size_splits=3, axis=1)
            #由于split时各组数据最后一维数据是1,故将各组数据的最后一维去掉
            input_acc = tf.reshape(input_acc, shape=input_acc.get_shape().as_list()[:-1])
            input_gry = tf.reshape(input_gry, shape=input_gry.get_shape().as_list()[:-1])
            input_mag = tf.reshape(input_mag, shape=input_mag.get_shape().as_list()[:-1])
            print('各个传感器一个批次的数据维度为: %s %s %s' % (input_acc, input_gry, input_mag))
        with tf.name_scope('placeholder'):
            is_training = tf.placeholder(tf.bool, name='is_training')
        with tf.name_scope('cnn_part1'):
            with tf.name_scope('conv_para_1'):
                conv_para_1 = {
                    'w1': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(3, 2 * 3 * CONV_LEN_1, 1, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                    'w2': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(1, CONV_LEN_2, OUT_NUM, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                    'w3': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(1, CONV_LEN_3, OUT_NUM, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                }
                summary_visalization.variable_summaries(var=conv_para_1['w1'], name='part1_w1')
                summary_visalization.variable_summaries(var=conv_para_1['w2'], name='part1_w2')
                summary_visalization.variable_summaries(var=conv_para_1['w3'], name='part1_w3')