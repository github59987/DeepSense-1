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
F = 100
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
                fshape= (18000),
                tshape= (6),
                batch_size= BATCH_SIZE,
                capacity= 600+60*50,
                batch_fun= 'shuffle'
            )
            file_train = r'F:\DeepSenseing\deepsense DataSet\output.tfrecords-*'
            feature_batch, label_batch = fileoperation.ParseDequeue(files=file_train)
            feature_batch, label_batch = tf.cast(feature_batch, tf.float32), tf.cast(label_batch, tf.float32)
            # print(feature_batch.shape) #shape=(batch_size, 9*2*F*T)
            #将单个feature_batch矩阵flat、按单个特征向量长度为9*2*F进行重组
            feature_batch_flat = tf.reshape(tensor=feature_batch, shape=(1, -1), name='feature_batch_flat')
            feature_batch_new = tf.reshape(tensor=feature_batch_flat, shape=(-1, 9*2*F), name='feature_batch_new')
            # print('数据按照9*2*F特征维度重组后数据维度为: %s' % feature_batch_new.shape) #shape=(batch_size*T, 9*2*F)
            #按传感器类比分为三组数据
            input_acc, input_gry, input_mag = tf.split(value=feature_batch_new, num_or_size_splits=3, axis=1)
            print('各个传感器一个批次的数据维度为: %s %s %s' % (input_acc.shape, input_gry.shape, input_mag.shape))
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
            # acc部分
            cnn_acc_1 = CNN(x=None, w_conv=conv_para_1['w1'], stride_conv=None,
                                stride_pool=None)  # x:(batch_size*T, 3*2*F)
            input_acc_reshape = cnn_acc_1.reshape(f_vector=input_acc, new_shape=(-1, 3, 2*F, 1))
            conv1_acc = sub_conv_bn_dropout(CNN_object=cnn_acc_1, op=input_acc_reshape, is_training=is_training,
                                            strides=[1, 1, CONV_LEN_1, 1])  # (batch_size*T, :, :, OUT_NUM)

            cnn_acc_2 = CNN(x=None, w_conv=conv_para_1['w2'], stride_conv=None,
                                stride_pool=None)  # (batch_size*T, ax1, ax2, OUT_NUM)
            conv2_acc = sub_conv_bn_dropout(CNN_object=cnn_acc_2, op=conv1_acc, is_training=is_training,
                                            strides=[1, 1, CONV_LEN_2, 1])  # (batch_size*T, :, :, OUT_NUM)

            cnn_acc_3 = CNN(x=None, w_conv=conv_para_1['w3'], stride_conv=None,
                                stride_pool=None)  # (batch_size*T, :, :, OUT_NUM)
            conv3_acc = sub_conv_bn_dropout(CNN_object=cnn_acc_3, op=conv2_acc, is_training=is_training,
                                            strides=[1, 1, CONV_LEN_3, 1])  # (batch_size*T, :, :, OUT_NUM)

            # gry部分
            cnn_gry_1 = CNN(x=None, w_conv=conv_para_1['w1'], stride_conv=None,
                            stride_pool=None)  # x:(batch_size*T, 3*2*F)
            input_gry_reshape = cnn_gry_1.reshape(f_vector=input_gry, new_shape=(-1, 3, 2 * F, 1))
            conv1_gry = sub_conv_bn_dropout(CNN_object=cnn_gry_1, op=input_gry, is_training=is_training,
                                            strides=[1, 1, CONV_LEN_1, 1])  # (batch_size*T, :, :, OUT_NUM)

            cnn_gry_2 = CNN(x=None, w_conv=conv_para_1['w2'], stride_conv=None,
                            stride_pool=None)  # (batch_size*T, :, :, OUT_NUM)
            conv2_gry = sub_conv_bn_dropout(CNN_object=cnn_gry_2, op=conv1_gry, is_training=is_training,
                                            strides=[1, 1, CONV_LEN_2, 1])  # (batch_size*T, :, :, OUT_NUM)

            cnn_gry_3 = CNN(x=None, w_conv=conv_para_1['w3'], stride_conv=None,
                            stride_pool=None)  # (batch_size*T, :, :, OUT_NUM)
            conv3_gry = sub_conv_bn_dropout(CNN_object=cnn_gry_3, op=conv2_gry, is_training=is_training,
                                            strides=[1, 1, CONV_LEN_3, 1])  # (batch_size*T, :, :, OUT_NUM)

            # mag部分
            # cnn_mag_1 = CNN(x=None, w_conv=conv_para_1['w1'], stride_conv=[1, 1, CONV_LEN_1, 1],
            #                 stride_pool=None)  # x:(batch_size*T, 3*2*F)
            # input_mag_reshape = cnn_mag_1.reshape(f_vector=input_mag, new_shape=(-1, 3, 2 *F, 1))
            # conv1_mag = sub_conv_bn_dropout(CNN_object=cnn_mag_1, op=input_mag_reshape, is_training=is_training,
            #                                 strides=[1, 1, CONV_LEN_1, 1]) # (batch_size*T, :, :, OUT_NUM)
            #
            # cnn_mag_2 = CNN(x=None, w_conv=conv_para_1['w2'], stride_conv=[1, 1, CONV_LEN_2, 1],
            #                 stride_pool=None)  # (batch_size*T, :, :, OUT_NUM)
            # conv2_mag = sub_conv_bn_dropout(CNN_object=cnn_mag_2, op=conv1_mag, is_training=is_training,
            #                                 strides=[1, 1, CONV_LEN_2, 1])  # (batch_size*T, :, :, OUT_NUM)
            #
            # cnn_mag_3 = CNN(x=None, w_conv=conv_para_1['w3'], stride_conv=[1, 1, CONV_LEN_3, 1],
            #                 stride_pool=None)  # (batch_size*T, :, :, OUT_NUM)
            # conv3_mag = sub_conv_bn_dropout(CNN_object=cnn_mag_3, op=conv2_mag, is_training=is_training,
            #                                 strides=[1, 1, CONV_LEN_1, 1]) # (batch_size*T, :, :, OUT_NUM)

            # flat、concat节点conv3_acc和conv3_gry(和conv3_mag)
            # 分别对各个传感器经过第一层输出进行axis=1上的flat
            axis = conv3_acc.get_shape().as_list()
            conv3_acc_flat = tf.reshape(tensor=conv3_acc, shape=(-1, axis[1]*axis[2]*axis[3]))
            conv3_gry_flat = tf.reshape(tensor=conv3_gry, shape=(-1, axis[1]*axis[2]*axis[3]))
            # conv3_mag_flat = tf.reshape(tensor=conv3_mag, shape=(-1, axis[1]*axis[2]*axis[3]))
            #分别对各个flat后的批次张量增加维并组合为4维
            conv3_acc_fin = tf.expand_dims(input=conv3_acc_flat, axis=1)
            conv3_gry_fin = tf.expand_dims(input=conv3_gry_flat, axis=1)
            # conv3_mag_fin = tf.expand_dims(input=conv3_mag_flat, axis=1)
            cnn_concat1 = tf.concat([conv3_acc_fin, conv3_gry_fin], axis=1)
            # cnn_concat1 = tf.concat([conv3_acc_fin, conv3_gry_fin, conv3_mag_fin], axis=1)
            print('第一部分卷积层输出单个批次维度为: %s' % cnn_concat1.shape) #shape=(batch_size*T, 2(3), :)

        # 第二部分卷积层（结构和第一部分每个传感器的卷积层结构相似）
        with tf.name_scope('cnn_part2'):
            with tf.name_scope('conv_para_2'):
                conv_para_2 = {
                    'w1': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(2, 2 * 3 * CONV_MEG_1, 1, OUT_NUM), mean=0,
                                                              stddev=1), dtype=tf.float32),
                    'w2': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(1, CONV_MEG_2, OUT_NUM, OUT_NUM), mean=0,
                                                              stddev=1), dtype=tf.float32),
                    'w3': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(1, CONV_MEG_3, OUT_NUM, OUT_NUM), mean=0,
                                                              stddev=1), dtype=tf.float32),
                }
                summary_visalization.variable_summaries(var=conv_para_2['w1'], name='part2_w1')
                summary_visalization.variable_summaries(var=conv_para_2['w2'], name='part2_w2')
                summary_visalization.variable_summaries(var=conv_para_2['w3'], name='part2_w3')

            cnn_merge_1 = CNN(x=cnn_concat1, w_conv=conv_para_2['w1'], stride_conv=None,
                              stride_pool=None)  # x:(batch_size*T, 2(3), :)
            conv1_merge = sub_conv_bn_dropout(CNN_object=cnn_merge_1, op=cnn_merge_1, is_training=is_training,
                                     strides=[1, 1, CONV_MEG_1, 1])  # (batch_size*T, :, :, OUT_NUM)

            cnn_merge_2 = CNN(x=None, w_conv=conv_para_2['w2'], stride_conv=None,
                              stride_pool=None)  # (batch_size*T, :, :, OUT_NUM)
            conv2_merge = sub_conv_bn_dropout(CNN_object=cnn_merge_2, op=conv1_merge, is_training=is_training,
                                     strides=[1, 1, CONV_MEG_2, 1])  # (batch_size*T, :, :, OUT_NUM)

            cnn_merge_3 = CNN(x=None, w_conv=conv_para_2['w3'], stride_conv=None,
                              stride_pool=None)  # (batch_size*T, :, :, OUT_NUM)
            cnn_fin = sub_conv_bn_dropout(CNN_object=cnn_merge_3, op=conv2_merge, is_training=is_training,
                                     strides=[1, 1, CONV_MEG_3, 1])  # (batch_size*T, :, :, OUT_NUM)
            # print('第二部分卷积层输出单个批次维度为: %s' % cnn_fin.shape)
            #对卷积层的输出在后三个维度上进行flat并组合WIDTH参数在各个特征向量的最后
            axis_cnn = cnn_fin.get_shape().as_list()
            cnn_flat = tf.reshape(tensor=cnn_fin, shape=(-1, axis_cnn[1]*axis_cnn[2]*axis_cnn[3])) #(batch_size*T, :)
            const_WIDTH = tf.constant(value=np.ones(shape=(axis_cnn[0], 1)*WIDTH), dtype= tf.float32)
            cnn_WIDTH = tf.concat([cnn_flat, const_WIDTH], axis=1)
            print('组合WIDTH后的维度为: %s' % cnn_WIDTH.shape)


