#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: deepsense
@time: 2019/3/2 22:54
@desc:
'''
import tensorflow as tf
import numpy as np
from tensorflow_own.AllNet import CNN, RNN, FNN
from tensorflow_own.TestEvaluation import Evaluation
from tensorflow_own.TFrecord_operation import FileoOperation
from tensorflow_own.Routine_operation import SaveFile, LoadFile
# CONST
CONV_LEN_1 = 3
CONV_LEN_2 = 3  # 4
CONV_LEN_3 = 4  # 5
CONV_MEG_1 = 8
CONV_MEG_2 = 6
CONV_MEG_3 = 4
CONV_KEEP_PROB = 0.8
WIDTH = 10
T = 100
OUT_NUM = 64
epoch = 1000000
DATASET_NUM = 600
BATCH_SIZE = 50
TRAIN_STEPS = (DATASET_NUM // BATCH_SIZE) * epoch #总的数据量/batch_size及为需要分几次读入所有样本

def batch_conv(CNN_object, op, is_training):
    '''
    对列表中每一个节点进行卷积、bn、dropout处理
    :param CNN_object: CNN对象节点
    :param op: Tensor, 待处理张量
    :param is_training: bool placeholder, 指示当前状态
    :return: 单个节点
    '''
    def sub_conv_bn_dropout(CNN_object, op, is_training):
        '''
        对单层数据进行卷积、bn、dropout处理
        :param CNN_object: CNN对象节点
        :param op: Tensor, 待处理张量
        :param is_training: bool, 指示当前状态
        :return: dropout处理后的张量节点
        '''
        conv = CNN_object.convolution(x=op, padding='VALID')
        conv_bn = CNN_object.batch_normoalization(input=conv, is_training=is_training)
        # bn形状为:(batch_size, size1, size2, size3/T)
        conv_relu = tf.nn.relu(conv_bn)
        axis = conv_relu.get_shape().as_list()
        conv_dropout = tf.nn.dropout(x=conv_relu, keep_prob=CONV_KEEP_PROB,
                                     noise_shape=[axis[0], 1, 1, axis[-1]])
        return conv_dropout

    # 所有处理后的结点存储列表
    conv_dropouts = []
    for per_op in tf.split(value=op, num_or_size_splits=T, axis=3): #per_op.shape= (batch_size, size1, size2, size3/T)
        conv_dropout = sub_conv_bn_dropout(CNN_object, per_op, is_training) #(batch_size, size1, size2, OUT_NUM=64)
        conv_dropouts.append(conv_dropout)

    return tf.concat(values=conv_dropouts, axis=3) #(batch_size, size2, size2, T*OUT_NUM)

def dim_transpose(op):
    '''
    对节点按最后维切T个op、在最后增加1维，合并
    :param op: shape=(batch_size, size1, size2, OUT_NUM*T), 待操作节点
    :return: op, shape= (batch_size, size1, size2, OUT_NUM, T)
    '''
    op_list = tf.split(value=op, num_or_size_splits= T, axis= 3)
    for i in range(len(op_list)):
        op_list[i] = tf.expand_dims(input=op_list[i], axis=-1)

    return  tf.concat(values=op_list, axis= -1)

def deepsense_model():
    '''
    DeepSense模型
    :return: None
    '''
    ds_graph = tf.Graph()
    with ds_graph.as_default():
        with tf.name_scope('dataset'):
            # 输入维度为batch_size*d*2f*T*k维度的单个传感器输入数据
            #建立TFRecord文件对象
            fileoperation = FileoOperation(
                ftype= tf.float64,
                ttype= tf.float64,
                fshape= (3, 200, 10, 2),
                tshape= (6),
                batch_size= BATCH_SIZE,
                capacity= 600+60*50,
                batch_fun= 'shuffle'
            )
            file_train = r'F:\DeepSenseing\deepsense DataSet\output.tfrecords-*'
            feature_batch, label_batch = fileoperation.ParseDequeue(files= file_train)
            input_acc, input_gry, input_mag = tf.split(value=feature_batch, num_or_size_splits=3, axis=5)
        with tf.name_scope('placeholder'):
            is_training = tf.placeholder(tf.bool, name='is_training')
        with tf.name_scope('cnn_part1'):
            with tf.name_scope('conv_para_1'):
                conv_para_1 = {
                    'w1': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(3, 2 * 3 * CONV_LEN_1, T, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                    'w2': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(1, CONV_LEN_2, T, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                    'w3': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(1, CONV_LEN_3, T, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                }
            # acc部分
            cnn_acc_1 = CNN(x=None, w_conv=conv_para_1['w1'], stride_conv=[1, 1, CONV_LEN_1, 1],
                            stride_pool=None)  # x:(batch_size, d, 2f, T)
            conv1_acc = batch_conv(CNN_object=cnn_acc_1, op=input_acc, is_training=is_training)  #(batch_size, size1, size2, T*OUT_NUM)

            cnn_acc_2 = CNN(x=None, w_conv=conv_para_1['w2'], stride_conv=[1, 1, CONV_LEN_2, 1],
                            stride_pool=None)  # x:(batch_size, size1, size2, T*OUT_NUM)
            conv2_acc = batch_conv(CNN_object=cnn_acc_2, op=conv1_acc, is_training=is_training)  # (batch_size, size1, size2, T*OUT_NUM)

            cnn_acc_3 = CNN(x=None, w_conv=conv_para_1['w3'], stride_conv=[1, 1, CONV_LEN_3, 1],
                            stride_pool=None)  # x:(batch_size, size1, size2, T*OUT_NUM)
            conv3_acc = batch_conv(CNN_object=cnn_acc_3, op=conv2_acc, is_training=is_training)  # (batch_size, size1, size2, T*OUT_NUM)

            # gry部分
            cnn_gry_1 = CNN(x=None, w_conv=conv_para_1['w1'], stride_conv=[1, 1, CONV_LEN_1, 1],
                            stride_pool=None)  # x:(batch_size, d, 2f, T)
            conv1_gry = batch_conv(CNN_object=cnn_gry_1, op=input_gry, is_training=is_training) # (batch_size, size1, size2, T*OUT_NUM)

            cnn_gry_2 = CNN(x=None, w_conv=conv_para_1['w2'], stride_conv=[1, 1, CONV_LEN_2, 1],
                            stride_pool=None)  # x:(batch_size, size1, size2, T*OUT_NUM)
            conv2_gry = batch_conv(CNN_object=cnn_gry_2, op=conv1_gry, is_training=is_training) # (batch_size, size1, size2, T*OUT_NUM)

            cnn_gry_3 = CNN(x=None, w_conv=conv_para_1['w3'], stride_conv=[1, 1, CONV_LEN_3, 1],
                            stride_pool=None)  # x:(batch_size, size1, size2, T*OUT_NUM)
            conv3_gry = batch_conv(CNN_object=cnn_gry_3, op=conv2_gry, is_training=is_training) # (batch_size, size1, size2, T*OUT_NUM)

            # # mag部分
            # cnn_mag_1 = CNN(x=None, w_conv=conv_para_1['w1'], stride_conv=[1, 1, CONV_LEN_1, 1],
            #                 stride_pool=None)  # x:(batch_size, d, 2f, T)
            # conv1_mag = batch_conv(CNN_object=cnn_mag_1, op=input_mag, is_training=is_training) # (batch_size, size1, size2, T*OUT_NUM)
            #
            # cnn_mag_2 = CNN(x=None, w_conv=conv_para_1['w2'], stride_conv=[1, 1, CONV_LEN_2, 1],
            #                 stride_pool=None)  # x:(batch_size, size1, size2, T*OUT_NUM)
            # conv2_mag = batch_conv(CNN_object=cnn_mag_2, op=conv1_mag, is_training=is_training) # (batch_size, size1, size2, T*OUT_NUM)
            #
            # cnn_mag_3 = CNN(x=None, w_conv=conv_para_1['w3'], stride_conv=[1, 1, CONV_LEN_3, 1],
            #                 stride_pool=None)  # x:(batch_size, size1, size2, T*OUT_NUM)
            # conv3_mag = batch_conv(CNN_object=cnn_mag_3, op=conv2_mag, is_training=is_training) # (batch_size, size1, size2, T*OUT_NUM)

            # flat、concat节点conv3_acc_output和conv3_gry_output ,此时shape=(batch_size, size1, size2, OUT_NUM*T)
            #分别对各个传感器经过第一层输出进行维度转换
            conv3_acc = dim_transpose(conv3_acc) #(batch_size, size1, size2, OUT_NUM, T)
            conv3_gry = dim_transpose(conv3_gry)
            # conv3_mag = dim_transpose(conv3_mag)
            axis = conv3_acc.get_shape().as_list()
            #将所有flat后的数据增加第二维度以便组合
            conv3_acc_flat = cnn_acc_3.reshape(f_vector=conv3_acc, new_shape=(-1, axis[1] * axis[2] * axis[3], T))
            conv3_acc_flat_dim = tf.expand_dims(input=conv3_acc_flat, axis=1)
            conv3_gry_flat = cnn_gry_3.reshape(f_vector=conv3_gry, new_shape=conv3_acc_flat.get_shape().as_list())
            conv3_gry_flat_dim = tf.expand_dims(input=conv3_gry_flat, axis=1)
            # conv3_mag_flat = cnn_mag_3.reshape(f_vector=conv3_mag, new_shape=conv3_acc_flat.get_shape().as_list())
            # conv3_mag_flat_dim = tf.expand_dims(input=conv3_mag_flat, axis=1)

            concat_flat = tf.concat([conv3_acc_flat_dim, conv3_gry_flat_dim], axis=1)  # shape= (batch_size, 2, axis[1]*axis[2]*axis[3], T)

        # 第二部分卷积层（结构和第一部分每个传感器的卷积层结构相似）
        with tf.name_scope('cnn_part2'):
            with tf.name_scope('conv_para_2'):
                conv_para_2 = {
                    'w1': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(2, 2 * 3 * CONV_MEG_1, T, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                    'w2': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(1, CONV_MEG_2, T, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                    'w3': tf.Variable(
                        initial_value=tf.truncated_normal(shape=(1, CONV_MEG_3, T, OUT_NUM), mean=0, stddev=1),
                        dtype=tf.float32),
                }
            cnn_merge_1 = CNN(x=concat_flat, w_conv=conv_para_2['w1'], stride_conv=[1, 1, CONV_MEG_1, 1],
                              stride_pool=None)  # x:(batch_size, 2, axis[1]*axis[2]*axis[3], T)
            conv1_merge = batch_conv(CNN_object=cnn_merge_1, op=concat_flat, is_training=is_training) #(batch_size, size1, size2, T*OUT_NUM)

            cnn_merge_2 = CNN(x=None, w_conv=conv_para_2['w2'], stride_conv=[1, 1, CONV_MEG_2, 1],
                              stride_pool=None)  # x:(batch_size, size1, size2, T*OUT_NUM)
            conv2_merge = batch_conv(CNN_object=cnn_merge_2, op=conv1_merge, is_training=is_training) #(batch_size, size1, size2, T*OUT_NUM)

            cnn_merge_3 = CNN(x=None, w_conv=conv_para_2['w3'], stride_conv=[1, 1, CONV_MEG_3, 1],
                              stride_pool=None)  # x:(batch_size, size1, size2, T*OUT_NUM)
            conv3_merge = batch_conv(CNN_object=cnn_merge_2, op=conv2_merge, is_training=is_training) #(batch_size, size1, size2, T*OUT_NUM)

            #维度变换
            conv3_merge = dim_transpose(conv3_merge) #(batch_size, size1, size2, OUT_NUM, T)
            axis = conv3_merge.get_shape().as_list()
            conv3_output = cnn_merge_3.reshape(f_vector=conv3_merge, new_shape=(-1, axis[1]*axis[2]*axis[3], T))

            #在各个数据末尾加入WIDE值
            axis = conv3_output.get_shape().as_list()
            wide_Tensor = tf.constant(np.ones(shape=(axis[0], 1, axis[-1]))*WIDTH, dtype=tf.float32)
            conv_output= tf.concat(values=(conv3_output, wide_Tensor), axis=1)

            # multiGRU部分
            gru = RNN(x=conv_output, max_time=T, num_units=128)
            rnn_outputs, _ = gru.dynamic_rnn(style='GRU', output_keep_prob=0.8)  # rnn_outputs:(batch_size, T, num_units)
            # 将T个时刻的所有输出求平均值
            rnn_output = tf.reduce_mean(rnn_outputs, axis=2)  # rnn_output:(batch_size, num_units)

        with tf.name_scope('output_layer'):
            w_size_in = rnn_outputs.get_shape().as_list()[-1]
            w_size_out = 8
            w_out = tf.Variable(tf.truncated_normal(shape=(w_size_in, w_size_out), mean=0, stddev=1), dtype=tf.float32)
            b = tf.Variable(tf.truncated_normal(shape=([w_size_out]), mean=0, stddev=1), dtype=tf.float32)
            fin_output = tf.matmul(rnn_output, w_out) + b
        with tf.name_scope('evaluation'):
            # 定义softmax交叉熵和损失函数以及精确度函数
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fin_output, labels=label_batch))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)
            evaluation = Evaluation(one_hot=True, logit=tf.nn.softmax(fin_output), label=label_batch, regression_pred=None,
                                    regression_label=None)
            acc = evaluation.acc_classification()
        with tf.name_scope('etc'):
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=ds_graph) as sess:
        sess.run(init_global)
        sess.run(init_local)
        # 线程调配管理器
        coord, threads = FileoOperation.coord_threads(sess=sess)
        train_steps = TRAIN_STEPS
        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                # feature_batch_, target_batch_ = sess.run([feature_batch, label_batch])
                # print(feature_batch_.shape, target_batch_.shape)
                _ = sess.run(optimizer, feed_dict={is_training: True})
                train_steps -= 1
                if train_steps <= 0:
                    coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True
                if train_steps % (12 * 100) == 0: #读入批次总数
                    acc = sess.run(acc, feed_dict={is_training: False})
                    print(acc)
        except tf.errors.OutOfRangeError:
            print('%s次训练完成' % (TRAIN_STEPS//epoch))
        finally:
            # When done, ask the threads to stop. 请求该线程停止
            coord.request_stop()
            # And wait for them to actually do it. 等待被指定的线程终止
            coord.join(threads)

if __name__ == '__main__':
    deepsense_model()






















if __name__ == '__main__':
    pass