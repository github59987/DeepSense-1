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
def deepsense_model(dataset):
    ''''''
    # CONST
    CONV_LEN_1 = 3
    CONV_LEN_2 = 3  # 4
    CONV_LEN_3 = 4  # 5
    CONV_MEG_1 = 8
    CONV_MEG_2 = 6
    CONV_MEG_3 = 4
    CONV_KEEP_PROB = 0.8
    T = 20
    OUT_NUM = 64
    ds_graph = tf.Graph()
    with ds_graph.as_default():
        with tf.name_scope('dataset'):
            # 输入维度为batch_size*d*2f*T*k维度的单个传感器输入数据
            #建立TFRecord文件对象
            fileoperation = FileoOperation(
                ftype= tf.float64,
                ttype= tf.float64,
                fshape= (60, 3, 200, 10, 2),
                tshape= (60, 6),
                batch_size= 5,
                capacity= 60+6*5,
                batch_fun= tf.train.shuffle_batch
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
            cnn_acc_1 = CNN(x=input_acc, w_conv=conv_para_1['w1'], stride_conv=[1, 1, CONV_LEN_1, 1],
                            stride_pool=None)  # x:(batch_size, d, 2f, T)
            conv1_acc = cnn_acc_1.depthwise_convolution(padding='VALID')
            conv1_acc_bn = cnn_acc_1.batch_normoalization(input=conv1_acc,
                                                          is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM)
            conv1_acc_relu = tf.nn.relu(conv1_acc_bn)
            conv1_axis_1, conv1_axis_2, conv1_axis_3, conv1_axis_4 = conv1_acc_relu.get_shape().as_list()
            conv1_acc_dropout = tf.nn.dropout(x=conv1_acc_relu, keep_prob=CONV_KEEP_PROB,
                                              noise_shape=[conv1_axis_1, 1, 1, conv1_axis_4])
            cnn_acc_2 = CNN(x=conv1_acc_dropout, w_conv=conv_para_1['w2'], stride_conv=[1, 1, CONV_LEN_2, 1],
                            stride_pool=None)  # x:(batch_size, d, 2f-, T*OUT_NUM)
            conv2_acc = cnn_acc_2.depthwise_convolution(padding='VALID')
            conv2_acc_bn = cnn_acc_2.batch_normoalization(input=conv2_acc,
                                                          is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM*OUT_NUM)
            conv2_acc_relu = tf.nn.relu(conv2_acc_bn)
            # 变换深度卷积后的维度
            conv2_axis_1, conv2_axis_2, conv2_axis_3, conv2_axis_4 = conv2_acc_relu.get_shape().as_list()
            conv2_acc_dropout = tf.nn.dropout(x=conv2_acc_relu, keep_prob=CONV_KEEP_PROB,
                                              noise_shape=[conv2_axis_1, 1, 1, conv2_axis_4])
            conv2_output = cnn_acc_2.reshape(f_vector=conv2_acc_dropout,
                                             new_shape=(conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM, OUT_NUM, T))
            conv2_output = tf.reduce_sum(conv2_output, axis=3)  # 此处可以试axis= 4
            conv2_output = cnn_acc_2.reshape(f_vector=conv2_output, new_shape=(
            conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM * T))  # (batch_size, d, 2f-, T*OUT_NUM)

            cnn_acc_3 = CNN(x=conv2_output, w_conv=conv_para_1['w3'], stride_conv=[1, 1, CONV_LEN_3, 1],
                            stride_pool=None)  # x:(batch_size, d, 2f-, T*OUT_NUM)
            conv3_acc = cnn_acc_3.depthwise_convolution(padding='VALID')
            conv3_acc_bn = cnn_acc_3.batch_normoalization(input=conv3_acc,
                                                          is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM*OUT_NUM)
            conv3_acc_relu = tf.nn.relu(conv3_acc_bn)
            # 变换深度卷积后的维度
            conv3_axis_1, conv3_axis_2, conv3_axis_3, conv3_axis_4 = conv3_acc_relu.get_shape().as_list()
            conv3_acc_dropout = tf.nn.dropout(x=conv3_acc_relu, keep_prob=CONV_KEEP_PROB,
                                              noise_shape=(conv3_axis_1, 1, 1, conv3_axis_4))
            conv3_output = cnn_acc_3.reshape(f_vector=conv3_acc_dropout,
                                             new_shape=(conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM, OUT_NUM, T))
            conv3_acc_output = tf.reduce_sum(conv3_output, axis=3)  # 此处可以试axis=4 shape= (batch_size, d, 2f, OUT_NUM, T)

            # gry部分
            cnn_gry_1 = CNN(x=input_gry, w_conv=conv_para_1['w1'], stride_conv=[1, 1, CONV_LEN_1, 1],
                            stride_pool=None)  # x:(batch_size, d, 2f, T)
            conv1_gry = cnn_gry_1.depthwise_convolution(padding='VALID')
            conv1_gry_bn = cnn_gry_1.batch_normoalization(input=conv1_gry,
                                                          is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM)
            conv1_gry_relu = tf.nn.relu(conv1_gry_bn)
            conv1_axis_1, conv1_axis_2, conv1_axis_3, conv1_axis_4 = conv1_gry_relu.get_shape().as_list()
            conv1_gry_dropout = tf.nn.dropout(x=conv1_gry_relu, keep_prob=CONV_KEEP_PROB,
                                              noise_shape=[conv1_axis_1, 1, 1, conv1_axis_4])
            cnn_gry_2 = CNN(x=conv1_gry_dropout, w_conv=conv_para_1['w2'], stride_conv=[1, 1, CONV_LEN_2, 1],
                            stride_pool=None)  # x:(batch_size, d, 2f-, T*OUT_NUM)
            conv2_gry = cnn_gry_2.depthwise_convolution(padding='VALID')
            conv2_gry_bn = cnn_gry_2.batch_normoalization(input=conv2_gry,
                                                          is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM*OUT_NUM)
            conv2_gry_relu = tf.nn.relu(conv2_gry_bn)
            # 变换深度卷积后的维度
            conv2_axis_1, conv2_axis_2, conv2_axis_3, conv2_axis_4 = conv2_gry_relu.get_shape().as_list()
            conv2_gry_dropout = tf.nn.dropout(x=conv2_gry_relu, keep_prob=CONV_KEEP_PROB,
                                              noise_shape=[conv2_axis_1, 1, 1, conv2_axis_4])
            conv2_output = cnn_gry_2.reshape(f_vector=conv2_gry_dropout,
                                             new_shape=(conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM, OUT_NUM, T))
            conv2_output = tf.reduce_sum(conv2_output, axis=3)  # 此处可以试axis= 4
            conv2_output = cnn_gry_2.reshape(f_vector=conv2_output, new_shape=(
            conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM * T))  # (batch_size, d, 2f-, T*OUT_NUM)

            cnn_gry_3 = CNN(x=conv2_output, w_conv=conv_para_1['w3'], stride_conv=[1, 1, CONV_LEN_3, 1],
                            stride_pool=None)  # x:(batch_size, d, 2f-, T*OUT_NUM)
            conv3_gry = cnn_gry_3.depthwise_convolution(padding='VALID')
            conv3_gry_bn = cnn_gry_3.batch_normoalization(input=conv3_gry,
                                                          is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM*OUT_NUM)
            conv3_gry_relu = tf.nn.relu(conv3_gry_bn)
            # 变换深度卷积后的维度
            conv3_axis_1, conv3_axis_2, conv3_axis_3, conv3_axis_4 = conv3_gry_relu.get_shape().as_list()
            conv3_gry_dropout = tf.nn.dropout(x=conv3_gry_relu, keep_prob=CONV_KEEP_PROB,
                                              noise_shape=(conv3_axis_1, 1, 1, conv3_axis_4))
            conv3_output = cnn_gry_3.reshape(f_vector=conv3_gry_dropout,
                                             new_shape=(conv3_axis_1, conv3_axis_2, conv3_axis_3, OUT_NUM, OUT_NUM, T))
            conv3_gry_output = tf.reduce_sum(conv3_output, axis=3)  # 此处可以试axis=4 shape= (batch_size, d, 2f, OUT_NUM, T)

            # mag部分
            # cnn_mag_1 = CNN(x=input_mag, w_conv=conv_para_1['w1'], stride_conv=[1, 1, CONV_LEN_1, 1],
            #                 stride_pool=None)  # x:(batch_size, d, 2f, T)
            # conv1_mag = cnn_mag_1.depthwise_convolution(padding='VALID')
            # conv1_mag_bn = cnn_mag_1.batch_normoalization(input=conv1_mag,
            #                                               is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM)
            # conv1_mag_relu = tf.nn.relu(conv1_mag_bn)
            # conv1_axis_1, conv1_axis_2, conv1_axis_3, conv1_axis_4 = conv1_mag_relu.get_shape().as_list()
            # conv1_mag_dropout = tf.nn.dropout(x=conv1_mag_relu, keep_prob=CONV_KEEP_PROB,
            #                                   noise_shape=[conv1_axis_1, 1, 1, conv1_axis_4])
            # cnn_mag_2 = CNN(x=conv1_mag_dropout, w_conv=conv_para_1['w2'], stride_conv=[1, 1, CONV_LEN_2, 1],
            #                 stride_pool=None)  # x:(batch_size, d, 2f-, T*OUT_NUM)
            # conv2_mag = cnn_mag_2.depthwise_convolution(padding='VALID')
            # conv2_mag_bn = cnn_mag_2.batch_normoalization(input=conv2_mag,
            #                                               is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM*OUT_NUM)
            # conv2_mag_relu = tf.nn.relu(conv2_mag_bn)
            # # 变换深度卷积后的维度
            # conv2_axis_1, conv2_axis_2, conv2_axis_3, conv2_axis_4 = conv2_gry_relu.get_shape().as_list()
            # conv2_mag_dropout = tf.nn.dropout(x=conv2_mag_relu, keep_prob=CONV_KEEP_PROB,
            #                                   noise_shape=[conv2_axis_1, 1, 1, conv2_axis_4])
            # conv2_output = cnn_gry_2.reshape(f_vector=conv2_mag_dropout,
            #                                  new_shape=(conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM, OUT_NUM, T))
            # conv2_output = tf.reduce_sum(conv2_output, axis=3)  # 此处可以试axis= 4
            # conv2_output = cnn_mag_2.reshape(f_vector=conv2_output, new_shape=(
            # conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM * T))  # (batch_size, d, 2f-, T*OUT_NUM)
            #
            # cnn_mag_3 = CNN(x=conv2_output, w_conv=conv_para_1['w3'], stride_conv=[1, 1, CONV_LEN_3, 1],
            #                 stride_pool=None)  # x:(batch_size, d, 2f-, T*OUT_NUM)
            # conv3_mag = cnn_mag_3.depthwise_convolution(padding='VALID')
            # conv3_mag_bn = cnn_mag_3.batch_normoalization(input=conv3_mag,
            #                                               is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM*OUT_NUM)
            # conv3_mag_relu = tf.nn.relu(conv3_mag_bn)
            # # 变换深度卷积后的维度
            # conv3_axis_1, conv3_axis_2, conv3_axis_3, conv3_axis_4 = conv3_mag_relu.get_shape().as_list()
            # conv3_mag_dropout = tf.nn.dropout(x=conv3_mag_relu, keep_prob=CONV_KEEP_PROB,
            #                                   noise_shape=(conv3_axis_1, 1, 1, conv3_axis_4))
            # conv3_output = cnn_mag_3.reshape(f_vector=conv3_mag_dropout,
            #                                  new_shape=(conv3_axis_1, conv3_axis_2, conv3_axis_3, OUT_NUM, OUT_NUM, T))
            # conv3_mag_output = tf.reduce_sum(conv3_output, axis=3)  # 此处可以试axis=4 shape= (batch_size, d, 2f, OUT_NUM, T)

            # flat、concat节点conv3_acc_output和conv3_gry_output ,此时shape=(batch_size, d, 2f-, OUT_NUM, T)
            conv3_acc_flat = cnn_acc_3.reshape(f_vector=conv3_acc_output,
                                               new_shape=(-1, conv3_axis_2 * conv3_axis_3 * OUT_NUM, T))
            conv3_gry_flat = cnn_gry_3.reshape(f_vector=conv3_gry_output,
                                               new_shape=conv3_acc_flat.get_shape().as_list())
            concat_flat = tf.concat([conv3_acc_flat, conv3_gry_flat],
                                    axis=1)  # shape= (batch_size, 2*conv3_axis_2*conv3_axis_3*OUT_NUM, T)

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
                              stride_pool=None)  # x:(batch_size, d, 2f, T)
            conv1_merge = cnn_merge_1.depthwise_convolution(padding='VALID')
            conv1_merge_bn = cnn_merge_1.batch_normoalization(input=conv1_merge,
                                                              is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM)
            conv1_merge_relu = tf.nn.relu(conv1_merge_bn)
            conv1_axis_1, conv1_axis_2, conv1_axis_3, conv1_axis_4 = conv1_merge_relu.get_shape().as_list()
            conv1_merge_dropout = tf.nn.dropout(x=conv1_merge_relu, keep_prob=CONV_KEEP_PROB,
                                                noise_shape=[conv1_axis_1, 1, 1, conv1_axis_4])
            cnn_merge_2 = CNN(x=conv1_merge_dropout, w_conv=conv_para_2['w2'], stride_conv=[1, 1, CONV_MEG_2, 1],
                              stride_pool=None)  # x:(batch_size, d, 2f-, T*OUT_NUM)
            conv2_merge = cnn_merge_2.depthwise_convolution(padding='VALID')
            conv2_merge_bn = cnn_merge_2.batch_normoalization(input=conv2_merge,
                                                              is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM*OUT_NUM)
            conv2_merge_relu = tf.nn.relu(conv2_merge_bn)
            # 变换深度卷积后的维度
            conv2_axis_1, conv2_axis_2, conv2_axis_3, conv2_axis_4 = conv2_merge_relu.get_shape().as_list()
            conv2_merge_dropout = tf.nn.dropout(x=conv2_merge_relu, keep_prob=CONV_KEEP_PROB,
                                                noise_shape=[conv2_axis_1, 1, 1, conv2_axis_4])
            conv2_output = cnn_merge_2.reshape(f_vector=conv2_merge_dropout,
                                               new_shape=(
                                               conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM, OUT_NUM, T))
            conv2_output = tf.reduce_sum(conv2_output, axis=3)  # 此处可以试axis= 4
            conv2_output = cnn_merge_2.reshape(f_vector=conv2_output, new_shape=(
                conv2_axis_1, conv2_axis_2, conv2_axis_3, OUT_NUM * T))  # (batch_size, d, 2f-, T*OUT_NUM)

            cnn_merge_3 = CNN(x=conv2_output, w_conv=conv_para_2['w3'], stride_conv=[1, 1, CONV_MEG_3, 1],
                              stride_pool=None)  # x:(batch_size, d, 2f-, T*OUT_NUM)
            conv3_merge = cnn_merge_3.depthwise_convolution(padding='VALID')
            conv3_merge_bn = cnn_merge_3.batch_normoalization(input=conv3_merge,
                                                              is_training=is_training)  # input:(batch_size, d, 2f-, T*OUT_NUM*OUT_NUM)
            conv3_merge_relu = tf.nn.relu(conv3_merge_bn)
            # 变换深度卷积后的维度
            conv3_axis_1, conv3_axis_2, conv3_axis_3, conv3_axis_4 = conv3_merge_relu.get_shape().as_list()
            conv3_merge_dropout = tf.nn.dropout(x=conv3_merge_relu, keep_prob=CONV_KEEP_PROB,
                                                noise_shape=(conv3_axis_1, 1, 1, conv3_axis_4))
            conv3_output = cnn_merge_3.reshape(f_vector=conv3_merge_dropout,
                                               new_shape=(
                                               conv3_axis_1, conv3_axis_2, conv3_axis_3, OUT_NUM, OUT_NUM, T))
            conv3_merge_output = tf.reduce_sum(conv3_output,
                                               axis=3)  # 此处可以试axis=4 shape= (batch_size, d, 2f, OUT_NUM, T)
            conv3_merge_flat = cnn_merge_3.reshape(f_vector=conv3_merge_output,
                                                   new_shape=(-1, conv3_axis_2 * conv3_axis_3 * OUT_NUM * T))

            # multiGRU部分
            gru = RNN(x=conv3_merge_flat, max_time=T, num_units=128)
            rnn_outputs, _ = gru.dynamic_rnn(style='GRU',
                                             output_keep_prob=0.8)  # rnn_outputs:(batch_size, T, num_units)
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
        for epoch in range(100000):
            try:
                while not coord.should_stop():  # 如果线程应该停止则返回True
                    # feature_batch_, target_batch_ = sess.run([feature_batch, label_batch])
                    # print(feature_batch_.shape, target_batch_.shape)
                    _ = sess.run(optimizer, feed_dict={is_training: True})
                    if epoch % 100 == 0:
                        acc = sess.run(acc, feed_dict={is_training: False})
                        print(acc)






















if __name__ == '__main__':
    pass