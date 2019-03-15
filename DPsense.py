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
from tensorflow_own.Routine_operation import SaveFile, LoadFile, Summary_Visualization, SaveRestore_model, Databatch
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
epoch = 10000
DATASET_NUM = 12000
BATCH_SIZE = 100

def sub_conv_bn_dropout(CNN_object, op, is_training, strides):
    '''
    对单层数据进行卷积、bn、dropout处理
    :param CNN_object: CNN对象节点
    :param op: Tensor, 待处理张量
    :param is_training: placeholder, 指示当前状态
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

def deepsense_model(is_training, feature_batch, summary_visalization=None):
    '''
    deepsense模型
    :param is_training: placeholder, 标记模型是否正在训练
    :param feature_batch: placeholder, 一个批次的特征矩阵
    :param summary_visalization: default:None, 文件摘要对象
    :return: deepsense模型最终结果和标签
    '''
    with tf.name_scope('input_dataset'):
        feature_batch = tf.cast(feature_batch, tf.float32)
        # print(train_feature_batch.shape, test_feature.shape)

        # print(feature_batch.shape)
        # print(feature_batch.shape) #shape=(batch_size, 9*2*F*T)
        # 将单个feature_batch矩阵flat、按单个特征向量长度为9*2*F进行重组
        feature_batch_flat = tf.reshape(tensor=feature_batch, shape=(1, -1), name='feature_batch_flat')
        feature_batch_new = tf.reshape(tensor=feature_batch_flat, shape=(-1, 9 * 2 * F),
                                       name='feature_batch_new')
        # print('数据按照9*2*F特征维度重组后数据维度为: %s' % feature_batch_new.shape) #shape=(batch_size*T, 9*2*F)
        # 按传感器类比分为三组数据
        input_acc, input_gry, input_mag = tf.split(value=feature_batch_new, num_or_size_splits=3, axis=1)
        # print('各个传感器一个批次的数据维度为: %s %s %s' % (input_acc.shape, input_gry.shape, input_mag.shape))

    with tf.name_scope('cnn_part1'):
        with tf.name_scope('conv_para_1'):
            conv_para_1 = {
                'w1': tf.Variable(
                    initial_value=tf.truncated_normal(shape=(3, 2 * 3 * CONV_LEN_1, 1, OUT_NUM), mean=0,
                                                      stddev=1),
                    dtype=tf.float32),
                'w2': tf.Variable(
                    initial_value=tf.truncated_normal(shape=(1, CONV_LEN_2, OUT_NUM, OUT_NUM), mean=0,
                                                      stddev=1),
                    dtype=tf.float32),
                'w3': tf.Variable(
                    initial_value=tf.truncated_normal(shape=(1, CONV_LEN_3, OUT_NUM, OUT_NUM), mean=0,
                                                      stddev=1),
                    dtype=tf.float32),
            }
            summary_visalization.variable_summaries(var=conv_para_1['w1'], name='part1_w1')
            summary_visalization.variable_summaries(var=conv_para_1['w2'], name='part1_w2')
            summary_visalization.variable_summaries(var=conv_para_1['w3'], name='part1_w3')
        # acc部分
        cnn_acc_1 = CNN(x=None, w_conv=conv_para_1['w1'], stride_conv=None,
                        stride_pool=None)  # x:(batch_size*T, 3*2*F)
        input_acc_reshape = cnn_acc_1.reshape(f_vector=input_acc, new_shape=(-1, 3, 2 * F, 1))
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
        conv1_gry = sub_conv_bn_dropout(CNN_object=cnn_gry_1, op=input_gry_reshape, is_training=is_training,
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
        conv3_acc_flat = tf.reshape(tensor=conv3_acc, shape=(-1, axis[1] * axis[2] * axis[3]))
        conv3_gry_flat = tf.reshape(tensor=conv3_gry, shape=(-1, axis[1] * axis[2] * axis[3]))
        # conv3_mag_flat = tf.reshape(tensor=conv3_mag, shape=(-1, axis[1]*axis[2]*axis[3]))
        # 分别对各个flat后的批次张量增加维并组合为4维
        conv3_acc_fin = tf.expand_dims(input=conv3_acc_flat, axis=1)
        conv3_gry_fin = tf.expand_dims(input=conv3_gry_flat, axis=1)
        # conv3_mag_fin = tf.expand_dims(input=conv3_mag_flat, axis=1)
        cnn_concat1 = tf.concat([conv3_acc_fin, conv3_gry_fin], axis=1)
        # cnn_concat1 = tf.concat([conv3_acc_fin, conv3_gry_fin, conv3_mag_fin], axis=1)
        cnn_concat1 = tf.expand_dims(input=cnn_concat1, axis=-1)
        # print('第一部分卷积层输出单个批次维度为: %s' % cnn_concat1.shape)  # shape=(batch_size*T, 2(3), :, 1)

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

        cnn_merge_1 = CNN(x=None, w_conv=conv_para_2['w1'], stride_conv=None,
                          stride_pool=None)  # x:(batch_size*T, 2(3), :)
        conv1_merge = sub_conv_bn_dropout(CNN_object=cnn_merge_1, op=cnn_concat1, is_training=is_training,
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
        # 对卷积层的输出在后三个维度上进行flat并组合WIDTH参数在各个特征向量的最后
        axis_cnn = cnn_fin.get_shape().as_list()
        cnn_flat = tf.reshape(tensor=cnn_fin,
                              shape=(-1, axis_cnn[1] * axis_cnn[2] * axis_cnn[3]))  # (batch_size*T, :)
        const_WIDTH = tf.constant(value=np.ones(shape=(cnn_flat.get_shape().as_list()[0], 1)) * WIDTH,
                                  dtype=tf.float32)
        cnn_WIDTH = tf.concat([cnn_flat, const_WIDTH], axis=1)
        # print('组合WIDTH后的维度为: %s' % cnn_WIDTH.shape)
    with tf.name_scope('rnn'):
        # 对第二部分卷积输出做维度转换，转换后为:(batch_size, T, :)
        axis_cnn_WIDTH = cnn_WIDTH.get_shape().as_list()
        cnn_WIDTH_flat = tf.reshape(tensor=cnn_WIDTH, shape=(1, axis_cnn_WIDTH[0] * axis_cnn_WIDTH[1]))
        cnn_output = tf.reshape(tensor=cnn_WIDTH_flat,
                                shape=(-1, T * axis_cnn_WIDTH[1]))  # (batch_size, T*:)
        cnn_output = tf.reshape(tensor=cnn_output, shape=(-1, axis_cnn_WIDTH[1], T))  # (batch_size, :, T)
        rnn_input = tf.transpose(a=cnn_output, perm=[0, 2, 1])  # (batch_size, T, :)
        # multiGRU部分
        gru = RNN(x=rnn_input, max_time=T, num_units=128)
        rnn_outputs, _ = gru.dynamic_multirnn(style='GRU', layers_num=2,
                                              output_keep_prob=0.8)  # rnn_outputs:(batch_size, T, num_units)
        # print(rnn_outputs.get_shape().as_list())
        # 将T个时刻的所有输出求平均值
        rnn_output = tf.reduce_mean(rnn_outputs, axis=1)  # rnn_output:(batch_size, num_units)
        # print(rnn_output.get_shape().as_list())
    with tf.name_scope('output_layer'):
        w_size_in = rnn_outputs.get_shape().as_list()[-1]
        w_size_out = 6
        w_out = tf.Variable(tf.truncated_normal(shape=(w_size_in, w_size_out), mean=0, stddev=1),
                            dtype=tf.float32)
        b = tf.Variable(tf.truncated_normal(shape=([w_size_out]), mean=0, stddev=1), dtype=tf.float32)
        summary_visalization.variable_summaries(var=w_out, name='fc_w')
        summary_visalization.variable_summaries(var=b, name='fc_b')
        fin_output = tf.matmul(rnn_output, w_out) + b
    # print(fin_output.shape)
    return fin_output

def model_training(training_time):
    '''
    DeepSense模型
    :param training_time: 标记当前是第几轮训练
    :return: None
    '''
    ds_graph = tf.Graph()
    with ds_graph.as_default():
        # 建立摘要对象
        summary_visalization = Summary_Visualization()
        with tf.name_scope('training_dataset'):
            train_feature_, train_label_ = LoadFile(p=r'F:\\DeepSenseing\deepsense DataSet\train.pickle')
        with tf.name_scope('testing_dataset'):
            test_feature_, test_label_ = LoadFile(p=r'F:\\DeepSenseing\deepsense DataSet\test.pickle')
        with tf.name_scope('placeholder'):
            feature_batch = tf.placeholder(shape=(BATCH_SIZE, 18000), dtype=tf.float32, name='feature_batch')
            label_batch = tf.placeholder(shape=(BATCH_SIZE, 6), dtype=tf.float32, name='label_batch')
            is_training = tf.placeholder(tf.bool, name='is_training')
        #导入模型
        print('正在导入模型')
        fin_output = deepsense_model(is_training=is_training,
                                                  feature_batch=feature_batch,
                                                  summary_visalization=summary_visalization)
        with tf.name_scope('evaluation'):
            # 定义softmax交叉熵和损失函数以及精确度函数
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fin_output, labels=label_batch))
            # 添加摘要loss
            summary_visalization.scalar_summaries(arg={'loss': loss})
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(loss)
            evaluation = Evaluation(one_hot=True, logit=tf.nn.softmax(fin_output), label=label_batch,
                                    regression_pred=None, regression_label=None)
            acc = evaluation.acc_classification()
        with tf.name_scope('etc'):
            merge = summary_visalization.summary_merge()
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=ds_graph) as sess:
        sess.run(init_global)
        sess.run(init_local)
        # 建立checkpoint节点保存对象
        saverestore_model = SaveRestore_model(sess=sess, save_file_name='deepsense', max_to_keep=1)
        saver = saverestore_model.saver_build()
        if training_time != 0:
            # 导入checkpoint节点，继续训练
            saverestore_model.restore_checkpoint(saver=saver)
        # 摘要文件
        summary_writer = summary_visalization.summary_file(p='logs/', graph=sess.graph)
        for i in range(epoch):
            flag = 1
            acc_ALL, n = 0, 1
            for batch_feature, batch_label in Databatch((train_feature_, train_label_), batch_size=100):
                summary = sess.run(merge, feed_dict={is_training: True,
                                                     feature_batch: batch_feature,
                                                     label_batch: batch_label})
                _ = sess.run(optimizer, feed_dict={is_training: True,
                                                   feature_batch: batch_feature,
                                                   label_batch: batch_label})
                summary_visalization.add_summary(summary_writer=summary_writer, summary=summary,
                                                 summary_information=(i))
                if (i % 100 == 0) and (flag == 1):
                    flag = 0
                    loss_ = sess.run(loss, feed_dict={is_training: True,
                                                      feature_batch: batch_feature,
                                                      label_batch: batch_label})
                    print('第 %s 轮训练集误差为: %s' % (i, loss_))
            if i % 100 == 0: #100的倍数轮输出
                for batch_test_feature, batch_test_label in Databatch((test_feature_, test_label_), batch_size=100):
                    acc_ = sess.run(acc, feed_dict={is_training: False,
                                                    feature_batch: batch_test_feature,
                                                    label_batch: batch_test_label})
                    acc_ALL = ((n - 1) * acc_ALL + acc_) / n
                    n += 1
                    # 保存checkpoint节点
                    saverestore_model.save_checkpoint(saver=saver, epoch=i, is_recording_max_acc=False)
                print('第 %s 轮测试集精度为: %s' % (i, acc_ALL))
        summary_visalization.summary_close(summary_writer=summary_writer)
    #计算最终的分类度量
    PRF_dict = evaluation.PRF_tables(mode_num=6)
    print(PRF_dict)


    #     # 线程调配管理器
    #     coord = tf.train.Coordinator()
    #     # Starts all queue runners collected in the graph.
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     train_steps = TRAIN_STEPS
    #     acc_ALL, n = 0, 1 #测试集准确率
    #     try:
    #         while not coord.should_stop():  # 如果线程应该停止则返回True
    #             # feature_batch_, target_batch_ = sess.run([feature_batch, label_batch])
    #             # print(feature_batch_.shape, target_batch_.shape)
    #             summary = sess.run(merge, feed_dict={is_training: True,
    #                                                  feature_batch: np.arange(18000*50).reshape(50, 18000),
    #                                                  label_batch: np.arange(6*50).reshape(50, 6)})
    #             _ = sess.run(optimizer, feed_dict={is_training: True,
    #                                                feature_batch:np.arange(18000*50).reshape(50, 18000),
    #                                                label_batch:np.arange(6*50).reshape(50, 6)})
    #             train_steps -= 1
    #             if train_steps <= 0:
    #                 coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True
    #             if (TRAIN_STEPS - train_steps) % (TRAIN_STEPS//BATCH_SIZE * 50) == 0:  # 读入批次总数
    #                 for batch_test_feature, batch_test_label in Databatch((test_feature, test_label), 50):
    #                     acc_ = sess.run(acc, feed_dict={is_training:False,
    #                                                     test_feature:batch_test_feature, test_label:batch_test_label})
    #                     acc_ALL =  ((n-1)*acc_ALL + acc_) / n
    #                 print(acc_ALL)
    #                 # 保存checkpoint节点
    #                 saverestore_model.save_checkpoint(saver=saver, epoch=TRAIN_STEPS//BATCH_SIZE,
    #                                                   is_recording_max_acc=False)
    #             summary_visalization.add_summary(summary_writer=summary_writer, summary=summary,
    #                                              summary_information=(TRAIN_STEPS - train_steps + 1))
    #     except tf.errors.OutOfRangeError:
    #         print('%s次训练完成' % (TRAIN_STEPS // 12))
    #     finally:
    #         # When done, ask the threads to stop. 请求该线程停止
    #         coord.request_stop()
    #         # And wait for them to actually do it. 等待被指定的线程终止
    #         coord.join(threads)
    #         summary_visalization.summary_close(summary_writer=summary_writer)
    # #计算最终的分类度量
    # PRF_dict = evaluation.PRF_tables(mode_num=6)
    # print(PRF_dict)

if __name__ == '__main__':
    model_training(training_time=0)
