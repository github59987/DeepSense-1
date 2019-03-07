#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: ceshi
@time: 2019/3/3 11:13
@desc:
'''
import tensorflow as tf
import numpy as np
from tensorflow_own.Routine_operation import SaveFile, LoadFile
from tensorflow_own.TFrecord_operation import FileoOperation







if __name__ == '__main__':
    # 建立TFRecord文件对象
    fileoperation = FileoOperation(
        ftype=tf.float64,  # tf.float64
        ttype=tf.float64,
        fshape=(3, 200, 10, 2),
        tshape=(6),
        batch_size=50,
        capacity=600 + 60 * 50,
        batch_fun='shuffle'
    )
    file_train = r'F:\DeepSenseing\deepsense DataSet\output.tfrecords-*'
    feature_batch, label_batch = fileoperation.ParseDequeue(files=file_train)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        # 线程调配管理器
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph.
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_steps = 10
        try:
            while not coord.should_stop():  # 如果线程应该停止则返回True
                cur_feature_batch, cur_target_batch = sess.run([feature_batch, label_batch])
                print(cur_feature_batch, cur_target_batch, '第一个线程', cur_target_batch.shape)

                train_steps -= 1
                if train_steps <= 0:
                    coord.request_stop()  # 请求该线程停止，若执行则使得coord.should_stop()函数返回True

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            # When done, ask the threads to stop. 请求该线程停止
            coord.request_stop()
            # And wait for them to actually do it. 等待被指定的线程终止
            coord.join(threads)

